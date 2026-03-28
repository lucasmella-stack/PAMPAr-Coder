#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr Neuro-Trainer — Entrenamiento correctivo con losses auxiliares.

Diagnóstico del Brain Scanner reveló 3 problemas en PamparV3:
  1. Tálamo no diferencia — TODOS los tokens van a SINTAXIS (~65-100%)
  2. Early Exit nunca activa — confianza máxima 48% (necesita 90%)
  3. Streams sub-utilizados — la especialización interna no se aprovecha

Este entrenador agrega 3 losses auxiliares para corregir sin cambiar la arquitectura:
  - loss_routing: Supervisión directa con LLAVES → CE contra territorio correcto
  - loss_exit:    Calibra la confianza del Early Exit
  - loss_balance: Previene streams muertos

la loss_routing usa clasificar_token() para determinar el territorio correcto
de cada token, y entrena el routing con CE → gradiente fuerte y direccional
que rompe la simetría del routing uniforme.

Uso:
  python scripts/neuro_trainer.py
  python scripts/neuro_trainer.py --data data/sft_v5.jsonl --pasos 500
  python scripts/neuro_trainer.py --alpha-diff 0.2 --alpha-exit 0.1
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import sentencepiece as spm

# Proyecto
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PamparV3, ConfigV3, PRESET_V3
from pampar.coder.v3.talamo import TalamoInicial
from pampar.coder.v3.llaves import clasificar_token
from pampar.coder.v3.zonas import ZONA_TERRITORIO

logger = logging.getLogger("neuro_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

STREAM_NAMES = ["SINTAXIS", "SEMANTICA", "LOGICO", "ESTRUCTURAL"]
_MARCADORES = ["### Solution:", "### Protocolo:"]


# ─────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────

def _cargar_datos(ruta: Path) -> list[str]:
    """Carga textos desde JSONL."""
    textos: list[str] = []
    with open(ruta, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texto = obj.get("text", "")
            if texto:
                textos.append(texto)
    logger.info("Cargados %d ejemplos desde %s", len(textos), ruta.name)
    return textos


def _tokenizar_con_mascara(
    textos: list[str],
    tokenizer: spm.SentencePieceProcessor,
    max_len: int,
) -> list[tuple[list[int], list[bool]]]:
    """Tokeniza y crea máscara: loss solo en porción Solution."""
    chunks: list[tuple[list[int], list[bool]]] = []
    for texto in textos:
        ids_full = tokenizer.Encode(texto)
        if len(ids_full) < 8:
            continue
        ids_full = ids_full[: max_len + 1]

        n_prefijo = len(ids_full)
        for marcador in _MARCADORES:
            pos = texto.find(marcador)
            if pos >= 0:
                ids_pre = tokenizer.Encode(texto[: pos + len(marcador)])
                n_prefijo = min(len(ids_pre), len(ids_full))
                break

        mascara = [False] * n_prefijo + [True] * (len(ids_full) - n_prefijo)
        chunks.append((ids_full, mascara))
    return chunks


def _hacer_batch(
    chunks: list[tuple[list[int], list[bool]]],
    indices: list[int],
    device: torch.device,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch con padding y máscara de loss."""
    sels = [chunks[i] for i in indices]
    max_len = min(max(len(ids) for ids, _ in sels), max_seq_len + 1)
    padded_ids: list[list[int]] = []
    padded_mask: list[list[bool]] = []
    for ids, mask in sels:
        t = ids[:max_len]
        m = mask[:max_len]
        pad_len = max_len - len(t)
        padded_ids.append(t + [0] * pad_len)
        padded_mask.append(m + [False] * pad_len)
    tokens = torch.tensor(padded_ids, dtype=torch.long, device=device)
    mascara = torch.tensor(padded_mask, dtype=torch.bool, device=device)
    return tokens, mascara


# ─────────────────────────────────────────────────────────────────
# Forward Instrumentado (con gradientes, sin checkpointing)
# ─────────────────────────────────────────────────────────────────

def forward_neuro(
    model: PamparV3,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    """
    Forward que captura datos por nivel para las losses auxiliares.

    NO usa gradient checkpointing → guarda todas las activaciones.
    Usar con seq_len cortas y batch=1 para caber en 4GB VRAM.

    Returns:
        logits [B, L, V], dict con datos por nivel
    """
    config = model.config
    n_streams = config.n_streams

    x = model.emb_drop(model.tok_emb(input_ids))
    terr_acts, zona_acts_n0 = model.talamo(x, input_ids)
    streams = [x.clone() for _ in range(n_streams)]

    all_terr_acts: list[torch.Tensor] = [terr_acts]
    all_conf_tensors: list[torch.Tensor] = []
    all_stream_norms: list[torch.Tensor] = []
    all_x_out: list[torch.Tensor] = []

    for i, nivel in enumerate(model.niveles):
        streams, terr_acts, _ = nivel(streams, terr_acts, TalamoInicial.agregar_fn)
        all_terr_acts.append(terr_acts)

        # Recomputar confianza per-token CON gradientes
        # (dentro de nivel.forward() se calcula pero se detacha con .item())
        x_out = sum(
            streams[t] * terr_acts[:, :, t : t + 1] for t in range(n_streams)
        )
        conf = torch.sigmoid(nivel.exit_head(x_out)).squeeze(-1)  # [B, L]
        all_conf_tensors.append(conf)
        all_x_out.append(x_out)

        # Norma L2 promedio por stream (diferenciable)
        norms = torch.stack(
            [streams[t].norm(dim=-1).mean() for t in range(n_streams)]
        )
        all_stream_norms.append(norms)

    # Combinación final + logits
    x_final = model._combinar_streams(streams, terr_acts)
    x_final = model.norm_f(x_final)
    logits = model.lm_head(x_final)

    return logits, {
        "all_terr_acts": all_terr_acts,  # [n_levels+1] × [B, L, 4]
        "all_conf_tensors": all_conf_tensors,  # [n_levels] × [B, L]
        "all_stream_norms": all_stream_norms,  # [n_levels] × [4]
        "all_x_out": all_x_out,  # [n_levels] × [B, L, D]
        "zona_acts_n0": zona_acts_n0,  # [B, L, 52] — zone acts from TalamoInicial
    }


# ─────────────────────────────────────────────────────────────────
# Losses Auxiliares
# ─────────────────────────────────────────────────────────────────

def _build_territory_table(
    tokenizer: spm.SentencePieceProcessor,
) -> torch.Tensor:
    """
    Construye lookup table: token_id → territorio target (0-3).

    Usa clasificar_token() de LLAVES para determinar la zona de cada
    token del vocabulario, y ZONA_TERRITORIO para mapear a territorio.
    Se ejecuta UNA vez al inicio (~48K tokens, <2s).
    """
    vocab_size = tokenizer.GetPieceSize()
    table = torch.zeros(vocab_size, dtype=torch.long)
    for token_id in range(vocab_size):
        piece = tokenizer.IdToPiece(token_id)
        zona, _conf = clasificar_token(piece)
        table[token_id] = ZONA_TERRITORIO[zona].value
    return table


def _build_zone_table(
    tokenizer: spm.SentencePieceProcessor,
) -> torch.Tensor:
    """Lookup table: token_id → zone index (0-51) para loss a nivel de zona."""
    vocab_size = tokenizer.GetPieceSize()
    table = torch.zeros(vocab_size, dtype=torch.long)
    for token_id in range(vocab_size):
        piece = tokenizer.IdToPiece(token_id)
        zona, _conf = clasificar_token(piece)
        table[token_id] = zona.value - 1  # 0-indexed (0-51)
    return table


def calcular_loss_talamo(
    all_terr_acts: list[torch.Tensor],
    zona_acts_n0: torch.Tensor,
    input_ids: torch.Tensor,
    territory_table: torch.Tensor,
    zone_table: torch.Tensor,
) -> torch.Tensor:
    """
    Loss enfocada en TalamoInicial: CE a nivel de zona + CE a nivel de territorio.

    La zona-level CE bypasses la dilución del MEAN en agregar_zonas_a_territorios,
    dando gradiente directo a attn_proj y context_conv para activar las zonas
    correctas. Sin esto, el gradiente a través de MEAN(15 zonas) es ~0.013x.
    """
    device = input_ids.device

    # CE sobre territorio en N0 (4-class)
    targets_terr = territory_table.to(device)[input_ids]
    terr_acts_n0 = all_terr_acts[0]
    loss_terr = F.cross_entropy(
        terr_acts_n0.reshape(-1, 4), targets_terr.reshape(-1)
    )

    # CE sobre zonas en N0 (52-class) — bypasses MEAN dilution
    targets_zona = zone_table.to(device)[input_ids]
    loss_zona = F.cross_entropy(
        zona_acts_n0.reshape(-1, 52), targets_zona.reshape(-1)
    )

    return loss_terr + 2.0 * loss_zona


def calcular_loss_routing(
    all_terr_acts: list[torch.Tensor],
    input_ids: torch.Tensor,
    territory_table: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised routing loss: CE contra el territorio correcto de LLAVES.

    Cada token tiene un territorio correcto determinado por clasificar_token().
    Usa class weights inversamente proporcionales a la frecuencia para compensar
    el desbalance masivo SEMA=47751 vs SINT=169 vs LOGI=72 vs ESTR=8.
    Sin esto, el gradiente para SINT/LOGI tokens es demasiado débil.
    """
    device = input_ids.device
    # SINT=169, SEMA=47751, LOGI=72, ESTR=8 → weights= sqrt(total/count), capped at 10
    weights = torch.tensor(
        [min(10.0, (48000 / 169) ** 0.5),   # SINT:  ~16.9 → 10
         1.0,                                 # SEMA:  ~1.0
         min(10.0, (48000 / 72) ** 0.5),     # LOGI:  ~25.8 → 10
         min(10.0, (48000 / 8) ** 0.5)],     # ESTR:  ~77.5 → 10
        device=device,
        dtype=torch.float32,
    )

    targets = territory_table.to(device)[input_ids]  # [B, L]

    total = torch.tensor(0.0, device=device)
    for ta in all_terr_acts:
        L_ta = ta.shape[1]
        t = targets[:, :L_ta]
        total = total + F.cross_entropy(
            ta.reshape(-1, 4), t.reshape(-1), weight=weights
        )
    return total / len(all_terr_acts)


def calcular_loss_exit(
    all_conf_tensors: list[torch.Tensor],
    all_intermediate_correct: list[torch.Tensor],
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Loss de calibración de Early Exit en TODOS los niveles.

    Para cada nivel: BCE entre su confianza y si predice correctamente
    (usando el LM head compartido en representaciones intermedias).
    Niveles profundos tienen más peso (predicen mejor).
    """
    device = all_conf_tensors[0].device
    total = torch.tensor(0.0, device=device)
    n_levels = len(all_conf_tensors)
    valid = loss_mask.float()
    n_valid = valid.sum().clamp(min=1)

    # — Calibración en CADA nivel (peso creciente con profundidad) —
    for i in range(n_levels):
        conf = all_conf_tensors[i]  # [B, L]
        correct = all_intermediate_correct[i]  # [B, L]
        weight = (i + 1) / n_levels  # 0.2, 0.4, 0.6, 0.8, 1.0
        total = total + weight * F.binary_cross_entropy(
            conf.clamp(1e-7, 1 - 1e-7) * valid,
            correct * valid,
            reduction="sum",
        ) / n_valid

    # — Monotonía: confianza debe subir con profundidad —
    for i in range(1, n_levels):
        violation = all_conf_tensors[i - 1] - all_conf_tensors[i]
        total = total + F.relu(violation).mean()

    return total


def calcular_loss_balance(all_stream_norms: list[torch.Tensor]) -> torch.Tensor:
    """
    Loss de utilización de streams.

    Penaliza el stream con menor norma → previene streams muertos.
    Todos los streams deben contribuir, no solo SINTAXIS.
    """
    total = torch.tensor(0.0, device=all_stream_norms[0].device)
    for norms in all_stream_norms:
        min_norm = norms.min()
        total = total - torch.log(min_norm.clamp(min=1e-8))
    return total / len(all_stream_norms)


# ─────────────────────────────────────────────────────────────────
# Training Step
# ─────────────────────────────────────────────────────────────────

def paso_neuro(
    modelo: PamparV3,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    mascara: torch.Tensor,
    max_grad_norm: float,
    alpha_diff: float,
    alpha_exit: float,
    alpha_balance: float,
    territory_table: torch.Tensor,
    warmup_factor: float = 1.0,
) -> dict[str, float]:
    """Un paso de entrenamiento con CE + 3 losses auxiliares."""
    modelo.train()
    optimizer.zero_grad(set_to_none=True)

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    loss_mask = mascara[:, 1:]

    # Forward instrumentado (sin checkpointing)
    logits, neuro_info = forward_neuro(modelo, input_ids)

    B, T, V = logits.shape

    # CE Loss (solo en solución)
    targets_masked = targets.masked_fill(~loss_mask, -100)
    loss_ce = F.cross_entropy(
        logits.reshape(B * T, V),
        targets_masked.reshape(B * T),
        ignore_index=-100,
    )

    # Losses auxiliares (rampean con warmup_factor)
    loss_diff = calcular_loss_routing(
        neuro_info["all_terr_acts"], input_ids, territory_table
    )

    # Intermediate correctness para calibrar exit en TODOS los niveles
    with torch.no_grad():
        all_intermediate_correct: list[torch.Tensor] = []
        for x_out in neuro_info["all_x_out"]:
            inter_logits = modelo.lm_head(modelo.norm_f(x_out))
            inter_pred = inter_logits.argmax(dim=-1)
            correct = (inter_pred == targets_masked).float()
            all_intermediate_correct.append(correct)

    loss_exit = calcular_loss_exit(
        neuro_info["all_conf_tensors"], all_intermediate_correct, loss_mask
    )
    loss_balance = calcular_loss_balance(neuro_info["all_stream_norms"])

    wf = warmup_factor
    loss = (
        loss_ce
        + wf * alpha_diff * loss_diff
        + wf * alpha_exit * loss_exit
        + wf * alpha_balance * loss_balance
    )

    if loss.isnan() or loss.isinf():
        logger.warning("Loss inestable — skipping step")
        return {"ce": 0.0, "diff": 0.0, "exit": 0.0, "balance": 0.0, "total": 0.0}

    loss.backward()
    nn_utils.clip_grad_norm_(modelo.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "ce": float(loss_ce.detach()),
        "diff": float(loss_diff.detach()),
        "exit": float(loss_exit.detach()),
        "balance": float(loss_balance.detach()),
        "total": float(loss.detach()),
    }


# ─────────────────────────────────────────────────────────────────
# Diagnóstico Periódico
# ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def diagnostico(
    modelo: PamparV3,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    territory_table: torch.Tensor | None = None,
) -> str:
    """Mini-scan con frase de test para ver si las losses están funcionando."""
    modelo.eval()

    test_code = "def fibonacci(n):"
    ids = tokenizer.Encode(test_code)
    input_ids = torch.tensor([ids], device=device)

    _, info = forward_neuro(modelo, input_ids)

    lines = ["  ── Diagnóstico ──"]

    # Routing diversity: std de terr_acts por nivel
    for lvl, ta in enumerate(info["all_terr_acts"]):
        std_mean = torch.std(ta, dim=-1).mean().item()
        lines.append(f"    Nivel {lvl} routing std: {std_mean:.4f}")

    # Confianza por nivel
    confs = [c.mean().item() for c in info["all_conf_tensors"]]
    conf_str = " → ".join(f"{c:.3f}" for c in confs)
    lines.append(f"    Confianza: {conf_str}")

    # Stream balance: normas del último nivel
    last_norms = info["all_stream_norms"][-1]
    norm_str = " | ".join(
        f"{STREAM_NAMES[t][:4]}={last_norms[t]:.1f}" for t in range(4)
    )
    lines.append(f"    Streams: {norm_str}")

    # Routing dominante (actual)
    final_ta = info["all_terr_acts"][-1]  # [1, L, 4]
    dom_counts = [0, 0, 0, 0]
    for i in range(final_ta.shape[1]):
        dom = final_ta[0, i].argmax().item()
        dom_counts[dom] += 1
    dom_str = " | ".join(
        f"{STREAM_NAMES[t][:4]}={dom_counts[t]}" for t in range(4)
    )
    lines.append(f"    Actual:   {dom_str}")

    # Routing esperado (LLAVES target)
    if territory_table is not None:
        expected = territory_table[input_ids[0].cpu()]
        exp_counts = [0, 0, 0, 0]
        for e in expected:
            exp_counts[e.item()] += 1
        exp_str = " | ".join(
            f"{STREAM_NAMES[t][:4]}={exp_counts[t]}" for t in range(4)
        )
        lines.append(f"    Esperado: {exp_str}")

    modelo.train()
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────

def _cosine_lr(
    paso: int, warmup: int, total: int, lr_max: float, lr_min: float
) -> float:
    if paso < warmup:
        return lr_max * (paso + 1) / warmup
    progreso = (paso - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progreso))


def _guardar(
    ruta: Path, modelo: PamparV3, optimizer: torch.optim.Optimizer, paso: int
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "paso_global": paso,
            "config": dataclasses.asdict(modelo.config),
            "tipo": "neuro_trainer",
        },
        ruta,
    )
    logger.info("Checkpoint guardado → %s (paso %d)", ruta.name, paso)


# ─────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PAMPAr Neuro-Trainer: corrige routing, exit y balance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-in",
        type=Path,
        default=ROOT / "checkpoints" / "v3_sft_v8.pt",
    )
    p.add_argument(
        "--checkpoint-out",
        type=Path,
        default=ROOT / "checkpoints" / "v3_neuro_v1.pt",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "data" / "tokenizer" / "pampar_48k.model",
    )
    p.add_argument("--data", type=Path, default=ROOT / "data" / "sft_v5.jsonl")

    p.add_argument("--lr", type=float, default=3e-7,
                   help="LR para params principales (muy bajo, proteger CE)")
    p.add_argument("--lr-min", type=float, default=3e-8)
    p.add_argument("--lr-routing", type=float, default=1e-3,
                   help="LR para params de routing (talamo, lateral.scale)")
    p.add_argument("--lr-routing-min", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-pasos", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1,
                   help="Batch=1 (4GB VRAM, sin checkpointing)")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Secuencia corta para caber sin checkpointing")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--guardar-cada", type=int, default=200)

    # Pesos de losses auxiliares
    p.add_argument("--alpha-diff", type=float, default=0.5,
                   help="Peso loss_routing (CE supervisado contra LLAVES)")
    p.add_argument("--alpha-exit", type=float, default=0.0,
                   help="Peso loss_exit (desactivado hasta resolver routing)")
    p.add_argument("--alpha-balance", type=float, default=0.05,
                   help="Peso loss_balance (utilización streams)")
    p.add_argument("--aux-warmup", type=int, default=50,
                   help="Pasos para rampear aux losses de 0→1")

    # Modo Focus-Talamo (Round 6)
    p.add_argument("--focus-talamo", action="store_true",
                   help="Solo entrena TalamoInicial (attn_proj, conv, gate)")
    p.add_argument("--lr-talamo", type=float, default=5e-3,
                   help="LR para TalamoInicial en modo focus-talamo")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--diag-cada", type=int, default=50,
                   help="Cada cuántos pasos mostrar diagnóstico")

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    logger.info("Device: %s", device)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        logger.info(
            "GPU: %s (%.1f GiB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # Tokenizer
    tok = spm.SentencePieceProcessor()
    tok.Load(str(args.tokenizer))
    logger.info("Tokenizer vocab=%d", tok.GetPieceSize())

    # Modelo
    if not args.checkpoint_in.exists():
        logger.error("Checkpoint no encontrado: %s", args.checkpoint_in)
        sys.exit(1)

    payload = torch.load(
        args.checkpoint_in, map_location=device, weights_only=False
    )
    config = ConfigV3(**payload["config"]) if "config" in payload else PRESET_V3
    modelo = PamparV3(config).to(device)
    modelo.load_state_dict(payload["modelo"])
    modelo.registrar_tokenizer(tok)
    logger.info(
        "Cargado '%s' (tipo: %s)", args.checkpoint_in.name, payload.get("tipo", "?")
    )
    del payload

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info("PamparV3 %.1fM params", n_params / 1e6)

    # Datos
    if not args.data.exists():
        logger.error("Dataset no encontrado: %s", args.data)
        sys.exit(1)

    textos = _cargar_datos(args.data)
    chunks = _tokenizar_con_mascara(textos, tok, args.seq_len)
    logger.info("Chunks tokenizados: %d (seq_len=%d)", len(chunks), args.seq_len)
    del textos

    pct_solution = sum(sum(m) for _, m in chunks) / max(
        1, sum(len(ids) for ids, _ in chunks)
    )
    logger.info("Porcentaje tokens en zona Solution: %.1f%%", pct_solution * 100)

    # Territory table para supervised routing loss
    territory_table = _build_territory_table(tok)
    counts = [(territory_table == t).sum().item() for t in range(4)]
    logger.info(
        "Territory table: SINT=%d SEMA=%d LOGI=%d ESTR=%d (total=%d)",
        *counts, sum(counts),
    )

    # Diagnóstico inicial
    logger.info("─── DIAGNÓSTICO INICIAL ───")
    print(diagnostico(modelo, tok, device, territory_table))

    # ── MODO FOCUS-TALAMO (Round 6) ──
    if args.focus_talamo:
        logger.info("═══ MODO FOCUS-TALAMO: entrenando solo TalamoInicial ═══")
        zone_table = _build_zone_table(tok)

        # Congelar TODO excepto TalamoInicial
        for name, p in modelo.named_parameters():
            if not name.startswith("talamo."):
                p.requires_grad_(False)

        n_trainable = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
        logger.info("Params entrenables: %.1fK (TalamoInicial)", n_trainable / 1e3)

        talamo_params = [p for p in modelo.talamo.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            talamo_params, lr=args.lr_talamo,
            betas=(0.9, 0.95), weight_decay=0.01,
        )

        t0 = time.time()
        losses_buf: deque[float] = deque(maxlen=100)
        paso = 0

        try:
            for epoch in range(args.epochs):
                idx_all = list(range(len(chunks)))
                random.shuffle(idx_all)
                logger.info("── Epoch %d/%d ──", epoch + 1, args.epochs)

                for i in range(0, len(idx_all) - args.batch_size + 1, args.batch_size):
                    batch_idx = idx_all[i : i + args.batch_size]
                    tokens, _ = _hacer_batch(chunks, batch_idx, device, args.seq_len)

                    opt.zero_grad(set_to_none=True)
                    input_ids = tokens[:, :-1]
                    _, info = forward_neuro(modelo, input_ids)

                    loss = calcular_loss_talamo(
                        info["all_terr_acts"], info["zona_acts_n0"],
                        input_ids, territory_table, zone_table,
                    )

                    if loss.isnan() or loss.isinf():
                        continue

                    loss.backward()
                    nn_utils.clip_grad_norm_(modelo.parameters(), args.max_grad_norm)
                    opt.step()

                    losses_buf.append(loss.item())
                    paso += 1

                    if paso % 10 == 0:
                        avg = sum(losses_buf) / len(losses_buf)
                        elapsed = time.time() - t0
                        logger.info(
                            "paso %4d/%d | loss=%.4f  avg=%.4f  (%.1f p/s)",
                            paso, args.max_pasos, loss.item(), avg,
                            paso / elapsed,
                        )

                    if paso % args.diag_cada == 0:
                        print(diagnostico(modelo, tok, device, territory_table))

                    if paso % args.guardar_cada == 0:
                        _guardar(args.checkpoint_out, modelo, opt, paso)

                    if paso >= args.max_pasos:
                        break

                if paso >= args.max_pasos:
                    break

        except KeyboardInterrupt:
            logger.info("Interrumpido — guardando...")

        _guardar(args.checkpoint_out, modelo, opt, paso)
        logger.info("─── DIAGNÓSTICO FINAL (TALAMO) ───")
        print(diagnostico(modelo, tok, device, territory_table))

        elapsed = time.time() - t0
        avg = sum(losses_buf) / max(1, len(losses_buf))
        print(f"\n── Focus-Talamo Completado ──")
        print(f"  Pasos: {paso}")
        print(f"  Tiempo: {int(elapsed // 60)}m{int(elapsed % 60):02d}s")
        print(f"  Loss final (avg): {avg:.4f}")
        print(f"  Checkpoint: {args.checkpoint_out}")
        print(f"\n  Verificar con Brain Scanner:")
        print(
            f'    python -X utf8 scripts/brain_scanner.py --suite --device cuda'
            f' --checkpoint {args.checkpoint_out}'
        )
        return

    # Separar parámetros: routing (LR alto) vs resto (LR bajo)
    routing_names = {"talamo", "talamo_nivel", "lateral"}
    routing_params: list[torch.nn.Parameter] = []
    main_params: list[torch.nn.Parameter] = []
    for name, p in modelo.named_parameters():
        parts = name.split(".")
        if any(rn in parts for rn in routing_names):
            routing_params.append(p)
        else:
            main_params.append(p)

    n_routing = sum(p.numel() for p in routing_params)
    n_main = sum(p.numel() for p in main_params)
    logger.info(
        "Param groups: routing=%.1fK (lr=%.1e) | main=%.1fM (lr=%.1e)",
        n_routing / 1e3, args.lr_routing,
        n_main / 1e6, args.lr,
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": main_params, "lr": args.lr},
            {"params": routing_params, "lr": args.lr_routing},
        ],
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8,
    )

    pasos_por_epoch = max(1, len(chunks) // args.batch_size)
    total_pasos = min(args.max_pasos, args.epochs * pasos_por_epoch)
    logger.info(
        "Neuro-Training: %d pasos | %d chunks | alphas: diff=%.3f exit=%.3f balance=%.3f",
        total_pasos,
        len(chunks),
        args.alpha_diff,
        args.alpha_exit,
        args.alpha_balance,
    )

    paso = 0
    t0 = time.time()
    losses_ce: deque[float] = deque(maxlen=100)
    losses_total: deque[float] = deque(maxlen=100)

    try:
        for epoch in range(args.epochs):
            idx = list(range(len(chunks)))
            random.shuffle(idx)
            logger.info("── Epoch %d/%d ──", epoch + 1, args.epochs)

            for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
                batch_idx = idx[i : i + args.batch_size]
                tokens, mascara = _hacer_batch(
                    chunks, batch_idx, device, args.seq_len
                )

                lr_main = _cosine_lr(paso, args.warmup, total_pasos, args.lr, args.lr_min)
                lr_rout = _cosine_lr(paso, args.warmup, total_pasos, args.lr_routing, args.lr_routing_min)
                optimizer.param_groups[0]["lr"] = lr_main
                optimizer.param_groups[1]["lr"] = lr_rout

                wf = min(1.0, paso / max(1, args.aux_warmup))

                loss_dict = paso_neuro(
                    modelo,
                    optimizer,
                    tokens,
                    mascara,
                    args.max_grad_norm,
                    args.alpha_diff,
                    args.alpha_exit,
                    args.alpha_balance,
                    territory_table,
                    warmup_factor=wf,
                )

                if loss_dict["ce"] > 0:
                    losses_ce.append(loss_dict["ce"])
                    losses_total.append(loss_dict["total"])

                paso += 1

                if paso % 10 == 0:
                    avg_ce = sum(losses_ce) / max(1, len(losses_ce))
                    avg_total = sum(losses_total) / max(1, len(losses_total))
                    elapsed = time.time() - t0
                    logger.info(
                        "paso %4d/%d | CE=%.3f  diff=%.3f  exit=%.3f  bal=%.3f  "
                        "total=%.3f  lr_r=%.1e  (%.1f p/s)",
                        paso,
                        total_pasos,
                        loss_dict["ce"],
                        loss_dict["diff"],
                        loss_dict["exit"],
                        loss_dict["balance"],
                        avg_total,
                        lr_rout,
                        paso / elapsed,
                    )

                if paso % args.diag_cada == 0:
                    print(diagnostico(modelo, tok, device, territory_table))

                if paso % args.guardar_cada == 0:
                    _guardar(args.checkpoint_out, modelo, optimizer, paso)

                if paso >= args.max_pasos:
                    break

            if paso >= args.max_pasos:
                break

    except KeyboardInterrupt:
        logger.info("Interrumpido — guardando...")

    _guardar(args.checkpoint_out, modelo, optimizer, paso)

    # Diagnóstico final
    logger.info("─── DIAGNÓSTICO FINAL ───")
    print(diagnostico(modelo, tok, device, territory_table))

    elapsed = time.time() - t0
    avg_ce = sum(losses_ce) / max(1, len(losses_ce))
    print(f"\n── Neuro-Training Completado ──")
    print(f"  Pasos: {paso}")
    print(f"  Tiempo: {int(elapsed // 3600)}h{int((elapsed % 3600) // 60):02d}m")
    print(f"  Loss CE final (avg100): {avg_ce:.3f}")
    print(f"  PPL final: {math.exp(min(avg_ce, 10)):.1f}")
    print(f"  Checkpoint: {args.checkpoint_out}")
    print(f"\n  Verificar con Brain Scanner:")
    print(
        f'    python -X utf8 scripts/brain_scanner.py --code "def fibonacci(n):" --device cuda'
    )


if __name__ == "__main__":
    main()
