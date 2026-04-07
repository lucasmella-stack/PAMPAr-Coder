#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
PAMPAr Ghidra-Trainer — SFT con monitorización GhidraProbe en tiempo real.

Entrena PamparV3 con el dataset master_sft.jsonl (1253 ejemplos únicos)
monitorizando con GhidraProbe cada N pasos para detectar regresiones
en routing, normas y balance de streams ANTES de que destruyan el modelo.

Lecciones de 18 rounds previos:
  - NO train-norm-clamp (destruye gradientes → Score 85→74)
  - NO alpha-exit (descalibra la confianza)
  - Proteger CE por encima de todo
  - El GhidraProbe detecta problemas que el loss no muestra

Uso:
  python scripts/ghidra_trainer.py
  python scripts/ghidra_trainer.py --max-pasos 800 --probe-cada 50
  python scripts/ghidra_trainer.py --data data/master_sft.jsonl --lr 2e-7
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import json
import logging
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PamparV3, ConfigV3, PRESET_V3
from pampar.coder.v3.talamo import TalamoInicial
from pampar.coder.v3.llaves import clasificar_token
from pampar.coder.v3.zonas import ZONA_TERRITORIO
from pampar.coder.v3.ghidra_probe import GhidraProbe, STREAM_NAMES

logger = logging.getLogger("ghidra_trainer")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

_MARCADORES = ["### Solution:", "### Protocolo:", "### Scan:", "### Response:"]


# ─────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────

def _cargar_datos(ruta: Path) -> list[str]:
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
# Territory Table
# ─────────────────────────────────────────────────────────────────

def _build_territory_table(
    tokenizer: spm.SentencePieceProcessor,
) -> torch.Tensor:
    vocab_size = tokenizer.GetPieceSize()
    table = torch.zeros(vocab_size, dtype=torch.long)
    for token_id in range(vocab_size):
        piece = tokenizer.IdToPiece(token_id)
        zona, _conf = clasificar_token(piece)
        table[token_id] = ZONA_TERRITORIO[zona].value
    return table


# ─────────────────────────────────────────────────────────────────
# Losses (simplificado: CE + routing solo)
# ─────────────────────────────────────────────────────────────────

def forward_neuro(
    model: PamparV3,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict]:
    config = model.config
    n_streams = config.n_streams

    x = model.emb_drop(model.tok_emb(input_ids))
    terr_acts, zona_acts_n0 = model.talamo(x, input_ids)
    streams = [x.clone() for _ in range(n_streams)]

    all_terr_acts: list[torch.Tensor] = [terr_acts]
    all_stream_norms: list[torch.Tensor] = []

    for nivel in model.niveles:
        streams, terr_acts, _ = nivel(streams, terr_acts, TalamoInicial.agregar_fn)
        all_terr_acts.append(terr_acts)

        norms = torch.stack(
            [streams[t].norm(dim=-1).mean() for t in range(n_streams)]
        )
        all_stream_norms.append(norms)

    x_final = model._combinar_streams(streams, terr_acts)
    x_final = model.norm_f(x_final)
    logits = model.lm_head(x_final)

    return logits, {
        "all_terr_acts": all_terr_acts,
        "all_stream_norms": all_stream_norms,
    }


def calcular_loss_routing(
    all_terr_acts: list[torch.Tensor],
    input_ids: torch.Tensor,
    territory_table: torch.Tensor,
    focal_gamma: float = 0.0,
) -> torch.Tensor:
    """Routing loss con focal loss opcional (gamma>0 focaliza en tokens mal ruteados)."""
    device = input_ids.device
    weights = torch.tensor(
        [min(10.0, (48000 / 169) ** 0.5),
         1.0,
         min(10.0, (48000 / 72) ** 0.5),
         min(10.0, (48000 / 8) ** 0.5)],
        device=device, dtype=torch.float32,
    )
    targets = territory_table.to(device)[input_ids]
    total = torch.tensor(0.0, device=device)
    for ta in all_terr_acts:
        L_ta = ta.shape[1]
        t = targets[:, :L_ta]
        if focal_gamma > 0:
            logprobs = F.log_softmax(ta.reshape(-1, 4), dim=-1)
            t_flat = t.reshape(-1)
            p_t = logprobs.exp().gather(1, t_flat.unsqueeze(1)).squeeze(1)
            focal_w = (1.0 - p_t).detach() ** focal_gamma
            per_token = F.nll_loss(logprobs, t_flat, weight=weights, reduction="none")
            total = total + (focal_w * per_token).mean()
        else:
            total = total + F.cross_entropy(
                ta.reshape(-1, 4), t.reshape(-1), weight=weights
            )
    return total / len(all_terr_acts)


def calcular_loss_balance(all_stream_norms: list[torch.Tensor]) -> torch.Tensor:
    """Coeficiente de variación: std/mean. Siempre positivo, bounded."""
    total = torch.tensor(0.0, device=all_stream_norms[0].device)
    for norms in all_stream_norms:
        mean_n = norms.mean()
        std_n = norms.std()
        total = total + std_n / mean_n.clamp(min=1e-8)
    return total / len(all_stream_norms)


# ─────────────────────────────────────────────────────────────────
# GhidraProbe Diagnosis (durante training)
# ─────────────────────────────────────────────────────────────────

@dataclass
class ProbeSnapshot:
    """Métricas clave extraídas del GhidraProbe en un paso."""
    paso: int
    # Normas por nivel [5]
    norms_per_level: list[float]
    # Routing: % de tokens dominados por SEMA en nivel final
    sema_dominance: float
    # Routing diversity: std promedio de terr_acts
    routing_std: float
    # LLAVES utilización
    llaves_nonzero_pct: float
    # Stream balance: ratio min/max norm en nivel final
    stream_balance: float
    # Lateral scales promedio
    lateral_scales_avg: list[float]


@dataclass
class ProbeBaseline:
    """Línea base del modelo pre-entrenamiento para detectar regresiones."""
    norms_per_level: list[float]
    sema_dominance: float
    routing_std: float
    llaves_nonzero_pct: float
    stream_balance: float


PROBE_SENTENCES = [
    "def fibonacci(n):",
    "import os\nfrom pathlib import Path",
    "if x > 0 and y < 10:",
    "class DataLoader:",
]


def _run_probe(
    probe: GhidraProbe,
    modelo: PamparV3,
    tokenizer: spm.SentencePieceProcessor,
    device: torch.device,
    paso: int,
) -> ProbeSnapshot:
    """Ejecuta GhidraProbe sobre frases de test y extrae métricas clave."""
    modelo.eval()

    # Promediar métricas sobre varias frases para estabilidad
    all_norms: list[list[float]] = []
    all_sema_pct: list[float] = []
    all_routing_std: list[float] = []
    all_llaves_pct: list[float] = []
    all_balance: list[float] = []
    all_lat_scales: list[list[float]] = []

    for sentence in PROBE_SENTENCES:
        probe.reset()
        ids = tokenizer.Encode(sentence)
        input_ids = torch.tensor([ids], device=device)

        with torch.no_grad():
            modelo(input_ids)

        cap = probe.report()

        # Normas por nivel
        level_norms = []
        for nc in cap.niveles:
            avg_norm = sum(nc.streams_out_norms) / max(len(nc.streams_out_norms), 1)
            level_norms.append(avg_norm)
        all_norms.append(level_norms)

        # SEMA dominance en nivel final
        if cap.niveles:
            last = cap.niveles[-1]
            terr = last.terr_acts_mean
            total_terr = sum(terr) if terr else 1.0
            sema_pct = (terr[1] / total_terr * 100) if len(terr) > 1 and total_terr > 0 else 0.0
            all_sema_pct.append(sema_pct)

            # Routing std
            if terr:
                mean_t = sum(terr) / len(terr)
                std_t = (sum((v - mean_t) ** 2 for v in terr) / len(terr)) ** 0.5
                all_routing_std.append(std_t)

            # Stream balance
            if last.streams_out_norms:
                mn = min(last.streams_out_norms)
                mx = max(last.streams_out_norms)
                all_balance.append(mn / mx if mx > 0 else 0.0)

            # Lateral scales
            if last.lateral_scales:
                all_lat_scales.append(last.lateral_scales)

        # LLAVES
        if cap.talamo:
            all_llaves_pct.append(cap.talamo.llaves_nonzero_pct)

    modelo.train()

    # Promediar
    n_levels = max(len(n) for n in all_norms) if all_norms else 5
    avg_norms = []
    for lvl in range(n_levels):
        vals = [n[lvl] for n in all_norms if lvl < len(n)]
        avg_norms.append(sum(vals) / max(len(vals), 1))

    avg_lat = []
    if all_lat_scales:
        n_s = len(all_lat_scales[0])
        for s in range(n_s):
            vals = [ls[s] for ls in all_lat_scales if s < len(ls)]
            avg_lat.append(sum(vals) / max(len(vals), 1))

    return ProbeSnapshot(
        paso=paso,
        norms_per_level=avg_norms,
        sema_dominance=sum(all_sema_pct) / max(len(all_sema_pct), 1),
        routing_std=sum(all_routing_std) / max(len(all_routing_std), 1),
        llaves_nonzero_pct=sum(all_llaves_pct) / max(len(all_llaves_pct), 1),
        stream_balance=sum(all_balance) / max(len(all_balance), 1),
        lateral_scales_avg=avg_lat,
    )


def _print_probe_report(
    snap: ProbeSnapshot,
    baseline: Optional[ProbeBaseline] = None,
) -> None:
    """Imprime reporte del GhidraProbe con deltas vs baseline."""
    print(f"\n  {'─' * 60}")
    print(f"  GHIDRA PROBE @ paso {snap.paso}")
    print(f"  {'─' * 60}")

    # Normas por nivel
    norm_parts = []
    for i, n in enumerate(snap.norms_per_level):
        delta = ""
        if baseline and i < len(baseline.norms_per_level):
            d = n - baseline.norms_per_level[i]
            sign = "+" if d >= 0 else ""
            color = "\033[31m" if abs(d) / max(baseline.norms_per_level[i], 1) > 0.2 else "\033[32m"
            delta = f" ({color}{sign}{d:.1f}\033[0m)"
        norm_parts.append(f"N{i}={n:.1f}{delta}")
    print(f"  Normas:    {' | '.join(norm_parts)}")

    # SEMA dominance
    sema_delta = ""
    if baseline:
        d = snap.sema_dominance - baseline.sema_dominance
        sign = "+" if d >= 0 else ""
        color = "\033[31m" if d > 5 else "\033[32m" if d < -2 else "\033[33m"
        sema_delta = f" ({color}{sign}{d:.1f}%\033[0m)"
    sema_color = "\033[31m" if snap.sema_dominance > 60 else "\033[33m" if snap.sema_dominance > 40 else "\033[32m"
    print(f"  SEMA dom:  {sema_color}{snap.sema_dominance:.1f}%\033[0m{sema_delta}")

    # Routing diversity
    rstd_delta = ""
    if baseline:
        d = snap.routing_std - baseline.routing_std
        sign = "+" if d >= 0 else ""
        rstd_delta = f" ({sign}{d:.4f})"
    print(f"  Route std: {snap.routing_std:.4f}{rstd_delta}")

    # LLAVES
    ll_delta = ""
    if baseline:
        d = snap.llaves_nonzero_pct - baseline.llaves_nonzero_pct
        sign = "+" if d >= 0 else ""
        ll_delta = f" ({sign}{d:.1f}%)"
    print(f"  LLAVES:    {snap.llaves_nonzero_pct:.1f}% non-zero{ll_delta}")

    # Stream balance
    bal_delta = ""
    if baseline:
        d = snap.stream_balance - baseline.stream_balance
        sign = "+" if d >= 0 else ""
        bal_delta = f" ({sign}{d:.3f})"
    bal_color = "\033[32m" if snap.stream_balance > 0.5 else "\033[33m" if snap.stream_balance > 0.2 else "\033[31m"
    print(f"  Balance:   {bal_color}{snap.stream_balance:.3f}\033[0m{bal_delta}")

    # Lateral scales
    if snap.lateral_scales_avg:
        scales_str = " | ".join(
            f"{STREAM_NAMES[i]}={v:.4f}" for i, v in enumerate(snap.lateral_scales_avg)
        )
        print(f"  Lat scale: {scales_str}")

    print(f"  {'─' * 60}")


def _check_regression(
    snap: ProbeSnapshot,
    baseline: ProbeBaseline,
    history: list[ProbeSnapshot],
) -> list[str]:
    """Detecta señales de regresión comparando con baseline."""
    warnings: list[str] = []

    # 1. Normas explotando (>2x baseline en cualquier nivel)
    for i, n in enumerate(snap.norms_per_level):
        if i < len(baseline.norms_per_level):
            ratio = n / max(baseline.norms_per_level[i], 1)
            if ratio > 2.0:
                warnings.append(
                    f"NORMA N{i} explotando: {n:.1f} ({ratio:.1f}x baseline)"
                )

    # 2. SEMA dominance aumentando (colapso territorial)
    if snap.sema_dominance > baseline.sema_dominance + 10:
        warnings.append(
            f"SEMA dominance subiendo: {snap.sema_dominance:.1f}% "
            f"(baseline {baseline.sema_dominance:.1f}%)"
        )

    # 3. Stream balance degradando
    if snap.stream_balance < baseline.stream_balance * 0.5:
        warnings.append(
            f"Stream balance degradando: {snap.stream_balance:.3f} "
            f"(baseline {baseline.stream_balance:.3f})"
        )

    # 4. Tendencia de las últimas 3 probes (normas subiendo consistentemente)
    if len(history) >= 3:
        last3 = history[-3:]
        for lvl in range(min(5, len(snap.norms_per_level))):
            vals = [h.norms_per_level[lvl] for h in last3 if lvl < len(h.norms_per_level)]
            if len(vals) == 3 and all(vals[j] < vals[j + 1] for j in range(2)):
                growth = vals[-1] / max(vals[0], 1)
                if growth > 1.3:
                    warnings.append(
                        f"N{lvl} normas subiendo 3 probes consecutivas "
                        f"({vals[0]:.0f}→{vals[-1]:.0f})"
                    )

    return warnings


# ─────────────────────────────────────────────────────────────────
# Training Step
# ─────────────────────────────────────────────────────────────────

def paso_entrenamiento(
    modelo: PamparV3,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    mascara: torch.Tensor,
    max_grad_norm: float,
    alpha_diff: float,
    alpha_balance: float,
    territory_table: torch.Tensor,
    warmup_factor: float = 1.0,
    focal_gamma: float = 0.0,
) -> dict[str, float]:
    """Un paso de entrenamiento: CE + routing + balance (sin exit, sin norm)."""
    modelo.train()
    optimizer.zero_grad(set_to_none=True)

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]
    loss_mask = mascara[:, 1:]

    logits, neuro_info = forward_neuro(modelo, input_ids)
    B, T, V = logits.shape

    # CE Loss principal
    targets_masked = targets.masked_fill(~loss_mask, -100)
    loss_ce = F.cross_entropy(
        logits.reshape(B * T, V),
        targets_masked.reshape(B * T),
        ignore_index=-100,
    )

    # Solo routing + balance (lecciones de 18 rounds: no exit, no norm train)
    loss_diff = calcular_loss_routing(
        neuro_info["all_terr_acts"], input_ids, territory_table,
        focal_gamma=focal_gamma,
    )
    loss_balance = calcular_loss_balance(neuro_info["all_stream_norms"])

    wf = warmup_factor
    loss = (
        loss_ce
        + wf * alpha_diff * loss_diff
        + wf * alpha_balance * loss_balance
    )

    # Si CE_mask era todo False (ej: agents sin marcador), CE=0 → no entrenar
    n_valid = loss_mask.sum().item()
    if n_valid == 0:
        return {"ce": 0.0, "diff": 0.0, "balance": 0.0, "total": 0.0, "skipped": True}

    if loss.isnan() or loss.isinf():
        logger.warning("Loss inestable, skipping step")
        return {"ce": 0.0, "diff": 0.0, "balance": 0.0, "total": 0.0, "skipped": True}

    loss.backward()
    nn_utils.clip_grad_norm_(modelo.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "ce": float(loss_ce.detach()),
        "diff": float(loss_diff.detach()),
        "balance": float(loss_balance.detach()),
        "total": float(loss.detach()),
        "skipped": False,
    }


# ─────────────────────────────────────────────────────────────────
# Utils
# ─────────────────────────────────────────────────────────────────

def _cosine_lr(
    paso: int, warmup: int, total: int, lr_max: float, lr_min: float,
) -> float:
    if paso < warmup:
        return lr_max * (paso + 1) / warmup
    progreso = (paso - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progreso))


def _guardar(
    ruta: Path, modelo: PamparV3, optimizer: torch.optim.Optimizer, paso: int,
) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "paso_global": paso,
            "config": dataclasses.asdict(modelo.config),
            "tipo": "ghidra_trainer",
        },
        ruta,
    )
    logger.info("Checkpoint guardado -> %s (paso %d)", ruta.name, paso)


# ─────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="PAMPAr Ghidra-Trainer: SFT con monitoreo GhidraProbe",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint-in", type=Path,
        default=ROOT / "checkpoints" / "v3_neuro_v9.pt",
    )
    p.add_argument(
        "--checkpoint-out", type=Path,
        default=ROOT / "checkpoints" / "v3_ghidra_v1.pt",
    )
    p.add_argument(
        "--tokenizer", type=Path,
        default=ROOT / "data" / "tokenizer" / "pampar_48k.model",
    )
    p.add_argument(
        "--data", type=Path,
        default=ROOT / "data" / "master_sft.jsonl",
    )

    # Learning rates
    p.add_argument("--lr", type=float, default=3e-7)
    p.add_argument("--lr-min", type=float, default=3e-8)
    p.add_argument("--lr-routing", type=float, default=1e-3)
    p.add_argument("--lr-routing-min", type=float, default=1e-4)

    # Training config
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-pasos", type=int, default=1000)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--guardar-cada", type=int, default=200)

    # Aux losses (conservador: solo routing + balance)
    p.add_argument("--alpha-diff", type=float, default=0.5)
    p.add_argument("--alpha-balance", type=float, default=0.05)
    p.add_argument("--focal-gamma", type=float, default=0.0,
                   help="Focal loss gamma (0=standard CE, 2+=focus on wrong tokens)")
    p.add_argument("--aux-warmup", type=int, default=50)

    # GhidraProbe
    p.add_argument("--probe-cada", type=int, default=50,
                   help="Cada cuantos pasos ejecutar GhidraProbe")
    p.add_argument("--abort-on-regression", action="store_true",
                   help="Detener si GhidraProbe detecta regresion severa")

    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

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
        args.checkpoint_in, map_location=device, weights_only=False,
    )
    config = ConfigV3(**payload["config"]) if "config" in payload else PRESET_V3
    modelo = PamparV3(config).to(device)
    modelo.load_state_dict(payload["modelo"])
    modelo.registrar_tokenizer(tok)
    logger.info(
        "Cargado '%s' (tipo: %s)", args.checkpoint_in.name, payload.get("tipo", "?"),
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
        1, sum(len(ids) for ids, _ in chunks),
    )
    logger.info("Porcentaje tokens en zona Solution: %.1f%%", pct_solution * 100)

    # Territory table
    territory_table = _build_territory_table(tok)

    # ── GhidraProbe setup ─────────────────────────────────────────
    probe = GhidraProbe(modelo)
    logger.info("GhidraProbe instalado (%d hooks)", len(probe._hooks))

    # Capturar baseline ANTES de entrenar
    logger.info("Capturando baseline GhidraProbe...")
    snap_baseline = _run_probe(probe, modelo, tok, device, paso=0)
    baseline = ProbeBaseline(
        norms_per_level=snap_baseline.norms_per_level[:],
        sema_dominance=snap_baseline.sema_dominance,
        routing_std=snap_baseline.routing_std,
        llaves_nonzero_pct=snap_baseline.llaves_nonzero_pct,
        stream_balance=snap_baseline.stream_balance,
    )
    _print_probe_report(snap_baseline)
    probe_history: list[ProbeSnapshot] = [snap_baseline]

    # ── Optimizer ─────────────────────────────────────────────────
    routing_names = {"talamo", "talamo_nivel", "lateral"}
    routing_params: list[nn.Parameter] = []
    main_params: list[nn.Parameter] = []
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
        n_routing / 1e3, args.lr_routing, n_main / 1e6, args.lr,
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
        "Ghidra-Training: %d pasos | %d chunks | "
        "alphas: diff=%.3f balance=%.3f | focal_gamma=%.1f | probe cada %d pasos",
        total_pasos, len(chunks),
        args.alpha_diff, args.alpha_balance, args.focal_gamma, args.probe_cada,
    )

    # ── Training Loop ─────────────────────────────────────────────
    paso = 0
    effective_steps = 0
    t0 = time.time()
    losses_ce: deque[float] = deque(maxlen=100)
    losses_total: deque[float] = deque(maxlen=100)
    best_ce = float("inf")
    best_paso = 0
    aborted = False

    try:
        for epoch in range(args.epochs):
            idx = list(range(len(chunks)))
            random.shuffle(idx)
            logger.info("== Epoch %d/%d ==", epoch + 1, args.epochs)

            for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
                batch_idx = idx[i : i + args.batch_size]
                tokens, mascara = _hacer_batch(
                    chunks, batch_idx, device, args.seq_len,
                )

                # LR scheduling
                lr_main = _cosine_lr(
                    paso, args.warmup, total_pasos, args.lr, args.lr_min,
                )
                lr_rout = _cosine_lr(
                    paso, args.warmup, total_pasos,
                    args.lr_routing, args.lr_routing_min,
                )
                optimizer.param_groups[0]["lr"] = lr_main
                optimizer.param_groups[1]["lr"] = lr_rout

                wf = min(1.0, paso / max(1, args.aux_warmup))

                loss_dict = paso_entrenamiento(
                    modelo, optimizer, tokens, mascara,
                    args.max_grad_norm, args.alpha_diff, args.alpha_balance,
                    territory_table, warmup_factor=wf,
                    focal_gamma=args.focal_gamma,
                )

                if not loss_dict.get("skipped", False) and loss_dict["ce"] > 0:
                    losses_ce.append(loss_dict["ce"])
                    losses_total.append(loss_dict["total"])
                    effective_steps += 1

                    # Track best CE
                    avg_ce = sum(losses_ce) / len(losses_ce)
                    if avg_ce < best_ce and paso > args.warmup:
                        best_ce = avg_ce
                        best_paso = paso

                paso += 1

                # Log cada 10 pasos
                if paso % 10 == 0:
                    avg_ce = sum(losses_ce) / max(1, len(losses_ce))
                    elapsed = time.time() - t0
                    logger.info(
                        "paso %4d/%d | CE=%.3f  diff=%.3f  bal=%.3f  "
                        "total=%.3f  lr=%.1e  (%.1f p/s)",
                        paso, total_pasos,
                        loss_dict["ce"], loss_dict["diff"],
                        loss_dict["balance"], avg_ce,
                        lr_rout, paso / elapsed,
                    )

                # GhidraProbe cada N pasos
                if paso % args.probe_cada == 0:
                    snap = _run_probe(probe, modelo, tok, device, paso)
                    _print_probe_report(snap, baseline)
                    probe_history.append(snap)

                    # Check regression
                    warnings = _check_regression(snap, baseline, probe_history)
                    if warnings:
                        print(f"\n  \033[33m{'!' * 40}")
                        print(f"  GHIDRA WARNINGS @ paso {paso}:")
                        for w in warnings:
                            print(f"    - {w}")
                        print(f"  {'!' * 40}\033[0m\n")

                        if args.abort_on_regression and len(warnings) >= 2:
                            logger.error(
                                "Abortando: %d warnings de regresion", len(warnings),
                            )
                            aborted = True
                            break

                # Guardar checkpoint periódico
                if paso % args.guardar_cada == 0:
                    _guardar(args.checkpoint_out, modelo, optimizer, paso)

                if paso >= args.max_pasos:
                    break

            if paso >= args.max_pasos or aborted:
                break

    except KeyboardInterrupt:
        logger.info("Interrumpido, guardando...")

    # ── Cleanup GhidraProbe ───────────────────────────────────────
    probe.detach()

    # ── Final probe + save ────────────────────────────────────────
    probe_final = GhidraProbe(modelo)
    snap_final = _run_probe(probe_final, modelo, tok, device, paso)
    probe_final.detach()

    _guardar(args.checkpoint_out, modelo, optimizer, paso)

    # ── Reporte final ─────────────────────────────────────────────
    elapsed = time.time() - t0
    avg_ce = sum(losses_ce) / max(1, len(losses_ce))

    print(f"\n{'=' * 60}")
    print(f"  GHIDRA-TRAINING COMPLETADO")
    print(f"{'=' * 60}")
    print(f"  Pasos:        {paso} ({effective_steps} efectivos)")
    print(f"  Tiempo:       {int(elapsed // 60)}m{int(elapsed % 60):02d}s")
    print(f"  Loss CE avg:  {avg_ce:.3f}")
    print(f"  PPL:          {math.exp(min(avg_ce, 10)):.1f}")
    print(f"  Best CE avg:  {best_ce:.3f} (paso {best_paso})")
    print(f"  Checkpoint:   {args.checkpoint_out}")
    if aborted:
        print(f"  \033[31mABORTADO por regresion detectada\033[0m")

    print(f"\n  --- Baseline vs Final ---")
    print(f"  {'Metrica':<20} {'Baseline':>10} {'Final':>10} {'Delta':>10}")
    print(f"  {'─' * 54}")

    for i in range(len(snap_final.norms_per_level)):
        b = baseline.norms_per_level[i] if i < len(baseline.norms_per_level) else 0
        f_val = snap_final.norms_per_level[i]
        print(f"  {'Norm N' + str(i):<20} {b:>10.1f} {f_val:>10.1f} {f_val - b:>+10.1f}")

    print(f"  {'SEMA dominance':<20} {baseline.sema_dominance:>9.1f}% {snap_final.sema_dominance:>9.1f}% {snap_final.sema_dominance - baseline.sema_dominance:>+9.1f}%")
    print(f"  {'Routing std':<20} {baseline.routing_std:>10.4f} {snap_final.routing_std:>10.4f} {snap_final.routing_std - baseline.routing_std:>+10.4f}")
    print(f"  {'Stream balance':<20} {baseline.stream_balance:>10.3f} {snap_final.stream_balance:>10.3f} {snap_final.stream_balance - baseline.stream_balance:>+10.3f}")

    # Guardar historial de probes como JSON
    probe_log = args.checkpoint_out.with_suffix(".probe.json")
    probe_entries = []
    for s in probe_history + [snap_final]:
        probe_entries.append({
            "paso": s.paso,
            "norms": s.norms_per_level,
            "sema_dom": s.sema_dominance,
            "routing_std": s.routing_std,
            "llaves_pct": s.llaves_nonzero_pct,
            "balance": s.stream_balance,
        })
    with open(probe_log, "w", encoding="utf-8") as f:
        json.dump(probe_entries, f, indent=2)
    print(f"\n  Probe log:    {probe_log.name}")

    print(f"\n  Verificar con Brain Scanner:")
    print(
        f"    python -X utf8 scripts/brain_scanner.py --suite --device cuda"
        f" --checkpoint {args.checkpoint_out}",
    )
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
