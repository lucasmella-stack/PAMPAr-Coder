#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
sft_v5.py — Third-pass SFT de PamparV3 sobre datos verificados en runtime.

Diferencias respecto a sft_v4.py:
  - Parte de v3_sft_v4.pt (el mejor checkpoint hasta ahora: 8/16)
  - Usa sft_v5.jsonl — dataset generado por generate_sft_v5.py
    donde CADA ejemplo fue exec() + assert antes de incluirse
  - Cubre los 8 patrones que el modelo falla (fizzbuzz, cuadrados,
    invertir_dict, busqueda_binaria, merge_sort, Punto, memoize, primos)
  - LR aun mas bajo: 3e-6 -> 3e-7 (no destruir el SFT v4)
  - Guarda en v3_sft_v5.pt

Uso:
  python -X utf8 scripts/sft_v5.py
"""

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
import torch.nn.functional as F
import torch.nn.utils as nn_utils
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PamparV3, ConfigV3, PRESET_V3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sft_v5")

# Token string que marca el inicio de la respuesta — solo se entrena a partir de aqui
_SOLUTION_MARKER = "### Solution:"


# ─────────────────────────────────────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────────────────────────────────────

def _cargar_targeted(ruta: Path) -> list[str]:
    """Carga targeted_sft.jsonl."""
    ejemplos: list[str] = []
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            obj = json.loads(linea)
            texto = obj.get("text", "")
            if texto and "### Solution:" in texto:
                ejemplos.append(texto)
        except json.JSONDecodeError:
            continue
    logger.info("Targeted SFT: %d ejemplos cargados", len(ejemplos))
    return ejemplos


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizacion con mascara de loss
# ─────────────────────────────────────────────────────────────────────────────

def _tokenizar_con_mascara(
    ejemplos: list[str],
    tok: spm.SentencePieceProcessor,
    max_seq_len: int,
) -> list[tuple[list[int], list[bool]]]:
    """
    Tokeniza cada ejemplo y calcula una mascara booleana.
    mask[i] = True si el token i pertenece a la seccion ### Solution: en adelante.
    Solo se computa loss sobre posiciones donde mask=True.
    """
    chunks: list[tuple[list[int], list[bool]]] = []
    max_chars = max_seq_len * 6

    for texto in ejemplos:
        if len(texto) > max_chars:
            texto = texto[:max_chars]

        # Encontrar posicion del marcador para saber desde donde calcular loss
        marker_pos = texto.find(_SOLUTION_MARKER)
        if marker_pos < 0:
            # Sin marcador: entrenar sobre todo (fallback)
            ids = tok.Encode(texto)
            if len(ids) >= 16:
                trunc = ids[:max_seq_len + 1]
                chunks.append((trunc, [True] * len(trunc)))
            continue

        # Tokenizar el prefijo hasta el marcador para saber la longitud en tokens
        prefijo = texto[:marker_pos + len(_SOLUTION_MARKER)]
        ids_prefijo = tok.Encode(prefijo)
        ids_full = tok.Encode(texto)

        if len(ids_full) < 16:
            continue

        ids_full = ids_full[:max_seq_len + 1]
        n_prefijo = min(len(ids_prefijo), len(ids_full))

        # mask: False para el prefijo del problema, True para la solucion
        mascara = [False] * n_prefijo + [True] * (len(ids_full) - n_prefijo)
        chunks.append((ids_full, mascara))

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hacer_batch_mascarado(
    chunks: list[tuple[list[int], list[bool]]],
    indices: list[int],
    device: torch.device,
    max_seq_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch con padding y mascara de loss."""
    sels = [chunks[i] for i in indices]
    max_len = min(max(len(ids) for ids, _ in sels), max_seq_len + 1)
    padded_ids = []
    padded_mask = []
    for ids, mask in sels:
        t = ids[:max_len]
        m = mask[:max_len]
        pad_len = max_len - len(t)
        padded_ids.append(t + [0] * pad_len)
        padded_mask.append(m + [False] * pad_len)
    tokens = torch.tensor(padded_ids, dtype=torch.long, device=device)
    mascara = torch.tensor(padded_mask, dtype=torch.bool, device=device)
    return tokens, mascara


def _paso_mascarado(
    modelo: PamparV3,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    mascara: torch.Tensor,
    max_grad_norm: float,
) -> float:
    """
    Forward + backward con loss mascarado.
    Solo se entrena sobre los tokens de la solucion.
    """
    modelo.train()
    optimizer.zero_grad(set_to_none=True)

    input_ids = tokens[:, :-1]     # (B, T)
    targets = tokens[:, 1:]        # (B, T)
    loss_mask = mascara[:, 1:]     # alinear la mascara con targets

    logits, _, _ = modelo(input_ids)  # (B, T, V)
    B, T, V = logits.shape

    # Loss solo sobre posiciones de la solucion
    logits_flat = logits.reshape(B * T, V)
    targets_flat = targets.reshape(B * T)
    mask_flat = loss_mask.reshape(B * T)

    # Ignorar tokens no enmascarados poniendo target=-100
    targets_masked = targets_flat.masked_fill(~mask_flat, -100)
    loss = F.cross_entropy(logits_flat, targets_masked, ignore_index=-100)

    if loss.isnan() or loss.isinf():
        logger.warning("Loss inestable — skipping step")
        return 0.0

    loss.backward()
    nn_utils.clip_grad_norm_(modelo.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.detach())


def _cosine_lr(paso: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    if paso < warmup:
        return lr_max * (paso + 1) / warmup
    progreso = (paso - warmup) / max(1, total - warmup)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progreso))


def _guardar(ruta: Path, modelo: PamparV3, optimizer: torch.optim.Optimizer, paso: int) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "modelo": modelo.state_dict(),
        "optimizer": optimizer.state_dict(),
        "paso_global": paso,
        "config": dataclasses.asdict(modelo.config),
        "tipo": "sft_v5",
    }, ruta)
    logger.info("Checkpoint SFT-v5 guardado -> paso %d", paso)


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--checkpoint-in", type=Path,
                   default=ROOT / "checkpoints" / "v3_sft_v4.pt")
    p.add_argument("--checkpoint-out", type=Path,
                   default=ROOT / "checkpoints" / "v3_sft_v5.pt")
    p.add_argument("--tokenizer", type=Path,
                   default=ROOT / "data" / "tokenizer" / "pampar_48k.model")
    p.add_argument("--targeted", type=Path,
                   default=ROOT / "data" / "sft_v5.jsonl")
    p.add_argument("--lr", type=float, default=3e-6)
    p.add_argument("--lr-min", type=float, default=3e-7)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--max-pasos", type=int, default=2000)
    p.add_argument("--epochs", type=int, default=30,
                   help="Más epochs sobre dataset pequeño verificado")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--guardar-cada", type=int, default=500)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else args.device if args.device != "auto" else "cpu"
    )
    logger.info("Device: %s", device)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)
        logger.info("GPU: %s (%.1f GiB)",
                    torch.cuda.get_device_name(0),
                    torch.cuda.get_device_properties(0).total_memory / 1e9)

    # Tokenizer
    tok = spm.SentencePieceProcessor()
    tok.Load(str(args.tokenizer))
    logger.info("Tokenizer vocab=%d", tok.GetPieceSize())

    # Modelo desde v3_sft.pt
    if not args.checkpoint_in.exists():
        logger.error("Checkpoint no encontrado: %s", args.checkpoint_in)
        sys.exit(1)
    payload = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
    config = ConfigV3(**payload["config"]) if "config" in payload else PRESET_V3
    modelo = PamparV3(config).to(device)
    modelo.load_state_dict(payload["modelo"])
    logger.info("Cargado desde %s (tipo: %s) → fine-tune v5", args.checkpoint_in.name,
                payload.get("tipo", "?"))
    del payload

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info("PamparV3 %.1fM params", n_params / 1e6)

    # Datos
    if not args.targeted.exists():
        logger.error("Dataset dirigido no encontrado: %s", args.targeted)
        logger.error("Primero ejecuta: python -X utf8 scripts/generate_sft_v5.py")
        sys.exit(1)

    ejemplos = _cargar_targeted(args.targeted)
    chunks = _tokenizar_con_mascara(ejemplos, tok, args.seq_len)
    logger.info("Chunks tokenizados con mascara: %d", len(chunks))
    del ejemplos

    # Verificar que hay suficiente texto en la zona de solucion
    pct_solution = sum(sum(m) for _, m in chunks) / max(1, sum(len(ids) for ids, _ in chunks))
    logger.info("Porcentaje de tokens en zona Solution: %.1f%%", pct_solution * 100)

    # Optimizer con LR bajo (no destruir lo aprendido)
    optimizer = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8,
    )

    pasos_por_epoch = max(1, len(chunks) // args.batch_size)
    total_pasos = min(args.max_pasos, args.epochs * pasos_por_epoch)
    logger.info("SFT-v4: %d pasos | %d chunks | %d p/epoch | lr=%.1e->%.1e",
                total_pasos, len(chunks), pasos_por_epoch, args.lr, args.lr_min)

    paso = 0
    t0 = time.time()
    losses: deque[float] = deque(maxlen=100)

    try:
        for epoch in range(args.epochs):
            idx = list(range(len(chunks)))
            random.shuffle(idx)
            logger.info("-- Epoch %d/%d --", epoch + 1, args.epochs)

            for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
                batch_idx = idx[i: i + args.batch_size]
                tokens, mascara = _hacer_batch_mascarado(chunks, batch_idx, device, args.seq_len)

                lr = _cosine_lr(paso, args.warmup, total_pasos, args.lr, args.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                loss = _paso_mascarado(modelo, optimizer, tokens, mascara, args.max_grad_norm)
                if loss > 0:
                    losses.append(loss)
                paso += 1

                if paso % 10 == 0:
                    avg = sum(losses) / max(1, len(losses))
                    elapsed = time.time() - t0
                    logger.info(
                        "paso %5d/%d | loss=%.3f  avg100=%.3f  lr=%.1e  ppl=%.1f  (%.1f p/s)",
                        paso, total_pasos, loss, avg, lr,
                        math.exp(min(avg, 10)),
                        paso / elapsed,
                    )

                if paso % args.guardar_cada == 0:
                    _guardar(args.checkpoint_out, modelo, optimizer, paso)

                if paso >= args.max_pasos:
                    break

            if paso >= args.max_pasos:
                break

    except KeyboardInterrupt:
        logger.info("Interrumpido — guardando...")

    _guardar(args.checkpoint_out, modelo, optimizer, paso)

    elapsed = time.time() - t0
    avg_final = sum(losses) / max(1, len(losses))
    print(f"-- SFT-v5 Completado --")
    print(f"  Pasos: {paso}")
    print(f"  Tiempo: {int(elapsed // 3600)}h{int((elapsed % 3600) // 60):02d}m")
    print(f"  Loss final (avg100): {avg_final:.3f}")
    print(f"  PPL final: {math.exp(min(avg_final, 10)):.1f}")
    print(f"  Checkpoint: {args.checkpoint_out}")
    print(f"\n  Evaluar con:")
    print(f"    python -X utf8 scripts/eval_v3.py --checkpoint checkpoints/v3_sft_v5.pt")


if __name__ == "__main__":
    main()
