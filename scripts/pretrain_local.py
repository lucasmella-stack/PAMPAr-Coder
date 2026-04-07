#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
pretrain_local.py — Continual pretrain de PamparV3 en GPU local (GTX 1650 4GB).

Entrena con datos textbook en formato CLM (next token prediction).
Optimizado para VRAM limitada: AMP fp16 + gradient accumulation + checkpointing.

Uso:
  # Pretrain con datos generados (cuando la generación termine)
  python scripts/pretrain_local.py

  # Con opciones custom
  python scripts/pretrain_local.py --epochs 8 --lr 1e-4 --grad-accum 8

  # Reanudar entrenamiento interrumpido
  python scripts/pretrain_local.py --resume

  # Esperar a que la generación termine antes de entrenar
  python scripts/pretrain_local.py --wait-for-data 200

Detener con Ctrl-C — guarda checkpoint antes de salir.
"""

import argparse
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
import torch.nn.functional as F
import torch.nn.utils as nn_utils

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PRESET_V3, ConfigV3, PamparV3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pretrain_local")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────


class TextbookDataset:
    """
    Carga textbook JSONL, tokeniza y sirve chunks aleatorios para CLM.

    Divide textos largos en chunks solapados de max_seq_len+1 tokens.
    El +1 es para tener input ([:L]) y target ([1:L+1]).
    """

    def __init__(
        self,
        ruta_jsonl: Path,
        tokenizer: spm.SentencePieceProcessor,
        max_seq_len: int = 512,
    ) -> None:
        self.tok = tokenizer
        self.max_seq_len = max_seq_len
        self.chunks: list[list[int]] = []

        self._cargar(ruta_jsonl)

    def _cargar(self, ruta: Path) -> None:
        """Lee el JSONL y tokeniza en chunks."""
        n_textos = 0
        for linea in ruta.read_text(encoding="utf-8").splitlines():
            if not linea.strip():
                continue
            try:
                obj = json.loads(linea)
                texto = obj.get("text", "")
            except json.JSONDecodeError:
                continue

            if not texto or len(texto) < 50:
                continue

            ids = self.tok.Encode(texto)
            n_textos += 1

            # Chunks solapados (50% overlap)
            step = self.max_seq_len // 2
            for i in range(0, max(1, len(ids) - self.max_seq_len), step):
                chunk = ids[i : i + self.max_seq_len + 1]
                if len(chunk) >= 32:
                    self.chunks.append(chunk)

        logger.info(
            "Dataset: %d textos → %d chunks (seq_len=%d)",
            n_textos,
            len(self.chunks),
            self.max_seq_len,
        )

    def __len__(self) -> int:
        return len(self.chunks)

    def get_batch(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Devuelve un batch aleatorio [B, L+1] padded."""
        indices = random.sample(
            range(len(self.chunks)), min(batch_size, len(self.chunks))
        )
        seleccionados = [self.chunks[i] for i in indices]

        max_len = min(
            max(len(c) for c in seleccionados),
            self.max_seq_len + 1,
        )

        padded = []
        for chunk in seleccionados:
            trunc = chunk[:max_len]
            pad = [0] * (max_len - len(trunc))
            padded.append(trunc + pad)

        return torch.tensor(padded, dtype=torch.long, device=device)

    @property
    def total_tokens(self) -> int:
        """Total de tokens en el dataset (sin padding)."""
        return sum(len(c) for c in self.chunks)


# ─────────────────────────────────────────────────────────────────────────────
# LR Scheduler
# ─────────────────────────────────────────────────────────────────────────────


def cosine_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    lr_max: float,
    lr_min: float = 1e-6,
) -> float:
    """Cosine schedule con warmup lineal."""
    if step < warmup_steps:
        return lr_max * (step + 1) / warmup_steps
    progreso = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * progreso))


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────


def entrenar(
    modelo: PamparV3,
    dataset: TextbookDataset,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    grad_accum: int,
    max_grad_norm: float,
    lr_max: float,
    guardar_cada: int,
    ruta_ckpt: Path,
    paso_inicio: int = 0,
    mejor_loss: float = float("inf"),
    use_amp: bool = False,
) -> None:
    """Bucle principal de continual pretrain. AMP opcional (fp32 por defecto)."""
    steps_per_epoch = max(1, len(dataset) // (batch_size * grad_accum))
    total_steps = steps_per_epoch * epochs
    warmup_steps = min(total_steps // 10, 100)

    logger.info(
        "Config: epochs=%d  batch=%d  grad_accum=%d  effective_batch=%d  amp=%s",
        epochs,
        batch_size,
        grad_accum,
        batch_size * grad_accum,
        use_amp,
    )
    logger.info(
        "Steps: %d/epoch  %d total  warmup=%d  lr_max=%.1e",
        steps_per_epoch,
        total_steps,
        warmup_steps,
        lr_max,
    )
    logger.info(
        "Dataset: %d chunks  ~%dK tokens", len(dataset), dataset.total_tokens // 1000
    )

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    paso = paso_inicio
    mejor = mejor_loss
    t_inicio = time.time()

    try:
        for epoch in range(1, epochs + 1):
            losses_epoch: list[float] = []
            modelo.train()

            for micro_step in range(steps_per_epoch * grad_accum):
                # ── LR schedule ──────────────────────────────────────────
                lr = cosine_lr(paso, warmup_steps, total_steps, lr_max)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                # ── Forward + loss ───────────────────────────────────────
                tokens = dataset.get_batch(batch_size, device)
                input_ids = tokens[:, :-1]
                targets = tokens[:, 1:]

                with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    logits, loss, _info = modelo(input_ids, targets=targets)
                    loss = loss / grad_accum

                # ── Backward ─────────────────────────────────────────────
                scaler.scale(loss).backward()

                loss_val = float(loss.detach()) * grad_accum
                losses_epoch.append(loss_val)

                # ── Optimizer step cada grad_accum micro-steps ────────────
                if (micro_step + 1) % grad_accum == 0:
                    scaler.unscale_(optimizer)
                    nn_utils.clip_grad_norm_(modelo.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    paso += 1

                    # ── Log cada 10 pasos ────────────────────────────────
                    if paso % 10 == 0:
                        avg_recent = sum(losses_epoch[-grad_accum * 10 :]) / min(
                            len(losses_epoch), grad_accum * 10
                        )
                        ppl = math.exp(min(avg_recent, 20.0))
                        elapsed = time.time() - t_inicio
                        eta_s = (
                            (total_steps - paso)
                            / max(1, (paso - paso_inicio) / elapsed)
                            if elapsed > 0
                            else 0
                        )
                        eta_h = eta_s / 3600

                        logger.info(
                            "epoch %d | paso %4d/%d | loss=%.3f  ppl=%.1f | lr=%.1e | ETA=%.1fh",
                            epoch,
                            paso,
                            total_steps,
                            avg_recent,
                            ppl,
                            lr,
                            eta_h,
                        )

                    # ── VRAM log (solo paso 1) ───────────────────────────
                    if paso == paso_inicio + 1:
                        alloc = torch.cuda.max_memory_allocated() / 1e9
                        logger.info("VRAM pico: %.2f GB / 4.00 GB", alloc)

                    # ── Guardar periódico ─────────────────────────────────
                    if guardar_cada > 0 and paso % guardar_cada == 0:
                        _guardar_checkpoint(
                            modelo,
                            optimizer,
                            paso,
                            mejor,
                            ruta_ckpt.parent / f"v3_pretrain_step{paso}.pt",
                        )

            # ── Fin de epoch ─────────────────────────────────────────────
            avg_epoch = sum(losses_epoch) / len(losses_epoch)
            ppl_epoch = math.exp(min(avg_epoch, 20.0))
            elapsed = time.time() - t_inicio
            hh, mm = int(elapsed // 3600), int((elapsed % 3600) // 60)

            logger.info(
                "═══ Epoch %d/%d  loss=%.3f  ppl=%.1f  [%02dh%02dm] ═══",
                epoch,
                epochs,
                avg_epoch,
                ppl_epoch,
                hh,
                mm,
            )

            # Guardar checkpoint de epoch
            _guardar_checkpoint(
                modelo,
                optimizer,
                paso,
                mejor,
                ruta_ckpt.parent / f"v3_pretrain_epoch{epoch}.pt",
            )

            # Guardar best
            if avg_epoch < mejor:
                mejor = avg_epoch
                _guardar_checkpoint(
                    modelo,
                    optimizer,
                    paso,
                    mejor,
                    ruta_ckpt,
                )
                logger.info("★ Nuevo mejor loss: %.3f", mejor)

    except KeyboardInterrupt:
        logger.info("\nInterrumpido — guardando checkpoint final...")

    # Guardar checkpoint final
    _guardar_checkpoint(
        modelo,
        optimizer,
        paso,
        mejor,
        ruta_ckpt.parent / "v3_pretrain_last.pt",
    )
    elapsed = time.time() - t_inicio
    logger.info(
        "Pretrain completado — %d pasos, mejor loss=%.3f, tiempo=%.1f min",
        paso - paso_inicio,
        mejor,
        elapsed / 60,
    )


def _guardar_checkpoint(
    modelo: PamparV3,
    optimizer: torch.optim.Optimizer,
    paso: int,
    mejor_loss: float,
    ruta: Path,
) -> None:
    """Guarda checkpoint con modelo + optimizer + metadata."""
    import dataclasses

    ruta.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "modelo": modelo.state_dict(),
            "optimizer": optimizer.state_dict(),
            "paso_global": paso,
            "config": dataclasses.asdict(modelo.config),
            "mejor_loss": mejor_loss,
            "tipo": "pretrain_local",
        },
        ruta,
    )
    logger.info("✓ Checkpoint → %s (paso %d)", ruta.name, paso)


# ─────────────────────────────────────────────────────────────────────────────
# Esperar datos
# ─────────────────────────────────────────────────────────────────────────────


def esperar_datos(ruta_jsonl: Path, min_ejemplos: int, intervalo: int = 30) -> None:
    """Espera hasta que el JSONL tenga al menos min_ejemplos líneas."""
    logger.info("Esperando ≥%d ejemplos en %s...", min_ejemplos, ruta_jsonl.name)
    while True:
        if ruta_jsonl.exists():
            n = sum(
                1
                for line in ruta_jsonl.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if n >= min_ejemplos:
                logger.info("Datos listos: %d ejemplos", n)
                return
            logger.info(
                "  %d/%d ejemplos — esperando %ds...", n, min_ejemplos, intervalo
            )
        else:
            logger.info("  Archivo no existe aún — esperando %ds...", intervalo)
        time.sleep(intervalo)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Continual pretrain de PamparV3 en GPU local",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rutas
    p.add_argument(
        "--checkpoint-base",
        type=Path,
        default=ROOT / "checkpoints" / "v3_ghidra_v9.pt",
        help="Checkpoint base del que partir",
    )
    p.add_argument(
        "--checkpoint-out",
        type=Path,
        default=ROOT / "checkpoints" / "v3_pretrain_best.pt",
        help="Ruta del checkpoint de salida (best)",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "textbook_v3" / "textbook_pretrain.jsonl",
        help="Datos de textbook JSONL",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "data" / "tokenizer" / "pampar_48k.model",
        help="SentencePiece 48K",
    )

    # Hiperparámetros
    p.add_argument("--epochs", type=int, default=5, help="Número de epochs")
    p.add_argument("--lr", type=float, default=1e-4, help="Learning rate máximo")
    p.add_argument("--batch-size", type=int, default=2, help="Micro-batch size")
    p.add_argument(
        "--grad-accum", type=int, default=4, help="Gradient accumulation steps"
    )
    p.add_argument(
        "--seq-len", type=int, default=512, help="Longitud máxima de secuencia"
    )
    p.add_argument("--max-grad-norm", type=float, default=0.5, help="Gradient clipping")
    p.add_argument("--weight-decay", type=float, default=0.1, help="Weight decay")
    p.add_argument(
        "--guardar-cada",
        type=int,
        default=200,
        help="Guardar cada N pasos. 0=solo epochs",
    )

    # Control
    p.add_argument(
        "--resume", action="store_true", help="Reanudar desde checkpoint de salida"
    )
    p.add_argument(
        "--wait-for-data",
        type=int,
        default=0,
        help="Esperar hasta N ejemplos en el JSONL antes de empezar",
    )
    p.add_argument(
        "--amp",
        action="store_true",
        help="Usar AMP fp16 (puede causar NaN en esta arquitectura)",
    )

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Validar CUDA ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        logger.error("CUDA no disponible. Este script requiere GPU.")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info("GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)

    # ── Esperar datos si se pide ──────────────────────────────────────────────
    if args.wait_for_data > 0:
        esperar_datos(args.data, args.wait_for_data)

    # ── Validar archivos ──────────────────────────────────────────────────────
    if not args.data.exists():
        logger.error("Datos no encontrados: %s", args.data)
        sys.exit(1)
    if not args.tokenizer.exists():
        logger.error("Tokenizer no encontrado: %s", args.tokenizer)
        sys.exit(1)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok = spm.SentencePieceProcessor()
    tok.Load(str(args.tokenizer))
    logger.info("Tokenizer: vocab=%d", tok.GetPieceSize())

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = TextbookDataset(args.data, tok, max_seq_len=args.seq_len)
    if len(dataset) == 0:
        logger.error("Dataset vacío — ¿generación de datos incompleta?")
        sys.exit(1)

    # ── Modelo ────────────────────────────────────────────────────────────────
    ruta_base = args.checkpoint_base
    paso_inicio = 0
    mejor_loss = float("inf")

    if args.resume and args.checkpoint_out.exists():
        ruta_base = args.checkpoint_out
        logger.info("Reanudando desde %s", ruta_base)

    if not ruta_base.exists():
        logger.error("Checkpoint base no encontrado: %s", ruta_base)
        sys.exit(1)

    logger.info("Cargando checkpoint: %s", ruta_base.name)
    payload = torch.load(ruta_base, map_location="cpu", weights_only=False)
    config = ConfigV3(**payload["config"]) if "config" in payload else PRESET_V3
    modelo = PamparV3(config).to(device)
    modelo.load_state_dict(payload["modelo"])

    if args.resume and "paso_global" in payload:
        paso_inicio = int(payload["paso_global"])
        mejor_loss = float(payload.get("mejor_loss", float("inf")))
        logger.info(
            "Reanudando desde paso %d, mejor_loss=%.3f", paso_inicio, mejor_loss
        )

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info(
        "PamparV3 — %.1fM params, gradient checkpointing=%s",
        n_params / 1e6,
        config.use_checkpoint,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
        eps=1e-8,
    )

    if args.resume and "optimizer" in payload:
        try:
            optimizer.load_state_dict(payload["optimizer"])
            logger.info("Optimizer restaurado")
        except Exception as e:
            logger.warning("No se pudo restaurar optimizer: %s", e)

    del payload
    torch.cuda.empty_cache()

    # ── Entrenar ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("CONTINUAL PRETRAIN — PamparV3 108M")
    logger.info("=" * 60)

    entrenar(
        modelo=modelo,
        dataset=dataset,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        max_grad_norm=args.max_grad_norm,
        lr_max=args.lr,
        guardar_cada=args.guardar_cada,
        ruta_ckpt=args.checkpoint_out,
        paso_inicio=paso_inicio,
        mejor_loss=mejor_loss,
        use_amp=args.amp,
    )


if __name__ == "__main__":
    main()
