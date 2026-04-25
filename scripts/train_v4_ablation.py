#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
train_v4_ablation.py — Entrenamiento de la ablación científica de Fase 6.

Lanza una corrida (config YAML + seed) y produce:
  - <output>/<name>_seed<S>/metrics.jsonl   (loss/eval/throughput por step)
  - <output>/<name>_seed<S>/config.json     (config resuelta + n_params)
  - <output>/<name>_seed<S>/final.pt        (state_dict + step + cfg)

Diseño:
  - Carga config YAML con `extends:` simple (un nivel de herencia).
  - Tokeniza data JSONL (`text` field) con SentencePiece.
  - Train/val split determinístico por hash.
  - AMP (autocast + GradScaler) si CUDA disponible y `amp: true`.
  - Eval cada N steps sobre val split fijo.
  - JSONL append-only para que un proceso externo pueda graficar live.

Uso:
  # CPU smoke (1 seed, pocos steps)
  python scripts/train_v4_ablation.py \
      --config configs/phase6_ablation/A_baseline.yaml \
      --max-steps 50 --seed 42

  # GPU full run
  python scripts/train_v4_ablation.py \
      --config configs/phase6_ablation/B_full.yaml \
      --seed 42

  # Lanzar las 5 variantes × 3 seeds
  bash scripts/launch_phase6_ablation.sh
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.utils as nn_utils
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v4 import ConfigV4, PamparV4

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("v4_ablation")


# ────────────────────────────────────────────────────────────────────────────
# Config YAML loader (con extends simple)
# ────────────────────────────────────────────────────────────────────────────


def load_config(path: Path) -> dict[str, Any]:
    """Carga YAML resolviendo un nivel de `extends:` (path relativo)."""
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    extends = raw.pop("extends", None)
    if extends:
        base_path = path.parent / extends
        base = load_config(base_path)
        merged = _deep_merge(base, raw)
        return merged
    return raw


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge recursivo: override gana en escalares, recursión en dicts."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ────────────────────────────────────────────────────────────────────────────
# Data loader: tokeniza un JSONL una sola vez
# ────────────────────────────────────────────────────────────────────────────


class JsonlTokenLoader:
    """
    Tokeniza un JSONL completo a memoria como un array contiguo y sirve
    batches deterministas. Train/val split por hash del índice del ejemplo.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: spm.SentencePieceProcessor,
        seq_len: int,
        batch_size: int,
        val_fraction: float,
        seed: int,
    ):
        self._tok = tokenizer
        self._seq_len = seq_len + 1  # +1 para target shift
        self._bs = batch_size

        logger.info("Tokenizando %s ...", jsonl_path)
        all_ids: list[int] = []
        eos = tokenizer.PieceToId("</s>") if tokenizer.PieceToId("</s>") >= 0 else 2
        with jsonl_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = obj.get("text") or obj.get("content") or ""
                if not text:
                    continue
                ids = tokenizer.Encode(text)
                ids.append(eos)
                all_ids.extend(ids)

        ids_np = np.asarray(all_ids, dtype=np.int32)
        n_chunks = max(0, (len(ids_np) - self._seq_len) // self._seq_len)
        if n_chunks < 4:
            raise RuntimeError(
                f"Pocos chunks ({n_chunks}) — dataset muy chico o seq_len muy grande"
            )

        chunks = np.empty((n_chunks, self._seq_len), dtype=np.int32)
        for i in range(n_chunks):
            chunks[i] = ids_np[i * self._seq_len : (i + 1) * self._seq_len]

        # Train/val split determinístico por hash del índice
        rng = np.random.RandomState(seed)
        order = rng.permutation(n_chunks)
        n_val = max(1, int(n_chunks * val_fraction))
        self._val = chunks[order[:n_val]]
        self._train = chunks[order[n_val:]]
        self._train_order = rng.permutation(len(self._train))
        self._cursor = 0

        logger.info(
            "Loader listo — %d tokens, %d chunks (train=%d, val=%d)",
            len(ids_np),
            n_chunks,
            len(self._train),
            len(self._val),
        )

    def next_train_batch(self, device: torch.device) -> torch.Tensor:
        idxs = []
        for _ in range(self._bs):
            idxs.append(self._train_order[self._cursor % len(self._train_order)])
            self._cursor += 1
        batch = self._train[idxs]
        return torch.from_numpy(batch.astype(np.int64)).to(device)

    def iter_val_batches(self, device: torch.device, max_batches: int = 16):
        n = min(max_batches, len(self._val) // self._bs)
        for i in range(n):
            batch = self._val[i * self._bs : (i + 1) * self._bs]
            yield torch.from_numpy(batch.astype(np.int64)).to(device)


# ────────────────────────────────────────────────────────────────────────────
# Helpers de training
# ────────────────────────────────────────────────────────────────────────────


def build_model_from_config(model_cfg: dict[str, Any]) -> tuple[PamparV4, ConfigV4]:
    """Filtra solo los fields válidos de ConfigV4 y construye el modelo."""
    valid_fields = {f.name for f in dataclasses.fields(ConfigV4)}
    filtered = {k: v for k, v in model_cfg.items() if k in valid_fields}
    cfg = ConfigV4(**filtered)
    model = PamparV4(cfg)
    return model, cfg


def cosine_lr(step: int, warmup: int, total: int, base_lr: float) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_loss(model: PamparV4, loader: JsonlTokenLoader, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader.iter_val_batches(device):
        ids = batch[:, :-1]
        targets = batch[:, 1:]
        _, loss, _ = model(ids, targets=targets)
        total += loss.item()
        n += 1
    model.train()
    return total / max(1, n)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────


def main() -> int:
    p = argparse.ArgumentParser(description="PAMPAr V4 — Fase 6 ablation training")
    p.add_argument("--config", type=Path, required=True, help="Path a YAML")
    p.add_argument("--seed", type=int, default=None, help="Override seed")
    p.add_argument("--max-steps", type=int, default=None, help="Override max_steps")
    p.add_argument("--device", default="auto")
    p.add_argument("--output-dir", type=Path, default=None, help="Override output dir")
    args = p.parse_args()

    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    name = cfg.get("name", args.config.stem)
    seed = args.seed if args.seed is not None else train_cfg.get("seed", 42)
    max_steps = args.max_steps if args.max_steps is not None else train_cfg["max_steps"]
    output_dir = args.output_dir or (ROOT / train_cfg["output_dir"])
    run_dir = output_dir / f"{name}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"

    logger.info("Run: %s | seed=%d | device=%s | amp=%s", name, seed, device, use_amp)
    logger.info("Config: %s", args.config)

    # ── Tokenizer
    tok_path = ROOT / train_cfg["tokenizer_path"]
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer no encontrado: {tok_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tok_path))

    # ── Data
    data_path = ROOT / train_cfg["data_path"]
    loader = JsonlTokenLoader(
        data_path,
        tokenizer,
        seq_len=train_cfg["seq_len"],
        batch_size=train_cfg["batch_size"],
        val_fraction=train_cfg["val_split"],
        seed=seed,
    )

    # ── Modelo
    model, resolved_cfg = build_model_from_config(model_cfg)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Modelo: %.2fM params", n_params / 1e6)

    # ── Persist config + meta
    meta = {
        "name": name,
        "description": cfg.get("description", ""),
        "seed": seed,
        "max_steps": max_steps,
        "device": str(device),
        "amp": use_amp,
        "n_params": n_params,
        "model_config": dataclasses.asdict(resolved_cfg),
        "training": train_cfg,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "torch_version": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "data_path_hash": hashlib.sha1(str(data_path).encode()).hexdigest()[:8],
    }
    (run_dir / "config.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    # ── Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        betas=(0.9, 0.95),
        weight_decay=train_cfg["weight_decay"],
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # ── Train
    metrics_path = run_dir / "metrics.jsonl"
    metrics_f = metrics_path.open("a", encoding="utf-8")
    model.train()
    t0 = time.time()
    losses_window: list[float] = []
    log_every = train_cfg.get("log_every", 25)
    eval_every = train_cfg.get("eval_every", 200)
    ckpt_every = train_cfg.get("ckpt_every", 1000)
    last_step = 0  # init defensivo: si exception antes del loop, finally no crashea

    try:
        for step in range(max_steps):
            last_step = step + 1
            lr = cosine_lr(step, train_cfg["warmup_steps"], max_steps, train_cfg["lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            batch = loader.next_train_batch(device)
            ids = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    _, loss, _ = model(ids, targets=targets)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                gn = nn_utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                _, loss, _ = model(ids, targets=targets)
                loss.backward()
                gn = nn_utils.clip_grad_norm_(
                    model.parameters(), train_cfg["grad_clip"]
                )
                optimizer.step()

            loss_val = float(loss.item())
            losses_window.append(loss_val)
            if len(losses_window) > 100:
                losses_window.pop(0)

            if (step + 1) % log_every == 0:
                avg = sum(losses_window) / len(losses_window)
                elapsed = time.time() - t0
                rec = {
                    "type": "train",
                    "step": step + 1,
                    "loss": round(loss_val, 4),
                    "avg_loss": round(avg, 4),
                    "ppl": round(math.exp(min(avg, 20.0)), 2),
                    "lr": round(lr, 6),
                    "grad_norm": round(float(gn), 3),
                    "steps_per_sec": round((step + 1) / elapsed, 2),
                    "elapsed_min": round(elapsed / 60, 2),
                }
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()

            if (step + 1) % eval_every == 0 or (step + 1) == max_steps:
                e_loss = eval_loss(model, loader, device)
                rec = {
                    "type": "eval",
                    "step": step + 1,
                    "eval_loss": round(e_loss, 4),
                    "eval_ppl": round(math.exp(min(e_loss, 20.0)), 2),
                }
                metrics_f.write(json.dumps(rec) + "\n")
                metrics_f.flush()
                logger.info(
                    "[%s/seed%d] step=%d eval_loss=%.4f ppl=%.1f",
                    name,
                    seed,
                    step + 1,
                    e_loss,
                    math.exp(min(e_loss, 20.0)),
                )

            if (step + 1) % ckpt_every == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "step": step + 1,
                        "cfg": dataclasses.asdict(resolved_cfg),
                    },
                    run_dir / "checkpoint.pt",
                )

    except KeyboardInterrupt:
        logger.warning("Interrumpido por usuario — guardando final")
    finally:
        metrics_f.close()
        torch.save(
            {
                "model": model.state_dict(),
                "step": last_step,
                "cfg": dataclasses.asdict(resolved_cfg),
            },
            run_dir / "final.pt",
        )
        logger.info("Guardado en %s", run_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
