#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
ablation_train.py — Entrenamiento simplificado para ablación científica.

Diferencias con train_v3.py:
  - Sin MotorCuriosidad (iteración lineal sobre datos)
  - Sin ReplayPareto (todos los modelos ven los mismos datos en el mismo orden)
  - Logging estructurado a JSON para comparación
  - 4 configuraciones de ablación:
    1. pampar_v3      → PRESET_V3 completo (control)
    2. no_llaves      → peso_llaves=0.0 (solo routing aprendido)
    3. single_stream  → n_streams=1 (sin estructura 2D)
    4. vanilla_gpt    → GPT estándar ~62M params

Uso:
  python scripts/ablation_train.py --experiment pampar_v3 --max-pasos 30000
  python scripts/ablation_train.py --experiment vanilla_gpt --max-pasos 30000
  python scripts/ablation_train.py --all --max-pasos 30000
"""

import argparse
import dataclasses
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PRESET_V3, ConfigV3, PamparV3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ablation")

# ── Configuraciones de ablación ──────────────────────────────────────────────

EXPERIMENTS: dict[str, dict[str, Any]] = {
    "pampar_v3": {
        "desc": "PAMPAr-V3 completo (control)",
        "model": "pampar",
        "config_overrides": {},
    },
    "no_llaves": {
        "desc": "Sin LLAVES — solo routing aprendido",
        "model": "pampar",
        "config_overrides": {"peso_llaves": 0.0},
    },
    "single_stream": {
        "desc": "1 stream — sin estructura 2D",
        "model": "pampar",
        "config_overrides": {"n_streams": 1, "n_territorios": 1},
    },
    "vanilla_gpt": {
        "desc": "GPT estándar ~62M params",
        "model": "vanilla",
        "config_overrides": {},
    },
}


# ── Data loader simplificado ────────────────────────────────────────────────


class SimpleDataLoader:
    """
    Cargador determinístico para ablación.

    Lee todos los JSONL de la biblioteca en orden fijo,
    tokeniza y sirve batches [B, L+1] cíclicamente.
    Almacena todos los tokens en un array numpy contiguo para eficiencia de memoria.
    """

    def __init__(
        self,
        biblioteca: Path,
        tokenizer: spm.SentencePieceProcessor,
        batch_size: int = 16,
        seq_len: int = 512,
        seed: int = 42,
    ):
        self._tok = tokenizer
        self._batch_size = batch_size
        self._seq_len = seq_len + 1  # +1 para target
        self._idx = 0

        logger.info("Cargando biblioteca desde %s...", biblioteca)
        all_ids = self._load_biblioteca(biblioteca)

        # Crear chunks como numpy array 2D contiguo (mucho menos memoria que list[list[int]])
        stride = self._seq_len // 2
        n_chunks = max(0, (len(all_ids) - self._seq_len) // stride)
        logger.info(
            "Creando %d chunks (seq_len=%d, stride=%d) desde %d tokens...",
            n_chunks,
            self._seq_len,
            stride,
            len(all_ids),
        )

        # Construir array [n_chunks, seq_len] directamente con numpy
        self._data = np.empty((n_chunks, self._seq_len), dtype=np.int32)
        ids_array = np.array(all_ids, dtype=np.int32)
        del all_ids  # Liberar la lista Python
        for i in range(n_chunks):
            start = i * stride
            self._data[i] = ids_array[start : start + self._seq_len]
        del ids_array  # Liberar el array temporal

        # Shuffle determinístico
        rng = np.random.RandomState(seed)
        self._order = rng.permutation(n_chunks)

        logger.info(
            "Biblioteca: %d chunks de %d tokens (%.1f MB en disco)",
            n_chunks,
            self._seq_len,
            self._data.nbytes / 1e6,
        )

    def _load_biblioteca(self, biblioteca: Path) -> list[int]:
        """Carga todos los JSONL de la biblioteca y devuelve token IDs."""
        indice_path = biblioteca / "indice.json"
        if not indice_path.exists():
            raise FileNotFoundError(f"Índice no encontrado: {indice_path}")

        with indice_path.open(encoding="utf-8") as f:
            indice = json.load(f)

        archivos: list[Path] = []
        for _cat, temas in indice.items():
            if not isinstance(temas, list):
                continue
            for t in temas:
                archivo = t.get("archivo", "")
                if archivo:
                    ruta = biblioteca / archivo
                    if ruta.exists():
                        archivos.append(ruta)

        archivos.sort()  # Orden determinístico
        logger.info("Encontrados %d archivos JSONL", len(archivos))

        all_ids: list[int] = []
        t0 = time.time()
        for i, ruta in enumerate(archivos):
            self._tokenize_file(ruta, all_ids)
            if (i + 1) % 20 == 0 or i == len(archivos) - 1:
                elapsed = time.time() - t0
                logger.info(
                    "  Tokenizando: %d/%d archivos — %.1fM tokens — %.0fs",
                    i + 1,
                    len(archivos),
                    len(all_ids) / 1e6,
                    elapsed,
                )

        return all_ids

    def _tokenize_file(self, ruta: Path, all_ids: list[int]) -> int:
        """Tokeniza un JSONL y acumula token IDs. Retorna tokens añadidos."""
        n_before = len(all_ids)
        with ruta.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    text = obj.get("text", obj.get("content", ""))
                    if not text and "instruction" in obj:
                        text = obj["instruction"] + "\n" + obj.get("output", "")
                    if text:
                        ids = self._tok.Encode(text)
                        all_ids.extend(ids)
                except (json.JSONDecodeError, AttributeError):
                    continue
        return len(all_ids) - n_before

    def get_batch(self, device: torch.device) -> torch.Tensor:
        """Devuelve un batch [B, seq_len] cíclico."""
        indices = []
        for _ in range(self._batch_size):
            indices.append(self._order[self._idx % len(self._order)])
            self._idx += 1
        batch_np = self._data[indices]
        return torch.from_numpy(batch_np.astype(np.int64)).to(device)

    def reset(self) -> None:
        """Resetea el índice al inicio (misma secuencia para fair comparison)."""
        self._idx = 0

    @property
    def n_chunks(self) -> int:
        return len(self._data)

    @property
    def epoch_steps(self) -> int:
        return len(self._data) // self._batch_size


# ── Modelo factory ───────────────────────────────────────────────────────────


def _create_model(
    experiment: str, device: torch.device
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Crea el modelo según el experimento."""
    spec = EXPERIMENTS[experiment]

    if spec["model"] == "vanilla":
        from vanilla_gpt import VanillaGPT, VanillaGPTConfig

        cfg = VanillaGPTConfig()
        model = VanillaGPT(cfg).to(device)
        config_dict = dataclasses.asdict(cfg)
    else:
        overrides = spec["config_overrides"]
        cfg = ConfigV3(**{**dataclasses.asdict(PRESET_V3), **overrides})
        model = PamparV3(cfg).to(device)
        config_dict = dataclasses.asdict(cfg)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "[%s] %s — %.1fM params",
        experiment,
        spec["desc"],
        n_params / 1e6,
    )
    config_dict["n_params"] = n_params
    return model, config_dict


# ── Eval ─────────────────────────────────────────────────────────────────────


@torch.no_grad()
def _eval_loss(
    model: torch.nn.Module,
    loader: SimpleDataLoader,
    device: torch.device,
    n_batches: int = 20,
) -> float:
    """Calcula loss promedio sobre n_batches (sin gradiente)."""
    model.eval()
    total_loss = 0.0
    saved_idx = loader._idx  # noqa: SLF001

    for _ in range(n_batches):
        tokens = loader.get_batch(device)
        input_ids = tokens[:, :-1]
        targets = tokens[:, 1:]
        _, loss, _ = model(input_ids, targets=targets)
        total_loss += loss.item()

    loader._idx = saved_idx  # noqa: SLF001 — restaurar posición
    model.train()
    return total_loss / n_batches


# ── Training loop ────────────────────────────────────────────────────────────


def train_experiment(
    experiment: str,
    args: argparse.Namespace,
    device: torch.device,
    loader: "SimpleDataLoader",
) -> Path:
    """Entrena un experimento y devuelve la ruta del log JSON."""
    spec = EXPERIMENTS[experiment]
    logger.info("━" * 60)
    logger.info("Iniciando: %s — %s", experiment, spec["desc"])
    logger.info("━" * 60)

    # ── Modelo
    model, config_dict = _create_model(experiment, device)

    # ── Resetear data loader al mismo punto para fair comparison
    loader.reset()

    # ── Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8,
    )

    # ── Log file
    log_dir = args.output / experiment
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "metrics.jsonl"
    ckpt_path = log_dir / "checkpoint.pt"

    # ── Resumir si existe
    start_step = 0
    if ckpt_path.exists():
        logger.info("Reanudando desde %s", ckpt_path)
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["modelo"])
        optimizer.load_state_dict(payload["optimizer"])
        start_step = payload["paso"]
        loader._idx = payload.get("data_idx", 0)  # noqa: SLF001
        logger.info("Reanudado — paso %d", start_step)

    # ── Metadata
    meta = {
        "experiment": experiment,
        "desc": spec["desc"],
        "config": config_dict,
        "args": {
            "lr": args.lr,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "max_pasos": args.max_pasos,
        },
        "n_chunks": loader.n_chunks,
        "start_step": start_step,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (log_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # ── Training loop
    model.train()
    t0 = time.time()
    losses_window: list[float] = []

    log_file = log_path.open("a", encoding="utf-8")

    try:
        for step in range(start_step, args.max_pasos):
            tokens = loader.get_batch(device)
            input_ids = tokens[:, :-1]
            targets = tokens[:, 1:]

            optimizer.zero_grad(set_to_none=True)
            _, loss, info = model(input_ids, targets=targets)
            loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            loss_val = loss.item()
            losses_window.append(loss_val)
            if len(losses_window) > 100:
                losses_window.pop(0)

            # ── Log cada 50 pasos
            if (step + 1) % 50 == 0:
                avg_loss = sum(losses_window) / len(losses_window)
                elapsed = time.time() - t0
                steps_per_sec = (step - start_step + 1) / elapsed
                ppl = math.exp(min(avg_loss, 20.0))

                record = {
                    "step": step + 1,
                    "loss": round(loss_val, 4),
                    "avg_loss": round(avg_loss, 4),
                    "ppl": round(ppl, 1),
                    "steps_per_sec": round(steps_per_sec, 2),
                    "elapsed_min": round(elapsed / 60, 1),
                }

                log_file.write(json.dumps(record) + "\n")
                log_file.flush()

                if (step + 1) % 500 == 0:
                    logger.info(
                        "[%s] paso %d/%d | loss=%.3f avg=%.3f ppl=%.1f | %.1f steps/s",
                        experiment,
                        step + 1,
                        args.max_pasos,
                        loss_val,
                        avg_loss,
                        ppl,
                        steps_per_sec,
                    )

            # ── Eval cada 2000 pasos
            if (step + 1) % 2000 == 0:
                eval_loss = _eval_loss(model, loader, device)
                eval_ppl = math.exp(min(eval_loss, 20.0))
                record = {
                    "step": step + 1,
                    "eval_loss": round(eval_loss, 4),
                    "eval_ppl": round(eval_ppl, 1),
                    "type": "eval",
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                logger.info(
                    "[%s] EVAL paso %d | eval_loss=%.3f eval_ppl=%.1f",
                    experiment,
                    step + 1,
                    eval_loss,
                    eval_ppl,
                )

            # ── Checkpoint cada 5000 pasos
            if (step + 1) % 5000 == 0:
                torch.save(
                    {
                        "modelo": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "paso": step + 1,
                        "data_idx": loader._idx,  # noqa: SLF001
                        "config": config_dict,
                    },
                    ckpt_path,
                )
                logger.info("[%s] Checkpoint guardado — paso %d", experiment, step + 1)

    except KeyboardInterrupt:
        logger.info("Interrumpido — guardando checkpoint...")
    finally:
        log_file.close()
        torch.save(
            {
                "modelo": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "paso": step + 1,  # noqa: F821
                "data_idx": loader._idx,  # noqa: SLF001
                "config": config_dict,
            },
            ckpt_path,
        )
        logger.info("[%s] Checkpoint final guardado — paso %d", experiment, step + 1)

    # ── Eval final
    eval_loss = _eval_loss(model, loader, device, n_batches=50)
    eval_ppl = math.exp(min(eval_loss, 20.0))
    logger.info(
        "[%s] FINAL | eval_loss=%.3f eval_ppl=%.1f",
        experiment,
        eval_loss,
        eval_ppl,
    )

    return log_path


# ── CLI ──────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Ablation training suite para PAMPAr-V3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment",
        type=str,
        choices=list(EXPERIMENTS.keys()),
        help="Experimento individual",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Ejecutar los 4 experimentos secuencialmente",
    )
    p.add_argument(
        "--tokenizer",
        type=Path,
        default=ROOT / "data" / "tokenizer" / "pampar_48k.model",
    )
    p.add_argument(
        "--biblioteca",
        type=Path,
        default=ROOT / "biblioteca",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=ROOT / "ablation_results",
        help="Directorio para logs y checkpoints",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-pasos", type=int, default=30_000)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    if not args.experiment and not args.all:
        logger.error("Debes especificar --experiment o --all")
        sys.exit(1)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    logger.info("Device: %s", device)
    if device.type == "cuda":
        logger.info(
            "GPU: %s (%.1f GiB)",
            torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

    # ── Tokenizer
    if not args.tokenizer.exists():
        logger.error("Tokenizer no encontrado: %s", args.tokenizer)
        sys.exit(1)
    tok = spm.SentencePieceProcessor()
    tok.Load(str(args.tokenizer))
    logger.info("Tokenizer: vocab=%d", tok.GetPieceSize())

    # ── Experiments
    experiments = list(EXPERIMENTS.keys()) if args.all else [args.experiment]

    # ── Crear data loader UNA sola vez (tokenización ~11 min para 865M tokens)
    loader = SimpleDataLoader(
        biblioteca=args.biblioteca,
        tokenizer=tok,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        seed=42,
    )

    results: dict[str, str] = {}
    for exp in experiments:
        log_path = train_experiment(exp, args, device, loader)
        results[exp] = str(log_path)
        torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Resumen
    logger.info("━" * 60)
    logger.info("ABLACIÓN COMPLETA")
    for exp, path in results.items():
        logger.info("  %s → %s", exp, path)
    logger.info("━" * 60)


if __name__ == "__main__":
    main()
