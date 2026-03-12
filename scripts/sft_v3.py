#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
sft_v3.py — Supervised Fine-Tuning de PamparV3 sobre instrucciones Python.

Carga el checkpoint de pretraining (paso 477K) y lo afina con ejemplos
instruction→code filtrados para Python únicamente.

Fuentes:
  - Magicoder-OSS-75K  (filtrado: solo ```python en ### Solution)
  - destilado_qwen     (convertido a ### Problem / ### Solution)

Uso:
  python scripts/sft_v3.py
  python scripts/sft_v3.py --lr 3e-5 --max-pasos 10000
  python scripts/sft_v3.py --checkpoint-in checkpoints/v3_train.pt

Detener limpiamente con Ctrl-C — guarda el checkpoint antes de salir.
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
import torch.nn.utils as nn_utils
import sentencepiece as spm

# ── Rutas relativas al script ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # PAMPAr-Coder/
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PamparV3, ConfigV3, PRESET_V3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sft_v3")


# ─────────────────────────────────────────────────────────────────────────────
# Filtrado y carga de datos
# ─────────────────────────────────────────────────────────────────────────────

_LANGS_NO_PYTHON = (
    "```cpp", "```c++", "```java", "```javascript", "```typescript",
    "```c\n", "```go", "```rust", "```swift", "```kotlin", "```ruby",
    "```php", "```scala", "```c#", "```csharp", "```html", "```sql",
)


def _es_python(texto: str) -> bool:
    """True si la sección ### Solution contiene código Python."""
    lower = texto.lower()
    idx = lower.find("### solution:")
    if idx < 0:
        return False
    sol = lower[idx:]
    for lang in _LANGS_NO_PYTHON:
        if lang in sol:
            return False
    if "```python" in sol:
        return True
    # Código sin tag de lenguaje — aceptar si parece Python
    if "```\n" in sol or "```\r" in sol:
        return "def " in texto or "import " in texto or "print(" in texto
    return False


def _cargar_magicoder(ruta: Path) -> list[str]:
    """Carga ejemplos Python-only de Magicoder JSONL."""
    ejemplos: list[str] = []
    n_total = 0
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        n_total += 1
        try:
            obj = json.loads(linea)
            texto = obj.get("text", "")
            if texto and _es_python(texto):
                ejemplos.append(texto)
        except json.JSONDecodeError:
            continue
    logger.info("Magicoder: %d/%d ejemplos son Python", len(ejemplos), n_total)
    return ejemplos


def _cargar_destilado(ruta: Path) -> list[str]:
    """Carga destilado_qwen y convierte a formato ### Problem / ### Solution."""
    ejemplos: list[str] = []
    if not ruta.exists():
        return ejemplos
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        try:
            obj = json.loads(linea)
            inst = obj.get("instruction", "")
            out = obj.get("output", "")
            if inst and out:
                texto = f"### Problem:\n{inst}\n### Solution:\n```python\n{out}\n```"
                ejemplos.append(texto)
        except json.JSONDecodeError:
            continue
    logger.info("Destilado: %d ejemplos cargados", len(ejemplos))
    return ejemplos


def _tokenizar(
    ejemplos: list[str],
    tok: spm.SentencePieceProcessor,
    max_seq_len: int,
) -> list[list[int]]:
    """Tokeniza ejemplos en chunks de max_seq_len+1 tokens."""
    chunks: list[list[int]] = []
    n = len(ejemplos)
    # ~4 chars/token heurística → cap de caracteres antes de tokenizar
    max_chars = max_seq_len * 6
    for i, texto in enumerate(ejemplos):
        if (i + 1) % 5000 == 0:
            logger.info("  tokenizando %d/%d ...", i + 1, n)
        # Truncar texto largo antes de tokenizar (evitar O(n²) en SPM)
        if len(texto) > max_chars:
            texto = texto[:max_chars]
        ids = tok.Encode(texto)
        if len(ids) > max_seq_len + 1:
            chunks.append(ids[: max_seq_len + 1])
        elif len(ids) >= 16:
            chunks.append(ids)
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hacer_batch(
    chunks: list[list[int]],
    indices: list[int],
    device: torch.device,
    max_seq_len: int,
) -> torch.Tensor:
    """Batch con padding a la derecha."""
    sels = [chunks[i] for i in indices]
    max_len = min(max(len(c) for c in sels), max_seq_len + 1)
    padded = []
    for c in sels:
        t = c[:max_len]
        padded.append(t + [0] * (max_len - len(t)))
    return torch.tensor(padded, dtype=torch.long, device=device)


def _paso(
    modelo: PamparV3,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
    max_grad_norm: float,
) -> float:
    """Forward + backward + step. Devuelve loss escalar."""
    modelo.train()
    optimizer.zero_grad(set_to_none=True)

    input_ids = tokens[:, :-1]
    targets = tokens[:, 1:]

    _, loss, _ = modelo(input_ids, targets=targets)
    loss.backward()
    nn_utils.clip_grad_norm_(modelo.parameters(), max_grad_norm)
    optimizer.step()

    return float(loss.detach())


def _cosine_lr(paso: int, warmup: int, total: int, lr_max: float, lr_min: float) -> float:
    """Warmup lineal → cosine decay."""
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
        "tipo": "sft",
    }, ruta)
    logger.info("✓ SFT checkpoint → paso %d", paso)


# ─────────────────────────────────────────────────────────────────────────────
# CLI + main
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SFT de PamparV3 sobre instrucciones Python",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint-in", type=Path,
                   default=ROOT / "checkpoints" / "v3_train.pt",
                   help="Checkpoint de pretraining a cargar")
    p.add_argument("--checkpoint-out", type=Path,
                   default=ROOT / "checkpoints" / "v3_sft.pt",
                   help="Checkpoint SFT a guardar")
    p.add_argument("--tokenizer", type=Path,
                   default=ROOT / "data" / "tokenizer" / "pampar_48k.model")
    p.add_argument("--magicoder", type=Path,
                   default=ROOT / "biblioteca" / "python_real" / "magicoder_oss_75k.jsonl")
    p.add_argument("--destilado", type=Path,
                   default=ROOT / "biblioteca" / "python_real" / "destilado_qwen.jsonl")

    p.add_argument("--lr", type=float, default=3e-5, help="LR máximo")
    p.add_argument("--lr-min", type=float, default=1e-6, help="LR mínimo (cosine)")
    p.add_argument("--warmup", type=int, default=200, help="Pasos de warmup")
    p.add_argument("--max-pasos", type=int, default=10000, help="Límite total de pasos")
    p.add_argument("--epochs", type=int, default=3, help="Nº de epochs")
    p.add_argument("--batch-size", type=int, default=2, help="Tamaño de batch")
    p.add_argument("--seq-len", type=int, default=512, help="Máxima longitud de secuencia")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="Clip de gradiente")
    p.add_argument("--guardar-cada", type=int, default=500, help="Pasos entre checkpoints")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ── Device ────────────────────────────────────────────────────────────────
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

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    if not args.tokenizer.exists():
        logger.error("Tokenizer no encontrado: %s", args.tokenizer)
        sys.exit(1)
    tok = spm.SentencePieceProcessor()
    tok.Load(str(args.tokenizer))
    logger.info("Tokenizer: vocab=%d", tok.GetPieceSize())

    # ── Modelo desde pretraining ──────────────────────────────────────────────
    if not args.checkpoint_in.exists():
        logger.error("Checkpoint no encontrado: %s", args.checkpoint_in)
        sys.exit(1)

    payload = torch.load(args.checkpoint_in, map_location=device, weights_only=False)
    config = ConfigV3(**payload["config"]) if "config" in payload else PRESET_V3
    modelo = PamparV3(config).to(device)
    modelo.load_state_dict(payload["modelo"])
    paso_pre = int(payload.get("paso_global", 0))
    del payload
    logger.info("Pretraining checkpoint cargado — paso %d", paso_pre)

    n_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    logger.info("PamparV3 — %.1fM parámetros", n_params / 1e6)

    if modelo.config.vocab_size != tok.GetPieceSize():
        logger.error("vocab_size mismatch: modelo=%d  tokenizer=%d",
                      modelo.config.vocab_size, tok.GetPieceSize())
        sys.exit(1)

    # ── Datos SFT ─────────────────────────────────────────────────────────────
    logger.info("Cargando datos SFT...")
    ejemplos: list[str] = []

    if args.magicoder.exists():
        ejemplos.extend(_cargar_magicoder(args.magicoder))
    else:
        logger.warning("Magicoder no encontrado: %s", args.magicoder)

    if args.destilado.exists():
        ejemplos.extend(_cargar_destilado(args.destilado))

    if not ejemplos:
        logger.error("Sin datos SFT — nada que entrenar")
        sys.exit(1)

    logger.info("Total ejemplos SFT: %d", len(ejemplos))

    chunks = _tokenizar(ejemplos, tok, args.seq_len)
    logger.info("Chunks tokenizados: %d", len(chunks))
    del ejemplos

    # ── Optimizer fresco (no restaurar el de pretraining) ─────────────────────
    optimizer = torch.optim.AdamW(
        modelo.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
        eps=1e-8,
    )

    # ── Bucle SFT ─────────────────────────────────────────────────────────────
    pasos_por_epoch = len(chunks) // args.batch_size
    total_pasos = min(args.max_pasos, args.epochs * pasos_por_epoch)
    logger.info(
        "SFT: %d pasos planificados | %d chunks | %d p/epoch | lr=%.1e→%.1e",
        total_pasos, len(chunks), pasos_por_epoch, args.lr, args.lr_min,
    )

    paso = 0
    t0 = time.time()
    losses: deque[float] = deque(maxlen=100)

    try:
        for epoch in range(args.epochs):
            idx = list(range(len(chunks)))
            random.shuffle(idx)
            logger.info("── Epoch %d/%d ──", epoch + 1, args.epochs)

            for i in range(0, len(idx) - args.batch_size + 1, args.batch_size):
                batch_idx = idx[i : i + args.batch_size]
                tokens = _hacer_batch(chunks, batch_idx, device, args.seq_len)

                # Cosine LR schedule
                lr = _cosine_lr(paso, args.warmup, total_pasos, args.lr, args.lr_min)
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                loss = _paso(modelo, optimizer, tokens, args.max_grad_norm)
                losses.append(loss)
                paso += 1

                if paso % 10 == 0:
                    avg = sum(losses) / len(losses)
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

            avg_ep = sum(losses) / len(losses)
            logger.info("── Epoch %d done | avg=%.3f  ppl=%.1f ──",
                         epoch + 1, avg_ep, math.exp(min(avg_ep, 10)))

    except KeyboardInterrupt:
        logger.info("\nInterrumpido — guardando...")

    # ── Guardar final ─────────────────────────────────────────────────────────
    _guardar(args.checkpoint_out, modelo, optimizer, paso)

    elapsed = time.time() - t0
    hh, mm = int(elapsed // 3600), int((elapsed % 3600) // 60)
    avg_final = sum(losses) / max(1, len(losses))

    print(f"\n── SFT Completado ──")
    print(f"  Pasos: {paso}")
    print(f"  Tiempo: {hh}h{mm:02d}m")
    print(f"  Loss final (avg100): {avg_final:.3f}")
    print(f"  PPL final: {math.exp(min(avg_final, 10)):.1f}")
    print(f"  Checkpoint: {args.checkpoint_out}")
    print(f"\n  Evaluar con:")
    print(f"    python -X utf8 scripts/eval_v3.py --checkpoint {args.checkpoint_out}")


if __name__ == "__main__":
    main()
