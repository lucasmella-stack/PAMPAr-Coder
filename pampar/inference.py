# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
pampar.inference — Servidor de inferencia para la extensión VS Code.

Protocolo JSON-lines (stdin/stdout):
  Entrada:
    { "type": "infer", "prompt": "...", "max_tokens": 256, "temperature": 0.4 }
    { "type": "boot",  "workspace": "/ruta/al/workspace" }
  Salida:
    { "type": "infer_ok",  "text": "..." }
    { "type": "boot_ok",   "agents_md": "..." }
    { "type": "ready" }
    { "type": "error",     "message": "..." }

Señal de listo:
  Escribe "READY" a stderr una vez que el modelo está cargado en memoria.

Uso:
  python -m pampar.inference --checkpoint checkpoints/v3_sft_v8.pt --device auto
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import traceback
from pathlib import Path

import torch

from pampar.constants import TOKENIZER_PATH


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            _stderr("ADVERTENCIA: CUDA solicitado pero no disponible. Usando CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "cpu":
        return torch.device("cpu")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _respond(obj: dict) -> None:
    print(json.dumps(obj, ensure_ascii=False), flush=True)


def _resolve_tokenizer_path(checkpoint_path: Path, tokenizer_arg: str | None) -> Path:
    """Intenta encontrar el tokenizer en ubicaciones conocidas."""
    candidates: list[Path] = []
    if tokenizer_arg:
        candidates.append(Path(tokenizer_arg))

    project_root = checkpoint_path.parent.parent
    candidates += [
        project_root / "data" / "tokenizer" / "pampar_48k.model",
        Path(TOKENIZER_PATH),
        Path("pampar_48k.model"),
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"Tokenizer no encontrado. Candidatos: {[str(c) for c in candidates]}"
    )


# ---------------------------------------------------------------------------
# Carga del modelo
# ---------------------------------------------------------------------------


def load_model(checkpoint_path: Path, device: torch.device):
    """Carga PamparV3 desde un checkpoint .pt."""
    import sentencepiece as spm

    from pampar.coder.v3.config import PRESET_V3
    from pampar.coder.v3.modelo import PamparV3

    tokenizer_path = _resolve_tokenizer_path(checkpoint_path, None)

    _stderr(f"Cargando tokenizer: {tokenizer_path}")
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(tokenizer_path))

    _stderr(f"Cargando modelo: {checkpoint_path}")
    model = PamparV3(PRESET_V3).to(device)

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    state_dict = ckpt.get("modelo", ckpt.get("model", ckpt))
    model.load_state_dict(state_dict, strict=False)
    model.registrar_tokenizer(tokenizer)
    model.eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    _stderr(f"Modelo listo: {params:.1f}M params en {device}")

    return model, tokenizer


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def handle_infer(model, tokenizer, device: torch.device, msg: dict) -> None:
    prompt: str = msg.get("prompt", "")
    max_tokens: int = int(msg.get("max_tokens", 256))
    temperature: float = float(msg.get("temperature", 0.4))

    if not prompt:
        _respond({"type": "error", "message": "prompt vacío"})
        return

    ids = tokenizer.Encode(prompt, out_type=int)
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    # Decodificar solo los tokens nuevos
    new_ids = output[0, len(ids) :].tolist()
    text = tokenizer.Decode(new_ids).replace("\u2047", "\n")
    _respond({"type": "infer_ok", "text": text})


def handle_boot(msg: dict) -> None:
    workspace: str = msg.get("workspace", ".")

    from pampar.runtime.generar_agents import generar_agents_md
    from pampar.runtime.scanner import Scanner

    try:
        scanner = Scanner(workspace_root=workspace)
        scan = scanner.scan()
        agents_md = generar_agents_md(scan, proyecto=Path(workspace).name)
        _respond({"type": "boot_ok", "agents_md": agents_md})
    except Exception as exc:
        _respond({"type": "error", "message": f"boot falló: {exc}"})


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> None:
    # Forzar UTF-8 en stdin/stdout — necesario en Windows (charmap por defecto)
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )
    if hasattr(sys.stdin, "buffer"):
        sys.stdin = io.TextIOWrapper(
            sys.stdin.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(description="PAMPAr inference server (JSON-lines)")
    parser.add_argument(
        "--checkpoint", required=True, help="Ruta al .pt del checkpoint"
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Dispositivo de inferencia",
    )
    parser.add_argument("--tokenizer", default=None, help="Ruta al tokenizer .model")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        _stderr(f"ERROR: checkpoint no encontrado: {checkpoint_path}")
        sys.exit(1)

    try:
        model, tokenizer = load_model(checkpoint_path, device)
    except Exception as exc:
        _stderr(f"ERROR al cargar modelo: {exc}")
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

    # Señalar que el servidor está listo
    _stderr("READY")
    _respond({"type": "ready"})

    # Loop interactivo — readline() en vez de `for line in stdin`
    # para evitar buffering ahead del iterador en Windows.
    while True:
        raw_line = sys.stdin.readline()
        if not raw_line:
            break
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            _respond({"type": "error", "message": f"JSON inválido: {exc}"})
            continue

        msg_type = msg.get("type")
        try:
            if msg_type == "infer":
                handle_infer(model, tokenizer, device, msg)
            elif msg_type == "boot":
                handle_boot(msg)
            else:
                _respond({"type": "error", "message": f"Tipo desconocido: {msg_type}"})
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _respond({"type": "error", "message": str(exc)})


if __name__ == "__main__":
    main()
