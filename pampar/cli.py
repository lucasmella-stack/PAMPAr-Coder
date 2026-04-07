# SPDX-License-Identifier: BUSL-1.1
"""
pampar.cli — Chat interactivo con PamparV3 en terminal.

Uso:
  python -m pampar.cli
  python -m pampar.cli --checkpoint checkpoints/v3_sft_v8.pt --device cuda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from pampar.inference import _resolve_device, _stderr, load_model

BANNER = r"""
╔═══════════════════════════════════════════╗
║       PAMPAr Coder v3 — Chat local       ║
║       108M params · Python · Local        ║
╠═══════════════════════════════════════════╣
║  Escribe tu pregunta y presiona Enter.    ║
║  Comandos: /exit /clear /device /help     ║
╚═══════════════════════════════════════════╝
"""

HELP = """
Comandos disponibles:
  /exit, /quit    Salir del chat
  /clear          Limpiar historial
  /device         Mostrar dispositivo actual
  /temp <valor>   Cambiar temperatura (ej: /temp 0.6)
  /tokens <n>     Cambiar max tokens (ej: /tokens 512)
  /help           Mostrar esta ayuda
"""


def find_checkpoint() -> Path | None:
    """Busca el mejor checkpoint automáticamente."""
    candidates = [
        Path("checkpoints/v3_sft_v8.pt"),
        Path("checkpoints/stable_best.pt"),
        Path("checkpoints/pampar_v2_best.pt"),
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_prompt(history: list[dict[str, str]], user_text: str) -> str:
    """Construye el prompt con historial (últimas 3 rondas)."""
    window = history[-6:]
    ctx = ""
    for msg in window:
        if msg["role"] == "user":
            ctx += f"### Problem:\n{msg['content']}\n"
        else:
            ctx += f"### Solution:\n{msg['content']}\n"
    return f"{ctx}### Problem:\n{user_text}\n### Solution:\n"


def generate(
    model: torch.nn.Module,
    tokenizer: object,
    device: torch.device,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.4,
) -> str:
    """Genera texto con el modelo."""
    ids = tokenizer.Encode(prompt, out_type=int)  # type: ignore[union-attr]
    input_tensor = torch.tensor([ids], dtype=torch.long, device=device)

    with torch.no_grad():
        output = model.generate(
            input_tensor,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    new_ids = output[0, len(ids) :].tolist()
    text = tokenizer.Decode(new_ids).replace("\u2047", "\n")  # type: ignore[union-attr]
    return text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="PAMPAr CLI Chat")
    parser.add_argument("--checkpoint", default=None, help="Ruta al .pt")
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.4)
    args = parser.parse_args()

    # Resolver checkpoint
    checkpoint_path: Path | None = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
    else:
        checkpoint_path = find_checkpoint()

    if not checkpoint_path or not checkpoint_path.exists():
        print("ERROR: No se encontró checkpoint.", file=sys.stderr)
        print("Usa: python -m pampar.cli --checkpoint <ruta>", file=sys.stderr)
        sys.exit(1)

    device = _resolve_device(args.device)
    max_tokens = args.max_tokens
    temperature = args.temperature

    # Cargar modelo
    print(f"Cargando modelo desde {checkpoint_path} en {device}...")
    model, tokenizer = load_model(checkpoint_path, device)
    print(BANNER)

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("\033[94m>>> \033[0m").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n¡Hasta luego!")
            break

        if not user_input:
            continue

        # Comandos
        if user_input.startswith("/"):
            cmd = user_input.lower().split()
            if cmd[0] in ("/exit", "/quit"):
                print("¡Hasta luego!")
                break
            elif cmd[0] == "/clear":
                history.clear()
                print("Historial limpiado.")
                continue
            elif cmd[0] == "/device":
                print(f"Device: {device}")
                continue
            elif cmd[0] == "/temp" and len(cmd) > 1:
                temperature = float(cmd[1])
                print(f"Temperatura: {temperature}")
                continue
            elif cmd[0] == "/tokens" and len(cmd) > 1:
                max_tokens = int(cmd[1])
                print(f"Max tokens: {max_tokens}")
                continue
            elif cmd[0] == "/help":
                print(HELP)
                continue
            else:
                print(f"Comando desconocido: {cmd[0]}. Usa /help")
                continue

        # Generar respuesta
        history.append({"role": "user", "content": user_input})
        prompt = build_prompt(history, user_input)

        print("\033[90mPensando...\033[0m", end="", flush=True)
        response = generate(model, tokenizer, device, prompt, max_tokens, temperature)
        print(f"\r\033[92m{response}\033[0m")

        history.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
