#!/usr/bin/env python3
"""
Prueba interactiva del modelo PAMPAr.

Uso:
    python scripts/probar_modelo.py
    python scripts/probar_modelo.py --checkpoint checkpoints/pampar_v2_best.pt
    python scripts/probar_modelo.py --prompt "def fibonacci("
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


PROMPTS_TEST = [
    # Nivel 1 — básico
    "def suma(a, b):",
    "# Calcular el factorial de n\ndef factorial(n):",
    "lista = [1, 2, 3, 4, 5]\n# Filtrar solo los pares\n",
    # Nivel 2 — intermedio
    "class Pila:\n    def __init__(self):",
    "def busqueda_binaria(arr, objetivo):",
    # Nivel 3 — avanzado
    "from typing import Generator\n\ndef fibonacci() -> Generator[int, None, None]:",
]


def cargar_modelo(checkpoint: Path, device: torch.device):
    from pampar.coder.v2.modelo import PampaRCoderV2
    from pampar.coder.v2.config import (
        ConfigV2, PRESET_1_5B, PRESET_4GB, PRESET_8GB, PRESET_24GB,
    )
    import dataclasses

    PRESET_MAP = {"4GB": PRESET_4GB, "8GB": PRESET_8GB,
                  "24GB": PRESET_24GB, "1_5B": PRESET_1_5B}

    config = PRESET_4GB  # default seguro
    if checkpoint.exists():
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        raw_cfg = ckpt.get("config", {})
        state = ckpt.get("modelo", ckpt.get("model", ckpt))

        if isinstance(raw_cfg, ConfigV2):
            config = raw_cfg
        elif isinstance(raw_cfg, dict):
            preset_name = raw_cfg.get("preset")
            if preset_name in PRESET_MAP:
                preset = PRESET_MAP[preset_name]
                emb = state.get("tok_emb.weight")
                if emb is not None and emb.shape[1] != preset.dim:
                    config = ConfigV2(vocab_size=int(emb.shape[0]), dim=int(emb.shape[1]))
                else:
                    config = preset
            else:
                valid = {f.name for f in dataclasses.fields(ConfigV2)}
                filtered = {k: v for k, v in raw_cfg.items() if k in valid}
                if filtered:
                    config = ConfigV2(**filtered)

        modelo = PampaRCoderV2(config).to(device)
        modelo.load_state_dict(state, strict=False)
        modelo.eval()
        print(f"  Modelo: {sum(p.numel() for p in modelo.parameters())/1e6:.0f}M params | vocab={config.vocab_size}")
        return modelo, config
    else:
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint}")


def cargar_tokenizer(vocab_size: int):
    import sentencepiece as spm
    tok_map = {
        16000: Path("data/tokenizer/code_tokenizer.model"),
        48000: Path("data/tokenizer/pampar_48k.model"),
    }
    tok_path = tok_map.get(vocab_size, Path("data/tokenizer/pampar_48k.model"))
    if not tok_path.exists():
        raise FileNotFoundError(f"Tokenizer no encontrado: {tok_path}")
    tok = spm.SentencePieceProcessor()
    tok.Load(str(tok_path))
    return tok


def generar(modelo, tok, prompt: str, max_tokens: int = 150,
            temperature: float = 0.8, device=None) -> str:
    import torch.nn.functional as F

    ids = tok.Encode(prompt)
    input_ids = torch.tensor([ids], device=device)

    generated = list(ids)
    with torch.no_grad():
        for _ in range(max_tokens):
            # Ventana de contexto
            ctx = torch.tensor([generated[-512:]], device=device)
            out = modelo(ctx)
            logits = out[0] if isinstance(out, (tuple, list)) else out
            next_logits = logits[0, -1] / temperature

            # Muestra del siguiente token
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)

            # Parar en EOS o newline doble
            decoded = tok.Decode(generated[len(ids):])
            if "\n\n" in decoded and len(decoded) > 30:
                break

    return tok.Decode(generated)


def evaluar_automatico(modelo, tok, device) -> dict:
    """Pruebas rápidas automáticas — no hay respuesta correcta, evaluamos coherencia."""
    import re

    resultados = []
    for prompt in PROMPTS_TEST:
        try:
            output = generar(modelo, tok, prompt, max_tokens=80, temperature=0.6, device=device)
            respuesta = output[len(prompt):]

            # Heurísticas de coherencia:
            tiene_codigo = bool(re.search(r'(return|if |for |def |=|print)', respuesta))
            tiene_indentacion = "    " in respuesta or "\t" in respuesta
            tiene_python = bool(re.search(r'[:()\[\]{}\'"#]', respuesta))
            longitud_ok = 10 < len(respuesta) < 300

            score = sum([tiene_codigo, tiene_indentacion, tiene_python, longitud_ok]) / 4

            resultados.append({
                "prompt": prompt[:40] + "...",
                "respuesta": respuesta[:80].replace("\n", "↵"),
                "score": score,
            })
        except Exception as e:
            resultados.append({"prompt": prompt[:40], "respuesta": f"ERROR: {e}", "score": 0})

    avg = sum(r["score"] for r in resultados) / len(resultados)
    return {"prompts": resultados, "score_medio": avg}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/pampar_v2_best.pt"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--auto", action="store_true", help="Evaluar automáticamente con prompts predefinidos")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"\n  PAMPAr -- Prueba del modelo")
    print(f"  Checkpoint: {args.checkpoint.name} | Device: {device}\n")

    modelo, config = cargar_modelo(args.checkpoint, device)
    tok = cargar_tokenizer(config.vocab_size)

    if args.auto or args.prompt is None:
        # Evaluación automática
        print("  Evaluacion automatica con 6 prompts...\n")
        resultado = evaluar_automatico(modelo, tok, device)
        for r in resultado["prompts"]:
            bar = "█" * int(r["score"] * 10) + "░" * (10 - int(r["score"] * 10))
            print(f"  [{bar}] {r['score']:.0%}  {r['prompt']}")
            print(f"         → {r['respuesta'][:70]}\n")

        score = resultado["score_medio"]
        print(f"  Score medio: {score:.0%}", end="  ")
        if score >= 0.8:
            print("EXCELENTE — el modelo genera código coherente")
        elif score >= 0.6:
            print("BIEN — sigue entrenando, ya hay estructura")
        elif score >= 0.4:
            print("REGULAR — reconoce patrones pero falla a veces")
        else:
            print("BAJO — necesita más entrenamiento (normal al inicio)")

        if args.prompt is None:
            return

    # Modo interactivo
    if args.prompt:
        prompts = [args.prompt]
    else:
        print("\n  Modo interactivo — escribe un prompt (o 'exit' para salir)\n")
        prompts = []

    if not prompts:
        while True:
            try:
                prompt = input("  >>> ").strip()
                if prompt.lower() in ("exit", "quit", "q"):
                    break
                if not prompt:
                    continue
                print()
                output = generar(modelo, tok, prompt, args.max_tokens, args.temperature, device)
                print(output)
                print()
            except KeyboardInterrupt:
                break
    else:
        for prompt in prompts:
            print(f"  Prompt: {prompt!r}\n")
            output = generar(modelo, tok, prompt, args.max_tokens, args.temperature, device)
            print(output)
            print()


if __name__ == "__main__":
    main()
