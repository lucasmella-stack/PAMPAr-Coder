#!/usr/bin/env python3
# SPDX-License-Identifier: BUSL-1.1
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Test end-to-end de GhidraProbe + EngramaStream.

Carga el modelo v3_neuro_v9 (Score=85), ejecuta GhidraProbe para
diagnosticar el forward pass, captura engramas de ejemplos exitosos,
y mide el impacto de la inyección en la inferencia.

Uso:
    python scripts/test_engrama.py
    python scripts/test_engrama.py --checkpoint checkpoints/v3_neuro_v9.pt
    python scripts/test_engrama.py --save-banco memoria/banco_engrama.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import torch
import sentencepiece as spm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pampar.coder.v3 import PamparV3, PRESET_V3
from pampar.coder.v3.ghidra_probe import GhidraProbe
from pampar.coder.v3.engrama_stream import BancoEngrama, EngramaCapture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EJEMPLOS_TEST = [
    "### Instruction:\nWrite a function that returns the factorial of n.\n### Solution:\ndef factorial(n):\n    return 1 if n <= 1 else n * factorial(n - 1)",
    "### Instruction:\nWrite a function to check if a string is a palindrome.\n### Solution:\ndef is_palindrome(s):\n    return s == s[::-1]",
    "### Instruction:\nWrite a function to compute fibonacci.\n### Solution:\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
    "### Instruction:\nWrite a function to flatten a nested list.\n### Solution:\ndef flatten(lst):\n    result = []\n    for item in lst:\n        if isinstance(item, list):\n            result.extend(flatten(item))\n        else:\n            result.append(item)\n    return result",
    "### Instruction:\nWrite a function to count vowels in a string.\n### Solution:\ndef count_vowels(s):\n    return sum(1 for c in s.lower() if c in 'aeiou')",
]


def cargar_modelo(checkpoint_path: str) -> tuple[PamparV3, spm.SentencePieceProcessor]:
    """Carga modelo y tokenizer."""
    config = PRESET_V3
    model = PamparV3(config).to(DEVICE)
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(str(ROOT / config.tokenizer_path))
    model.registrar_tokenizer(tokenizer)

    ckpt = Path(checkpoint_path)
    if ckpt.exists():
        state = torch.load(ckpt, map_location=DEVICE, weights_only=False)
        if "modelo" in state:
            model.load_state_dict(state["modelo"])
        elif "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)
        print(f"Checkpoint cargado: {ckpt.name}")
    else:
        print(f"WARN: No se encontró {ckpt}, usando pesos aleatorios")

    model.eval()
    return model, tokenizer


def evaluar_calidad(texto_generado: str) -> float:
    """Evalúa calidad del código generado (0-1)."""
    # Extraer código después de ### Solution:
    marcador = "### Solution:"
    pos = texto_generado.find(marcador)
    if pos < 0:
        return 0.1
    codigo = texto_generado[pos + len(marcador):].strip()

    score = 0.0

    # ¿Tiene def?
    if "def " in codigo:
        score += 0.3

    # ¿Parsea como Python válido?
    try:
        ast.parse(codigo)
        score += 0.4
    except SyntaxError:
        score += 0.1

    # ¿Tiene return?
    if "return" in codigo:
        score += 0.2

    # ¿No está vacío?
    if len(codigo.strip()) > 10:
        score += 0.1

    return min(score, 1.0)


def test_ghidra_probe(
    model: PamparV3,
    tokenizer: spm.SentencePieceProcessor,
) -> None:
    """Ejecuta GhidraProbe en ejemplos de test."""
    print("\n" + "=" * 70)
    print("  FASE 1: GhidraProbe — Diagnóstico del Forward Pass")
    print("=" * 70)

    probe = GhidraProbe(model)

    for i, ejemplo in enumerate(EJEMPLOS_TEST[:3]):
        ids = tokenizer.Encode(ejemplo)[:256]
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        probe.reset()
        with torch.no_grad():
            logits, loss, info = model(input_ids[:, :-1], targets=input_ids[:, 1:])

        print(f"\n{'─' * 70}")
        print(f"  Ejemplo {i + 1}: {ejemplo[:60]}...")
        print(f"  Loss: {loss.item():.4f}")
        probe.print_summary()

        # Mostrar trayectoria de routing del primer token interesante
        tokens = [tokenizer.IdToPiece(t) for t in ids[:10]]
        print(f"\n  Primeros 10 tokens: {tokens}")
        for t_idx in [0, 1, 2]:
            traj = probe.routing_trajectory(t_idx)
            if traj:
                traj_str = " → ".join(
                    f"[{','.join(f'{v:.2f}' for v in step)}]"
                    for step in traj
                )
                print(f"  Token {t_idx} ({tokens[t_idx]!r}) routing: {traj_str}")

    probe.detach()
    print("\nGhidraProbe detachado.")


def test_engrama_captura(
    model: PamparV3,
    tokenizer: spm.SentencePieceProcessor,
    banco: BancoEngrama,
) -> None:
    """Captura engramas de ejemplos exitosos."""
    print("\n" + "=" * 70)
    print("  FASE 2: EngramaStream — Captura de Activaciones")
    print("=" * 70)

    probe = GhidraProbe(model)
    capture = EngramaCapture(banco, score_minimo=0.5)
    total_engramas = 0

    for i, ejemplo in enumerate(EJEMPLOS_TEST):
        ids = tokenizer.Encode(ejemplo)[:256]
        input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

        probe.reset()
        with torch.no_grad():
            logits, loss, info = model(input_ids[:, :-1], targets=input_ids[:, 1:])

        # Evaluar calidad
        score = evaluar_calidad(ejemplo)

        # Capturar engramas si la calidad es suficiente
        n = capture.capturar_desde_probe(probe._raw, score, n_levels=5)
        total_engramas += n
        print(f"  Ejemplo {i + 1}: score={score:.2f}, engramas capturados={n}")

    probe.detach()

    # Stats del banco
    stats = banco.stats()
    print(f"\n  Banco stats: {json.dumps(stats, indent=2, default=str)}")
    print(f"  Total engramas en banco: {banco.total_engramas}")


def test_engrama_inyeccion(
    model: PamparV3,
    tokenizer: spm.SentencePieceProcessor,
    banco: BancoEngrama,
) -> None:
    """Compara inferencia con y sin inyección de engramas."""
    print("\n" + "=" * 70)
    print("  FASE 3: EngramaStream — Inyección en Inferencia")
    print("=" * 70)

    prompt = "### Instruction:\nWrite a function to reverse a string.\n### Solution:\n"
    ids = tokenizer.Encode(prompt)
    input_ids = torch.tensor([ids], dtype=torch.long, device=DEVICE)

    # Sin engramas
    print("\n  --- Sin EngramaStream ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits_sin, loss_sin, _ = model(
            input_ids[:, :-1], targets=input_ids[:, 1:]
        )
    t_sin = time.perf_counter() - t0

    gen_sin = model.generate(input_ids, max_tokens=80, temperature=0.7)
    texto_sin = tokenizer.Decode(gen_sin[0].tolist())
    print(f"  Loss: {loss_sin.item():.4f}")
    print(f"  Tiempo: {t_sin * 1000:.1f}ms")
    print(f"  Output: {texto_sin[:300]}")

    # Con engramas
    print("\n  --- Con EngramaStream ---")
    t0 = time.perf_counter()
    with torch.no_grad():
        logits_con, loss_con, _ = model(
            input_ids[:, :-1], targets=input_ids[:, 1:],
            banco_engrama=banco,
        )
    t_con = time.perf_counter() - t0

    gen_con = model.generate(
        input_ids, max_tokens=80, temperature=0.7,
        banco_engrama=banco,
    )
    texto_con = tokenizer.Decode(gen_con[0].tolist())
    print(f"  Loss: {loss_con.item():.4f}")
    print(f"  Tiempo: {t_con * 1000:.1f}ms")
    print(f"  Output: {texto_con[:300]}")

    # Comparación
    print("\n  --- Comparación ---")
    diff_loss = loss_con.item() - loss_sin.item()
    print(f"  ΔLoss: {diff_loss:+.4f} ({'mejor' if diff_loss < 0 else 'peor'})")
    print(f"  Inyecciones realizadas: {banco.total_inyecciones}")

    # Distribución de logits
    probs_sin = torch.softmax(logits_sin[0, -1], dim=-1)
    probs_con = torch.softmax(logits_con[0, -1], dim=-1)
    entropy_sin = -(probs_sin * probs_sin.log().clamp(min=-100)).sum().item()
    entropy_con = -(probs_con * probs_con.log().clamp(min=-100)).sum().item()
    print(f"  Entropy sin engrama: {entropy_sin:.2f}")
    print(f"  Entropy con engrama: {entropy_con:.2f}")
    print(f"  ΔEntropy: {entropy_con - entropy_sin:+.2f} ({'más certero' if entropy_con < entropy_sin else 'más disperso'})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Test GhidraProbe + EngramaStream")
    parser.add_argument(
        "--checkpoint",
        default=str(ROOT / "checkpoints" / "v3_neuro_v9.pt"),
        help="Ruta al checkpoint del modelo",
    )
    parser.add_argument(
        "--save-banco",
        default="",
        help="Ruta para guardar el banco de engramas (JSON)",
    )
    parser.add_argument(
        "--load-banco",
        default="",
        help="Ruta para cargar un banco existente",
    )
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {args.checkpoint}")

    model, tokenizer = cargar_modelo(args.checkpoint)

    # Crear o cargar banco
    banco = BancoEngrama(dim=PRESET_V3.dim, max_engramas_por_clave=10)
    if args.load_banco:
        n = banco.cargar(Path(args.load_banco))
        print(f"Banco cargado: {n} engramas desde {args.load_banco}")

    # Fase 1: Diagnóstico
    test_ghidra_probe(model, tokenizer)

    # Fase 2: Captura
    test_engrama_captura(model, tokenizer, banco)

    # Fase 3: Inyección
    test_engrama_inyeccion(model, tokenizer, banco)

    # Guardar banco si se pidió
    if args.save_banco:
        ruta = Path(args.save_banco)
        banco.guardar(ruta)
        print(f"\nBanco guardado en {ruta} ({banco.total_engramas} engramas)")

    print("\n✓ Test completo.")


if __name__ == "__main__":
    main()
