# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Prueba de generación de código con PAMPAr-Coder v2.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import sentencepiece as spm
from pampar.coder import crear_modelo, PRESET_4GB


def main():
    print("=" * 60)
    print("  PAMPAr-Coder v2 - Test de Generacion")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    # Cargar modelo v2
    print("\n  Cargando modelo...")
    model = crear_modelo(PRESET_4GB)

    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "stable_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"    Checkpoint: {checkpoint_path.name}")
        print(f"    Loss: {checkpoint.get('loss', 'N/A')}")
    else:
        print("    [WARN] No checkpoint found, using random weights")

    model = model.to(device)
    model.eval()

    # Cargar tokenizer
    tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizer" / "pampar_48k.model"
    if not tokenizer_path.exists():
        print(f"    [ERROR] Tokenizer not found: {tokenizer_path}")
        return
    tokenizer = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    model.registrar_tokenizer(tokenizer)

    # Prompts de prueba
    prompts = [
        "def fibonacci(n):",
        "def factorial(n):",
        "class Calculator:",
        "def bubble_sort(arr):",
        "# Function to check if a number is prime",
        "def hello_world():",
    ]

    print("\n" + "=" * 60)
    print("  GENERACION DE CODIGO")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n{'='*50}")
        print(f"Prompt: {prompt}")
        print("-" * 50)

        # Tokenizar prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        # Generar con model.generate() (v2 API)
        with torch.no_grad():
            generated = model.generate(
                input_tensor,
                max_tokens=80,
                temperature=0.8,
            )

        # Decodificar
        output_ids = generated[0].tolist()
        output_text = tokenizer.decode(output_ids)
        print(output_text)

    print("\n" + "=" * 60)
    print("  TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
