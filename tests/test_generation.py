# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2026 Lucas Ricardo Mella Chillemi
"""
Prueba de generación de código con PAMPAr-Coder.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import sentencepiece as spm
from pampar.coder import crear_modelo

def main():
    print("=" * 60)
    print("  PAMPAr-Coder - Test de Generacion")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")
    
    # Cargar modelo
    print("\n  Cargando modelo...")
    model = crear_modelo("4GB")
    
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "stable_best.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        print(f"    Checkpoint: {checkpoint_path.name}")
        print(f"    Loss: {checkpoint.get('loss', 'N/A')}")
    else:
        print("    [WARN] No checkpoint found, using random weights")
    
    model = model.to(device)
    model.eval()
    model.training = True  # Hack: evitar uso de cache para generación simple
    
    # Cargar tokenizer
    tokenizer_path = Path(__file__).parent.parent / "data" / "tokenizer" / "code_tokenizer.model"
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
        
        # Generar
        with torch.no_grad():
            generated = input_tensor
            
            for _ in range(100):  # Max 100 tokens
                # Solo pasar los últimos 256 tokens para evitar overflow
                if generated.shape[1] > 256:
                    context = generated[:, -256:]
                else:
                    context = generated
                
                logits, _ = model(context)
                next_logits = logits[:, -1, :]
                
                # Sampling con temperatura
                temperature = 0.8
                probs = torch.softmax(next_logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop en newline doble o max length
                if generated.shape[1] > len(input_ids) + 80:
                    break
        
        # Decodificar
        output_ids = generated[0].tolist()
        output_text = tokenizer.decode(output_ids)
        print(output_text)
    
    print("\n" + "=" * 60)
    print("  TEST COMPLETADO")
    print("=" * 60)


if __name__ == "__main__":
    main()
