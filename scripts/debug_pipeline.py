"""Debug: analizar el pipeline tokenización→training del Classroom."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentencepiece as spm
import torch
import torch.nn.functional as F
from pampar.coder.v3.config import PRESET_V3
from pampar.coder.v3.modelo import PamparV3

# Cargar tokenizer
tok = spm.SentencePieceProcessor()
tok.Load("data/tokenizer/pampar_48k.model")

# 1. Simular _tokenize_pair
problem = "Write a Python function suma(a, b) that returns the sum of two numbers."
solution = "def suma(a, b):\n    return a + b"
text = f"### Problem:\n{problem}\n### Solution:\n```python\n{solution}\n```"
ids = tok.Encode(text)

print("=" * 60)
print("1. TOKENIZACIÓN")
print("=" * 60)
print(f"Text length: {len(text)} chars")
print(f"Token IDs count: {len(ids)}")
print(f"First 30 IDs: {ids[:30]}")
print(f"ID 0 appears: {ids.count(0)} times")
print(f"ID 0 decoded: '{tok.IdToPiece(0)}'")
print(f"ID 1 decoded: '{tok.IdToPiece(1)}'")
print(f"ID 2 decoded: '{tok.IdToPiece(2)}'")
print()

# 2. Simular _train_step
tokens = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
input_ids = tokens[:, :-1]
targets = tokens[:, 1:]

print("=" * 60)
print("2. SPLIT INPUT/TARGET")
print("=" * 60)
print(f"input_ids shape: {input_ids.shape}")
print(f"targets shape: {targets.shape}")
total = targets.numel()
t0 = (targets == 0).sum().item()
print(f"Targets == 0:    {t0}/{total} ({100 * t0 / total:.1f}%)")
print(f"Targets == -100: {(targets == -100).sum().item()}/{total}")
print()

# 3. BUG CHECK: ignore_index mismatch
print("=" * 60)
print("3. BUG: ignore_index MISMATCH")
print("=" * 60)
print(f"PamparV3.forward() usa:   ignore_index=-100")
print(f"classroom._train_step usa: ignore_index=0")
print()

# Calcular loss con ambos
model = PamparV3(PRESET_V3)
model.eval()
with torch.no_grad():
    logits, _, _ = model(input_ids)

loss_0 = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=0,
)
loss_100 = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=-100,
)
loss_pad = F.cross_entropy(
    logits.reshape(-1, logits.size(-1)),
    targets.reshape(-1),
    ignore_index=tok.pad_id() if hasattr(tok, "pad_id") else -1,
)

print(f"Loss con ignore_index=0:    {loss_0.item():.4f}")
print(f"Loss con ignore_index=-100: {loss_100.item():.4f}")
print(f"Pad token ID: {tok.pad_id() if hasattr(tok, 'pad_id') else 'N/A'}")
print()

# 4. Verificar qué tokens produce el modelo vs targets
print("=" * 60)
print("4. GENERACIÓN DEL ALUMNO")
print("=" * 60)
prompt = f"### Problem:\n{problem}\n### Solution:\n```python\n"
prompt_ids = tok.Encode(prompt)
print(f"Prompt tokens: {len(prompt_ids)}")
print(f"Prompt text: {prompt[:80]}...")

# Verificar si el modelo puede generar algo sensato
model.registrar_tokenizer(tok)
input_t = torch.tensor([prompt_ids], dtype=torch.long)
with torch.no_grad():
    logits, _, _ = model(input_t)
    # Top-5 predicciones para el último token
    probs = F.softmax(logits[0, -1], dim=-1)
    top5 = probs.topk(5)
    print("\nTop-5 siguientes tokens (modelo sin entrenar):")
    for i in range(5):
        tid = top5.indices[i].item()
        p = top5.values[i].item()
        piece = tok.IdToPiece(tid)
        print(f"  {tid:6d} ({p:.4f}): '{piece}'")

# 5. Verificar seq_len
print()
print("=" * 60)
print("5. SEQ_LEN CHECK")
print("=" * 60)
print(f"Config seq_len: 256")
print(f"Texto tokenizado tiene: {len(ids)} tokens")
print(f"Un ejemplo largo:")
long_problem = "Write a Python function that takes a list of integers and returns a dictionary where keys are the integers and values are their squares."
long_solution = """def squares_dict(numbers):
    result = {}
    for num in numbers:
        result[num] = num ** 2
    return result"""
long_text = (
    f"### Problem:\n{long_problem}\n### Solution:\n```python\n{long_solution}\n```"
)
long_ids = tok.Encode(long_text)
print(f"Texto largo: {len(long_text)} chars → {len(long_ids)} tokens")
if len(long_ids) > 256:
    print(f"⚠️  Se truncará a 256, se pierden {len(long_ids) - 256} tokens!")

# 6. Multiple train steps
print()
print("=" * 60)
print("6. TRAIN STEPS ANALYSIS")
print("=" * 60)
print(f"train_steps config: 3 (actual)")
print(f"Pero _train_step se llama UNA VEZ y acumula loss de todo el batch")
print(f"No hace 3 gradient steps separados — verifica el código")

# 7. Loss masking analysis
print()
print("=" * 60)
print("7. LOSS MASKING — ¿EL MODELO ENTRENA EN EL PROMPT?")
print("=" * 60)
prompt_text = f"### Problem:\n{problem}\n### Solution:\n```python\n"
prompt_tokens = tok.Encode(prompt_text)
solution_tokens = tok.Encode(solution)
print(f"Prompt tokens: {len(prompt_tokens)}")
print(f"Solution tokens: {len(solution_tokens)}")
print(f"Total tokens: {len(ids)}")
print(f"Ratio prompt/total: {len(prompt_tokens) / len(ids) * 100:.1f}%")
print()
print("⚠️  El loss se calcula sobre TODOS los tokens incluyendo el prompt!")
print("   El modelo gasta gradientes prediciendo '### Problem:', 'Write a Python'...")
print("   Solo debería aprender en los tokens de la SOLUCIÓN.")
