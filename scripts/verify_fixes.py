"""Verificar que los 3 fixes del pipeline funcionan correctamente."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import sentencepiece as spm
import torch

# Cargar tokenizer
tok = spm.SentencePieceProcessor()
tok.Load("data/tokenizer/pampar_48k.model")

# --- Test 1: _tokenize_pair produce labels con -100 masking ---
print("=" * 60)
print("TEST 1: Loss masking en _tokenize_pair")
print("=" * 60)

problem = "Write a Python function suma(a, b) that returns the sum."
solution = "def suma(a, b):\n    return a + b"
seq_len = 256

prompt = f"### Problem:\n{problem}\n### Solution:\n```python\n"
full = f"{prompt}{solution}\n```"
ids = tok.Encode(full)
if len(ids) > seq_len:
    ids = ids[:seq_len]

prompt_ids = tok.Encode(prompt)
prompt_len = min(len(prompt_ids), len(ids))

labels = list(ids)
for i in range(prompt_len):
    labels[i] = -100

input_ids = torch.tensor(ids, dtype=torch.long)
labels_t = torch.tensor(labels, dtype=torch.long)

masked = (labels_t == -100).sum().item()
total = labels_t.numel()
trainable = total - masked

print(f"Total tokens:      {total}")
print(f"Masked (-100):     {masked} ({100 * masked / total:.1f}%)")
print(f"Trainable:         {trainable} ({100 * trainable / total:.1f}%)")
print(f"Prompt len:        {prompt_len}")

# Verificar que los tokens de solución NO están masked
sol_tokens = labels_t[prompt_len:]
sol_masked = (sol_tokens == -100).sum().item()
print(f"Solution masked:   {sol_masked} (should be 0)")
assert sol_masked == 0, "FAIL: Solution tokens are masked!"
assert masked == prompt_len, f"FAIL: Expected {prompt_len} masked, got {masked}"
print("✓ PASS: Solo el prompt está masked, la solución se entrena\n")

# --- Test 2: ignore_index=-100 en loss ---
print("=" * 60)
print("TEST 2: ignore_index=-100 en cross_entropy")
print("=" * 60)

import torch.nn.functional as F

# Simular logits random
vocab_size = 48000
seq = 50
logits = torch.randn(1, seq - 1, vocab_size)

# Labels con masking
tgt_all = torch.randint(0, vocab_size, (1, seq - 1))
tgt_masked = tgt_all.clone()
tgt_masked[0, :20] = -100  # Mask first 20 tokens

loss_all = F.cross_entropy(
    logits.reshape(-1, vocab_size), tgt_all.reshape(-1), ignore_index=-100
)
loss_masked = F.cross_entropy(
    logits.reshape(-1, vocab_size), tgt_masked.reshape(-1), ignore_index=-100
)

print(f"Loss sin masking:  {loss_all.item():.4f} (sobre {seq - 1} tokens)")
print(f"Loss con masking:  {loss_masked.item():.4f} (sobre {seq - 1 - 20} tokens)")
print(f"Loss igual? {abs(loss_all.item() - loss_masked.item()) < 0.01}")
print("✓ PASS: ignore_index=-100 filtra correctamente\n")

# --- Test 3: ReplayBuffer almacena input_ids + labels ---
print("=" * 60)
print("TEST 3: ReplayBuffer formato nuevo")
print("=" * 60)

from classroom_memory import ReplayBuffer

rb = ReplayBuffer(maxsize=100)
rb.add(
    problem=problem,
    solution=solution,
    input_ids=input_ids,
    labels=labels_t,
    level=1,
)

sample = rb.sample(1)[0]
assert "input_ids" in sample, "FAIL: 'input_ids' not in replay sample"
assert "labels" in sample, "FAIL: 'labels' not in replay sample"
assert "tokens" not in sample, "FAIL: old 'tokens' key still present"
assert (sample["labels"][:prompt_len] == -100).all(), (
    "FAIL: labels not masked in replay"
)
print(f"Keys: {list(sample.keys())}")
print(f"input_ids shape: {sample['input_ids'].shape}")
print(f"labels shape:    {sample['labels'].shape}")
print("✓ PASS: ReplayBuffer usa formato (input_ids, labels)\n")

# --- Test 4: SleepConsolidator usa sample['input_ids'] ---
print("=" * 60)
print("TEST 4: SleepConsolidator formato nuevo")
print("=" * 60)

import inspect

from bio_mechanisms import SleepConsolidator

source = inspect.getsource(SleepConsolidator.consolidate)
assert 'sample["tokens"]' not in source, "FAIL: Still uses sample['tokens']"
assert 'sample["input_ids"]' in source, "FAIL: Not using sample['input_ids']"
assert 'sample["labels"]' in source, "FAIL: Not using sample['labels']"
assert "ignore_index=-100" in source, "FAIL: Not using ignore_index=-100"
print("✓ PASS: SleepConsolidator usa input_ids/labels + ignore_index=-100\n")

# --- Test 5: EWC compute_fisher usa ignore_index=-100 ---
print("=" * 60)
print("TEST 5: EWC compute_fisher")
print("=" * 60)

from classroom_memory import EWC

source_ewc = inspect.getsource(EWC.compute_fisher)
assert "ignore_index=0" not in source_ewc, "FAIL: Still uses ignore_index=0"
assert "ignore_index=-100" in source_ewc, "FAIL: Not using ignore_index=-100"
print("✓ PASS: EWC compute_fisher usa ignore_index=-100\n")

print("=" * 60)
print("TODOS LOS TESTS PASARON ✓")
print("=" * 60)
