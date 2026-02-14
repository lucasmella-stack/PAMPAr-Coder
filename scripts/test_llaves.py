#!/usr/bin/env python3
"""Test LLAVES v2 changes: no duplicates, proper classification, PRESET_1_5B."""
import sys
sys.path.insert(0, ".")

from pampar.coder.v2.zonas import ZONAS, Zona, Territorio, ZONA_TERRITORIO
from pampar.coder.v2.llaves import clasificar_token, normalizar

print("=" * 60)
print("TEST 1: Verificar zonas cargadas sin duplicados")
print("=" * 60)

all_tokens = {}
dupes = []
for zona, tokens in ZONAS.items():
    for t in tokens:
        if t in all_tokens:
            dupes.append((t, all_tokens[t].name, zona.name))
        else:
            all_tokens[t] = zona

print(f"  Zonas totales: {len(ZONAS)}")
print(f"  Tokens únicos: {len(all_tokens)}")
with_tokens = sum(1 for z, t in ZONAS.items() if len(t) > 0)
empty = sum(1 for z, t in ZONAS.items() if len(t) == 0)
print(f"  Zonas con tokens: {with_tokens}")
print(f"  Zonas vacías (contexto): {empty}")

if dupes:
    print(f"\n  ❌ DUPLICADOS ({len(dupes)}):")
    for t, z1, z2 in dupes[:15]:
        print(f"    '{t}' -> {z1} vs {z2}")
else:
    print("  ✅ Sin duplicados entre zonas")

print()
print("=" * 60)
print("TEST 2: Clasificación de tokens Python")
print("=" * 60)

test_cases = [
    # (token, expected_zona_prefix, description)
    ("def", "B01", "keyword def"),
    ("class", "B02", "keyword class"),
    ("import", "B03", "keyword import"),
    ("return", "B04", "keyword return"),
    ("if", "B05", "keyword if"),
    ("for", "B06", "keyword for"),
    ("try", "B07", "keyword try"),
    ("async", "B08", "keyword async"),
    ("with", "B09", "keyword with"),
    ("pass", "B10", "keyword pass"),
    ("(", "B11", "paren open"),
    ("[", "B12", "bracket open"),
    ("{", "B13", "brace open"),
    (",", "B14", "comma"),
    ("#", "B15", "comment"),
    ("self", "B16", "self"),
    ("True", "B24", "bool"),
    ("None", "B25", "none"),
    ("int", "B26", "type prim"),
    ("list", "B27", "type coll"),
    ("Optional", "B28", "type generic"),
    ("print", "B29", "builtin"),
    ("len", "B29", "builtin len"),
    ("ValueError", "B29", "exception"),
    ("__init__", "B30", "magic"),
    ("__repr__", "B30", "magic repr"),
    ("+", "B31", "arith"),
    ("==", "B32", "comparison"),
    ("and", "B33", "logic"),
    ("=", "B35", "assign"),
    (".", "B36", "member"),
    ("->", "B43", "arrow"),
    ("\n", "B48", "newline"),
    ("\t", "B47", "indent"),
    (" ", "B49", "space"),
    # Edge cases
    ("42", "B21", "integer literal"),
    ("3.14", "B22", "float literal"),
    ("MyClass", "B18", "CamelCase class"),
    ("HTTP_ERROR", "B18", "UPPER_CASE constant"),
    ("my_var", "B16", "snake_case variable"),
    ("__custom_dunder__", "B30", "custom dunder"),
]

passed = 0
failed = 0
for token, expected_prefix, desc in test_cases:
    zona, conf = clasificar_token(token)
    zona_prefix = zona.name[:3]  # e.g. "B01"
    ok = zona.name.startswith(expected_prefix)
    status = "✅" if ok else "❌"
    if ok:
        passed += 1
    else:
        failed += 1
    print(f"  {status} '{token:20s}' -> {zona.name:25s} (conf={conf:.2f}) [{desc}]" + 
          ("" if ok else f" EXPECTED {expected_prefix}"))

print(f"\n  Results: {passed}/{passed+failed} passed")

print()
print("=" * 60)
print("TEST 3: Config seq_len")
print("=" * 60)

from cloud.runpod.config_3b import Config1_5B
config = Config1_5B()
print(f"  max_seq_len: {config.max_seq_len}")
print(f"  batch_size: {config.batch_size}")
print(f"  gradient_accumulation: {config.gradient_accumulation}")
print(f"  effective_batch: {config.effective_batch}")
assert config.max_seq_len == 1024, f"Expected 1024, got {config.max_seq_len}"
assert config.batch_size * config.gradient_accumulation == config.effective_batch
print("  ✅ Config OK")

print()
print("=" * 60)
print("TEST 4: LLAVES module import + forward")
print("=" * 60)

import torch
from pampar.coder.v2.llaves import LlavesV2

llaves = LlavesV2(vocab_size=100, n_zonas=52)
# Simulate token IDs
ids = torch.tensor([[0, 1, 2, 3, 4]])  # [1, 5]
out = llaves(ids)
print(f"  Input shape: {ids.shape}")
print(f"  Output shape: {out.shape}")
assert out.shape == (1, 5, 52), f"Expected (1, 5, 52), got {out.shape}"
print("  ✅ Forward pass OK")

print()
print("=" * 60)
print("ALL TESTS COMPLETE")
print("=" * 60)
