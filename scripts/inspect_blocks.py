#!/usr/bin/env python3
"""Inspect block structure."""
import sys
sys.path.insert(0, '/workspace/PAMPAr-Coder')
from pampar.coder.v2 import crear_modelo, PRESET_1_5B

m = crear_modelo(PRESET_1_5B)

# Show block structure
bloque = m.bloques[0]
print("=== BLOQUE[0] ===")
for n, mod in bloque.named_children():
    print(f"  {n}: {type(mod).__name__}")
    for n2, mod2 in mod.named_children():
        cname = type(mod2).__name__
        print(f"    {n2}: {cname}")

print(f"\n=== TOTAL BLOQUES: {len(m.bloques)} ===")

# Check if territories are in FFN
print("\n=== FFN TERRITORIES? ===")
for n, mod in bloque.named_modules():
    low = n.lower()
    if any(w in low for w in ['terr', 'territory', 'mix', 'expert']):
        print(f"  {n}: {type(mod).__name__}")

# Model forward signature
print("\n=== FORWARD ===")
import inspect
src = inspect.getsource(type(m).forward)
for line in src.split('\n')[:30]:
    print(line)
