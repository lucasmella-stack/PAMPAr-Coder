#!/usr/bin/env python3
"""Inspect the actual architecture being trained."""
import sys
sys.path.insert(0, '/workspace/PAMPAr-Coder')

from pampar.coder.v2 import crear_modelo, PRESET_1_5B

m = crear_modelo(PRESET_1_5B)

print("=== MODELO ===")
print(f"Clase: {type(m).__name__}")
print()

print("=== COMPONENTES TOP-LEVEL ===")
for name, mod in m.named_children():
    print(f"  {name}: {type(mod).__name__}")
print()

print("=== CONFIG CEREBRAL ===")
c = PRESET_1_5B
attrs = ['n_territorios', 'n_zonas', 'peso_llaves', 'use_checkpoint',
         'dim', 'n_capas', 'vocab_size', 'ffn_mult', 'umbral_exit', 'capas_min']
for k in attrs:
    v = getattr(c, k, 'N/A')
    print(f"  {k}: {v}")
print()

# Check if brain modules exist
print("=== MODULOS CEREBRALES ===")
has_talamo = any('talamo' in n.lower() for n, _ in m.named_modules())
has_territorio = any('territorio' in n.lower() for n, _ in m.named_modules())
has_brodmann = any('brodmann' in n.lower() for n, _ in m.named_modules())
has_llaves = any('llaves' in n.lower() or 'llave' in n.lower() for n, _ in m.named_modules())
has_frontera = any('frontera' in n.lower() for n, _ in m.named_modules())

print(f"  Tálamo: {'SI' if has_talamo else 'NO'}")
print(f"  Territorios: {'SI' if has_territorio else 'NO'}")
print(f"  Brodmann zones: {'SI' if has_brodmann else 'NO'}")
print(f"  LLAVES system: {'SI' if has_llaves else 'NO'}")
print(f"  Fronteras: {'SI' if has_frontera else 'NO'}")
print()

# Count brain-specific modules
brain_count = 0
for n, _ in m.named_modules():
    low = n.lower()
    if any(w in low for w in ['talamo', 'territorio', 'brodmann', 'llave', 'frontera', 'zona']):
        brain_count += 1
        if brain_count <= 20:
            print(f"  BRAIN: {n}")

print(f"\n  Total brain modules: {brain_count}")
