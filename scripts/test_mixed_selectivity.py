"""Test de compilación y conteo de params para Mixed Selectivity."""

import sys

sys.path.insert(0, ".")

from pampar.coder.v3.config import PRESET_V3, ConfigV3

# Mixed Selectivity (nueva)
cfg_ms = PRESET_V3
params_ms = cfg_ms.estimate_params()
print("=== Mixed Selectivity ===")
for k, v in params_ms.items():
    print(f"  {k}: {v:>12,}")

# Legacy (4 FFN separados)
cfg_leg = ConfigV3(use_mixed_selectivity=False)
params_leg = cfg_leg.estimate_params()
print("\n=== Legacy (4 FFN) ===")
for k, v in params_leg.items():
    print(f"  {k}: {v:>12,}")

diff = params_leg["total"] - params_ms["total"]
pct = diff / params_leg["total"] * 100
print(f"\nAHORRO: {diff:,} params ({pct:.1f}%)")
total_leg = params_leg["total"] / 1e6
total_ms = params_ms["total"] / 1e6
print(f"Legacy: {total_leg:.1f}M -> Mixed: {total_ms:.1f}M")

# Test de instanciación del modelo completo
print("\n=== Instanciando modelo con Mixed Selectivity... ===")
import torch
from pampar.coder.v3.modelo import PamparV3

model = PamparV3(cfg_ms)
real_params = sum(p.numel() for p in model.parameters())
print(f"Parámetros reales: {real_params:,} ({real_params / 1e6:.1f}M)")

# Test forward pass
print("\n=== Forward pass... ===")
input_ids = torch.randint(0, 48000, (1, 32))
with torch.no_grad():
    logits, loss, info = model(input_ids)
print(f"logits shape: {logits.shape}")
print(f"exit_nivel: {info['exit_nivel']}")
print(f"terr_acts shape: {info['terr_acts'].shape}")

# Verificar que el modelo tiene ffn_shared y modulators
nivel0 = model.niveles[0]
has_shared = hasattr(nivel0, "ffn_shared")
has_mods = hasattr(nivel0, "modulators")
has_legacy = hasattr(nivel0, "ffns")
print(f"\nffn_shared: {has_shared}")
print(f"modulators: {has_mods} (count: {len(nivel0.modulators) if has_mods else 0})")
print(f"ffns (legacy): {has_legacy}")

print("\n=== TODO OK ===")
