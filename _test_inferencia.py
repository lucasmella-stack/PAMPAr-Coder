"""Test de inferencia end-to-end del modelo PamparV3."""
import torch
from pampar.coder.v3 import PamparV3, PRESET_V3
from pampar.coder.v3.config import ConfigV3

# ── Instanciar ────────────────────────────────────────────────────────────────
model = PamparV3(PRESET_V3)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
estimado = PRESET_V3.estimate_params()["total"]
mem = PRESET_V3.memory_estimate_mb()

print(f"Parámetros totales:    {n_params:,}")
print(f"Estimación config:     {estimado:,}")
print(f"VRAM modelo fp16:      {mem['model_fp16_mb']} MB")
print(f"VRAM training total:   {mem['training_total_mb']} MB")
print()

# ── Forward pass ──────────────────────────────────────────────────────────────
torch.manual_seed(42)
ids = torch.randint(0, PRESET_V3.vocab_size, (1, 16))
with torch.no_grad():
    logits, loss, info = model(ids)

print(f"Input shape:           {ids.shape}")
print(f"Logits shape:          {logits.shape}")
print(f"Loss (sin targets):    {loss}")
print(f"Info keys:             {list(info.keys())}")
print()

# ── Forward con loss ─────────────────────────────────────────────────────────
targets = ids.clone()
with torch.no_grad():
    _, loss_val, _ = model(ids, targets=targets)
print(f"Loss con targets:      {loss_val.item():.4f}")
print()

# ── Generación ───────────────────────────────────────────────────────────────
gen_ids = model.generate(ids, max_tokens=10, temperature=0.8, top_k=50)
print(f"Tokens generados:      {gen_ids.shape[1] - ids.shape[1]}")
print()

# ── Early exit info ──────────────────────────────────────────────────────────
ids2 = torch.randint(0, PRESET_V3.vocab_size, (1, 32))
with torch.no_grad():
    _, _, info2 = model(ids2)
exit_rate = info2.get("early_exit_rate", "N/A")
print(f"Early exit rate (seq32): {exit_rate}")
print()

print("=" * 50)
print("✓  PamparV3 — inferencia OK")
print("=" * 50)
