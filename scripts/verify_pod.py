# Quick verification script for pod setup
import torch
from pampar.coder import PRESET_1_5B, crear_modelo

print(f"GPU: {torch.cuda.get_device_name()}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = crear_modelo(PRESET_1_5B)
params = sum(p.numel() for p in model.parameters())
print(f"Model: {params / 1e9:.2f}B params")

# Quick forward pass test on GPU
model = model.cuda()
x = torch.randint(0, 48000, (2, 128), device='cuda')
with torch.amp.autocast('cuda'):
    logits, loss, info = model(x, x)
print(f"Forward OK - loss: {loss.item():.4f}")
print(f"VRAM used: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
print("ALL OK")
