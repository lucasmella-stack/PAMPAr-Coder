"""Quick verification of PRESET_1_5B config and param counts."""
import sys
sys.path.insert(0, '.')

from pampar.coder import PRESET_1_5B, PRESET_4GB, crear_modelo

# Show 1.5B config
print('=== PRESET_1_5B ===')
print(f'  dim:         {PRESET_1_5B.dim}')
print(f'  n_heads:     {PRESET_1_5B.n_heads} Q, {PRESET_1_5B.kv_heads} KV (GQA)')
print(f'  n_capas:     {PRESET_1_5B.n_capas}')
print(f'  vocab_size:  {PRESET_1_5B.vocab_size}')
print(f'  max_seq_len: {PRESET_1_5B.max_seq_len}')
print(f'  ffn_mult:    {PRESET_1_5B.ffn_mult}')
print(f'  head_dim:    {PRESET_1_5B.head_dim}')

# Estimated params
est = PRESET_1_5B.estimate_params()
print(f'\n  Estimated params:  {est:,} ({est/1e9:.2f}B)')
print(f'  FP16 memory:       {PRESET_1_5B.memory_estimate_mb(2):.0f} MB')
print(f'  INT4 memory:       {PRESET_1_5B.memory_estimate_mb(0.5):.0f} MB')

# Create PRESET_4GB model for sanity check
print('\n=== PRESET_4GB (sanity check) ===')
m = crear_modelo(PRESET_4GB)
params = m.count_params()
for k, v in params.items():
    print(f'  {k}: {v:,}')
print(f'  ({params["total"]/1e6:.1f}M params)')

# Now instantiate 1.5B (just config — don't allocate weights, too much RAM)
print('\n=== MANUAL 1.5B PARAM CALCULATION ===')
c = PRESET_1_5B
kv_dim = c.kv_heads * c.head_dim
ffn_hidden = int(c.dim * c.ffn_mult * 2 / 3)

emb = c.vocab_size * c.dim
print(f'  Embeddings (weight-tied):  {emb:,}')

# Per layer
q_proj = c.dim * (c.n_heads * c.head_dim)
k_proj = c.dim * kv_dim
v_proj = c.dim * kv_dim
o_proj = c.dim * c.dim
attn_total = q_proj + k_proj + v_proj + o_proj
print(f'  Attention per layer:       {attn_total:,}')

# 4 FFNs per territory
ffn_per = 3 * c.dim * ffn_hidden  # gate + up + down
ffn_total = ffn_per * c.n_territorios
print(f'  FFN per layer (x4 terr):   {ffn_total:,}')

# Mix layer
mix = c.dim * c.n_territorios * c.dim
print(f'  Mix per layer:             {mix:,}')

# Norms (2 RMSNorm per layer = 2 * dim)
norms = 2 * c.dim
print(f'  Norms per layer:           {norms:,}')

# Exit head
exit_h = c.dim
print(f'  Exit head per layer:       {exit_h:,}')

# Talamo (once, not per layer)
talamo_attn = c.dim * (c.dim // 2) + (c.dim // 2) * c.n_zonas
talamo_gate = c.n_territorios * c.n_territorios
print(f'  Talamo:                    {talamo_attn + talamo_gate:,}')

per_layer = attn_total + ffn_total + mix + norms + exit_h
total = emb + per_layer * c.n_capas + talamo_attn + talamo_gate + c.dim  # final norm

print(f'\n  Per layer total:           {per_layer:,}')
print(f'  Total model params:        {total:,} ({total/1e9:.3f}B)')
print(f'  Target: ~1.54B')
