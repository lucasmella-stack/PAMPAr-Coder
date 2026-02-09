"""Verifica modelo 1.5B + tokenizer."""
import sys; sys.path.insert(0, '.')
from pampar.coder.v2 import crear_modelo, PRESET_1_5B
import sentencepiece as spm

# Config
c = PRESET_1_5B
print(f'Config: dim={c.dim}, heads={c.n_heads}, kv={c.n_kv_heads}, capas={c.n_capas}')
print(f'Vocab: {c.vocab_size}, seq_len={c.max_seq_len}, ffn_mult={c.ffn_mult}')
print(f'Params estimados: {c.estimate_params():,} ({c.estimate_params()/1e9:.2f}B)')
print(f'Memoria FP16: {c.memory_estimate_mb(2):.0f}MB')
print(f'Memoria INT4: {c.memory_estimate_mb(0.5):.0f}MB')

# Tokenizer
tok = spm.SentencePieceProcessor()
tok.Load('data/tokenizer/pampar_48k.model')
print(f'\nTokenizer vocab: {tok.GetPieceSize()}')
assert tok.GetPieceSize() == c.vocab_size, 'VOCAB MISMATCH!'
print('Vocab match: OK')

test = 'función para calcular índice'
pieces = tok.EncodeAsPieces(test)
print(f'Test: {test!r} -> {pieces}')

# Model
import torch
print('\nCreando modelo (CPU)...')
model = crear_modelo(c)
params = model.count_params()
total = params['total']
print(f'Params reales: {total:,} ({total/1e9:.2f}B)')
for k, v in params.items():
    if k != 'total':
        print(f'  {k}: {v:,}')

# Quick forward test
print('\nForward test...')
x = torch.randint(0, c.vocab_size, (1, 32))
logits, loss, info = model(x)
print(f'Logits shape: {logits.shape}')
print(f'OK!')
