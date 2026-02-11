import sys

with open('scripts/train_1_5B.py', 'r') as f:
    content = f.read()

content = content.replace('"max_seq_len": 4096', '"max_seq_len": 1024')
content = content.replace('"batch_size": 4,', '"batch_size": 1,')
content = content.replace('"grad_accum": 8,', '"grad_accum": 32,')

with open('scripts/train_1_5B.py', 'w') as f:
    f.write(content)

print("Config updated: batch_size=1, grad_accum=32, max_seq_len=1024")
