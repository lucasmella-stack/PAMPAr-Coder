#!/bin/bash
# Setup + training completo para PAMPAr-Coder 1.5B
set -e

echo "=== SETUP STARTING ==="
cd /workspace

# 1. Clone repo
if [ ! -d "PAMPAr-Coder" ]; then
    git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
fi
cd PAMPAr-Coder

# 2. Install deps
pip install -q sentencepiece tqdm bitsandbytes datasets filelock 2>/dev/null

# 3. Download data
if [ ! -f "data/distillation/distillation_data.jsonl" ] || [ $(wc -l < "data/distillation/distillation_data.jsonl") -lt 100 ]; then
    python scripts/download_distillation_data.py 2>&1 | tail -5
fi

# 4. Verify tokenizer
if [ ! -f "data/tokenizer/pampar_48k.model" ]; then
    echo "ERROR: Tokenizer not found! Upload pampar_48k.model first"
    exit 1
fi

# 5. Verify bitsandbytes
python3 -c "import bitsandbytes; print('bitsandbytes OK')" 2>/dev/null || echo "WARNING: bitsandbytes not available"

echo "=== SETUP COMPLETE ==="
echo "=== STARTING TRAINING ==="

# 6. Start training in screen
apt-get install -y -qq screen 2>/dev/null
screen -dmS train bash -c "cd /workspace/PAMPAr-Coder && python scripts/train_24gb.py > training.log 2>&1"
echo "Training started in screen session 'train'"
echo "Monitor: screen -r train"
echo "Log: tail -f /workspace/PAMPAr-Coder/training.log"
