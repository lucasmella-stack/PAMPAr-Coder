#!/bin/bash
# Script para descargar datos masivos y lanzar entrenamiento 3B
set -e

cd /workspace/PAMPAr-Coder

echo "============================================"
echo "  ESTADO DEL POD"
echo "============================================"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
echo "Disco: $(df -h /workspace | tail -1 | awk '{print $4}') libre"
echo "RAM: $(free -h | awk '/Mem:/{print $3 "/" $2}')"
echo "Datos actuales:"
wc -l data/distillation/*.jsonl 2>/dev/null || echo "  Sin datos"
echo ""

echo "============================================"
echo "  1/3: DESCARGANDO DATOS MASIVOS"
echo "============================================"
pip install datasets -q 2>/dev/null

python3 << 'PYEOF'
from datasets import load_dataset
import json, os
from tqdm import tqdm

OUT = "data/distillation"
os.makedirs(OUT, exist_ok=True)

# 1. GitHub Python code (1M samples)
print(">>> Descargando codeparrot/github-code Python (1M)...")
try:
    ds = load_dataset("codeparrot/github-code", streaming=True, split="train", languages=["Python"])
    count = 0
    with open(f"{OUT}/github_python.jsonl", "w") as f:
        for item in tqdm(ds, total=1000000, desc="GitHub Python"):
            code = item.get("code", "")
            if 100 < len(code) < 3000:
                f.write(json.dumps({"instruction": "Complete:", "output": code[:2500]}) + "\n")
                count += 1
            if count >= 1000000:
                break
    print(f"  OK: {count:,} samples")
except Exception as e:
    print(f"  SKIP: {e}")

# 2. Python code instructions
print(">>> Descargando iamtarun/python_code_instructions_18k_alpaca...")
try:
    ds = load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train")
    count = 0
    with open(f"{OUT}/python_instructions.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps({"instruction": item.get("instruction","")[:800], "output": item.get("output","")[:2500]}) + "\n")
            count += 1
    print(f"  OK: {count:,} samples")
except Exception as e:
    print(f"  SKIP: {e}")

# 3. TokenBender code-instruct
print(">>> Descargando TokenBender/code_instructions_122k_alpaca_style...")
try:
    ds = load_dataset("TokenBender/code_instructions_122k_alpaca_style", split="train")
    count = 0
    with open(f"{OUT}/code_122k.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps({"instruction": item.get("instruction","")[:800], "output": item.get("output","")[:2500]}) + "\n")
            count += 1
    print(f"  OK: {count:,} samples")
except Exception as e:
    print(f"  SKIP: {e}")

# 4. nickrosh/Evol-Instruct-Code (70K)
print(">>> Descargando nickrosh/Evol-Instruct-Code-80k-v1...")
try:
    ds = load_dataset("nickrosh/Evol-Instruct-Code-80k-v1", split="train")
    count = 0
    with open(f"{OUT}/evol_code_80k.jsonl", "w") as f:
        for item in ds:
            f.write(json.dumps({"instruction": item.get("instruction","")[:800], "output": item.get("output","")[:2500]}) + "\n")
            count += 1
    print(f"  OK: {count:,} samples")
except Exception as e:
    print(f"  SKIP: {e}")

# Resumen
print("\n=== RESUMEN DATOS ===")
total = 0
for fname in os.listdir(OUT):
    if fname.endswith(".jsonl"):
        path = f"{OUT}/{fname}"
        lines = sum(1 for _ in open(path))
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {fname}: {lines:,} samples ({size_mb:.1f} MB)")
        total += lines
print(f"  TOTAL: {total:,} samples")
print(f"  Tokens estimados: ~{total * 400 / 1e9:.1f}B")
PYEOF

echo ""
echo "============================================"
echo "  2/3: CONFIGURANDO MODELO 3B"
echo "============================================"

# Calcular vocab_size real
VOCAB=$(python3 -c "import sentencepiece as spm; sp=spm.SentencePieceProcessor(); sp.Load('data/tokenizer/code_bpe.model'); print(sp.GetPieceSize())")
echo "Vocab size del tokenizer: $VOCAB"

# Crear config optimizada para A40
cat > cloud/runpod/config_3b.py << CONFIGEOF
from dataclasses import dataclass

@dataclass
class Config3B:
    """3B real - optimizado para A40 46GB."""
    vocab_size: int = $VOCAB
    dim: int = 2048
    n_heads: int = 16
    n_capas: int = 24
    dropout: float = 0.1
    max_seq_len: int = 512
    n_zonas: int = 52
    n_territorios: int = 4
    peso_llaves: float = 0.80
    usar_cuantizacion: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    batch_size: int = 2
    gradient_accumulation: int = 32
    effective_batch: int = 64
    learning_rate: float = 2e-4
    weight_decay: float = 0.1
    warmup_steps: int = 1000
    max_steps: int = 100000
    save_every_steps: int = 2000
    eval_every_steps: int = 1000
    keep_checkpoints: int = 3

@dataclass
class Config1_5B:
    """Modelo pequeño para pruebas."""
    vocab_size: int = $VOCAB
    dim: int = 512
    n_heads: int = 8
    n_capas: int = 12
    dropout: float = 0.1
    max_seq_len: int = 512
    n_zonas: int = 52
    n_territorios: int = 4
    peso_llaves: float = 0.80
    usar_cuantizacion: bool = True
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    batch_size: int = 4
    gradient_accumulation: int = 16
    effective_batch: int = 64
    learning_rate: float = 5e-4
    weight_decay: float = 0.1
    warmup_steps: int = 200
    max_steps: int = 30000
    save_every_steps: int = 500
    eval_every_steps: int = 250
    keep_checkpoints: int = 3

CONFIGS = {
    "3B": Config3B(),
    "1.5B": Config1_5B(),
}
CONFIGEOF

echo "Config creada!"

# Test rápido del modelo
echo ""
echo "Probando instanciación del modelo..."
python3 -c "
from cloud.runpod.config_3b import CONFIGS
from pampar.coder.model_v2 import PampaRCoderV2, ConfigPampaRCoderV2
config = CONFIGS['3B']
model_config = ConfigPampaRCoderV2(
    vocab_size=config.vocab_size, dim=config.dim, n_heads=config.n_heads,
    n_capas=config.n_capas, max_seq_len=config.max_seq_len,
)
model = PampaRCoderV2(model_config)
params = model.count_parameters()
print(f'Modelo 3B: {params[\"total\"]/1e9:.2f}B parametros')
" 2>&1

echo ""
echo "============================================"
echo "  3/3: LANZANDO ENTRENAMIENTO"
echo "============================================"

rm -f training_3b.log
nohup python3 cloud/runpod/train_cloud.py \
    --config 3B \
    --data-dir data/distillation \
    --tokenizer data/tokenizer/code_bpe.model \
    --epochs 30 \
    --no-wandb \
    > training_3b.log 2>&1 &

echo "PID: $!"
sleep 10
echo ""
echo "=== PRIMERAS LINEAS DEL LOG ==="
tail -30 training_3b.log
echo ""
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader
echo ""
echo "LISTO! Monitorear con: tail -f training_3b.log"
