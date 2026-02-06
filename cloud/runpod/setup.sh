#!/bin/bash
# =============================================================================
# PAMPAr-Coder: Setup para RunPod A40
# =============================================================================

set -e  # Exit on error

echo "=========================================="
echo "🚀 PAMPAr-Coder: Setup RunPod A40"
echo "=========================================="

# -----------------------------------------------------------------------------
# 1. Actualizar sistema
# -----------------------------------------------------------------------------
echo "[1/6] Actualizando sistema..."
apt-get update -qq
apt-get install -y -qq git wget htop nvtop tmux

# -----------------------------------------------------------------------------
# 2. Verificar GPU
# -----------------------------------------------------------------------------
echo "[2/6] Verificando GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# -----------------------------------------------------------------------------
# 3. Instalar dependencias
# -----------------------------------------------------------------------------
echo "[3/6] Instalando dependencias..."
pip install -q --upgrade pip
pip install -q \
    torch>=2.0 \
    sentencepiece \
    tqdm \
    wandb \
    huggingface_hub \
    safetensors \
    accelerate \
    bitsandbytes

# -----------------------------------------------------------------------------
# 4. Clonar repositorio
# -----------------------------------------------------------------------------
echo "[4/6] Clonando repositorio..."
cd /workspace

if [ ! -d "PAMPAr-Coder" ]; then
    git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
fi
cd PAMPAr-Coder

# -----------------------------------------------------------------------------
# 5. Descargar datos
# -----------------------------------------------------------------------------
echo "[5/6] Preparando datos..."
mkdir -p data/distillation
mkdir -p data/tokenizer
mkdir -p checkpoints

# Si no hay datos, descargar
if [ ! -f "data/tokenizer/code_tokenizer.model" ]; then
    echo "⚠️  Necesitas subir el tokenizer manualmente o desde HuggingFace"
fi

# -----------------------------------------------------------------------------
# 6. Verificar setup
# -----------------------------------------------------------------------------
echo "[6/6] Verificando setup..."
python -c "
import torch
import sys
sys.path.insert(0, '.')
from pampar.coder.model_v2 import PampaRCoderV2, ConfigPampaRCoderV2

# Test con config 3B
config = ConfigPampaRCoderV2(
    vocab_size=16000,
    dim=2048,
    n_heads=16,
    n_capas=24,
    max_seq_len=2048,
)
model = PampaRCoderV2(config)
params = model.count_parameters()
print(f'✅ Modelo 3B creado: {params[\"total\"]/1e9:.2f}B parámetros')
print(f'💾 Memoria estimada: {params[\"total\"] * 2 / 1024**3:.1f} GB (FP16)')
"

echo ""
echo "=========================================="
echo "✅ Setup completo!"
echo ""
echo "Próximos pasos:"
echo "  1. Subir datos: scp data/* root@<pod>:/workspace/PAMPAr-Coder/data/"
echo "  2. Entrenar: bash train.sh"
echo "=========================================="
