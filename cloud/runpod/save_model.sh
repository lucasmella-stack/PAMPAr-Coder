#!/bin/bash
# =============================================================================
# PAMPAr-Coder: Guardar y Subir Modelo
# =============================================================================

set -e

cd /workspace/PAMPAr-Coder

echo "=========================================="
echo "💾 PAMPAr-Coder: Guardar Modelo"
echo "=========================================="

# Verificar que existe el modelo
if [ ! -f "checkpoints/best_model.pt" ]; then
    echo "❌ No se encontró checkpoints/best_model.pt"
    exit 1
fi

# -----------------------------------------------------------------------------
# 1. Convertir a formato HuggingFace (opcional)
# -----------------------------------------------------------------------------
echo "[1/3] Preparando modelo..."

python -c "
import torch
from pathlib import Path

# Cargar checkpoint
ckpt = torch.load('checkpoints/best_model.pt', map_location='cpu', weights_only=False)
model_state = ckpt['model']
config = ckpt['config']

# Guardar solo el modelo (sin optimizer)
torch.save({
    'model': model_state,
    'config': config,
}, 'checkpoints/pampar_coder_3b.pt')

print(f'✅ Modelo guardado: pampar_coder_3b.pt')
print(f'   Config: {config}')
"

# -----------------------------------------------------------------------------
# 2. Comprimir
# -----------------------------------------------------------------------------
echo "[2/3] Comprimiendo..."

cd checkpoints
tar -czvf pampar_coder_3b.tar.gz pampar_coder_3b.pt
ls -lh pampar_coder_3b.tar.gz
cd ..

# -----------------------------------------------------------------------------
# 3. Subir a HuggingFace (opcional)
# -----------------------------------------------------------------------------
echo "[3/3] Opciones de descarga:"
echo ""
echo "  Opción A: Descargar vía SCP"
echo "    scp -P <port> root@<pod-ip>:/workspace/PAMPAr-Coder/checkpoints/pampar_coder_3b.tar.gz ."
echo ""
echo "  Opción B: Subir a HuggingFace"
echo "    pip install huggingface_hub"
echo "    huggingface-cli login"
echo "    python -c \"from huggingface_hub import upload_file; upload_file('checkpoints/pampar_coder_3b.tar.gz', repo_id='tu-usuario/pampar-coder-3b', path_in_repo='pampar_coder_3b.tar.gz')\""
echo ""
echo "  Opción C: Google Drive"
echo "    pip install gdown"
echo "    # Subir manualmente o usar rclone"
echo ""

# Si HF_TOKEN está configurado, subir automáticamente
if [ -n "$HF_TOKEN" ]; then
    echo "🚀 HF_TOKEN detectado, subiendo a HuggingFace..."
    python -c "
from huggingface_hub import HfApi, upload_file
import os

api = HfApi()
repo_id = os.environ.get('HF_REPO', 'lucasmella-stack/pampar-coder-3b')

# Crear repo si no existe
try:
    api.create_repo(repo_id, exist_ok=True)
except:
    pass

# Subir modelo
upload_file(
    path_or_fileobj='checkpoints/pampar_coder_3b.tar.gz',
    path_in_repo='pampar_coder_3b.tar.gz',
    repo_id=repo_id,
    token=os.environ['HF_TOKEN'],
)
print(f'✅ Modelo subido a: https://huggingface.co/{repo_id}')
"
fi

echo ""
echo "=========================================="
echo "✅ Modelo listo para descargar!"
echo "=========================================="
