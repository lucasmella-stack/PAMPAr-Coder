#!/bin/bash
# =============================================================================
# PAMPAr-Coder: Script de Entrenamiento
# =============================================================================

set -e

cd /workspace/PAMPAr-Coder

echo "=========================================="
echo "🧠 PAMPAr-Coder: Entrenamiento 3B"
echo "=========================================="

# Verificar GPU
nvidia-smi --query-gpu=name,memory.free --format=csv

# Configuración
CONFIG="${1:-3B}"          # 3B o 1.5B
EPOCHS="${2:-10}"
MAX_SAMPLES="${3:-}"       # Vacío = todos

echo ""
echo "Configuración:"
echo "  Config: $CONFIG"
echo "  Epochs: $EPOCHS"
echo "  Max samples: ${MAX_SAMPLES:-todos}"
echo ""

# Preparar comando
CMD="python cloud/runpod/train_cloud.py \
    --config $CONFIG \
    --data-dir data/distillation \
    --tokenizer data/tokenizer/code_tokenizer.model \
    --output checkpoints \
    --epochs $EPOCHS"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

# Verificar si hay checkpoint para resumir
if [ -f "checkpoints/best_model.pt" ]; then
    echo "🔄 Encontrado checkpoint previo, resumiendo..."
    CMD="$CMD --resume checkpoints/best_model.pt"
fi

# Ejecutar en tmux para persistencia
echo "Ejecutando en tmux (session: training)..."
echo "  Para ver: tmux attach -t training"
echo "  Para salir sin detener: Ctrl+B, luego D"
echo ""

tmux new-session -d -s training "$CMD"
tmux attach -t training
