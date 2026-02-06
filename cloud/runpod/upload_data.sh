#!/bin/bash
# =============================================================================
# PAMPAr-Coder: Subir datos a RunPod
# Ejecutar DESDE tu PC local
# =============================================================================

# Configuración - EDITAR ESTOS VALORES
POD_IP="<tu-pod-ip>"        # Ejemplo: 123.456.789.012
POD_PORT="<tu-pod-port>"    # Ejemplo: 22222
LOCAL_DATA_DIR="data"       # Carpeta local con datos

echo "=========================================="
echo "📤 Subiendo datos a RunPod"
echo "=========================================="

# Verificar que existen los datos
if [ ! -d "$LOCAL_DATA_DIR" ]; then
    echo "❌ No se encontró directorio: $LOCAL_DATA_DIR"
    exit 1
fi

# Subir tokenizer
echo "[1/2] Subiendo tokenizer..."
scp -P $POD_PORT \
    $LOCAL_DATA_DIR/tokenizer/code_tokenizer.model \
    $LOCAL_DATA_DIR/tokenizer/code_tokenizer.vocab \
    root@$POD_IP:/workspace/PAMPAr-Coder/data/tokenizer/

# Subir datos de entrenamiento
echo "[2/2] Subiendo datos de entrenamiento..."
scp -P $POD_PORT \
    $LOCAL_DATA_DIR/distillation/*.jsonl \
    root@$POD_IP:/workspace/PAMPAr-Coder/data/distillation/

echo ""
echo "=========================================="
echo "✅ Datos subidos!"
echo ""
echo "Ahora en RunPod ejecuta:"
echo "  cd /workspace/PAMPAr-Coder"
echo "  bash cloud/runpod/train.sh 3B 10"
echo "=========================================="
