# Cloud Training Instructions

> Entrenamiento de PAMPAr-Coder en RunPod y otros providers.

## RunPod Setup

### Conectar
```bash
ssh root@IP -p PORT
# Password: en RunPod dashboard o usar SSH key
```

### Preparar entorno
```bash
cd /workspace/PAMPAr-Coder
pip install sentencepiece tqdm datasets
```

### Lanzar entrenamiento
```bash
# Background con log
nohup python3 cloud/runpod/train_cloud.py \
    --config 3B \
    --data-dir data/distillation \
    --tokenizer data/tokenizer/code_bpe.model \
    --epochs 10 \
    --no-wandb \
    > training.log 2>&1 &

# Monitorear
tail -f training.log
nvidia-smi -l 5  # GPU cada 5 segundos
```

## Configuraciones

| Config | Params | VRAM | GPU recomendada |
|--------|--------|------|-----------------|
| 1.5B | ~230M | 8GB | RTX 3090, A10 |
| 3B | ~3B | 24GB | A40, A100 |

### Ajustar config
```python
# cloud/runpod/config_3b.py
@dataclass
class Config3B:
    vocab_size: int = 32000
    dim: int = 2560
    n_heads: int = 20
    n_capas: int = 32
    max_seq_len: int = 2048
    batch_size: int = 4
    gradient_accumulation: int = 16
```

## Troubleshooting

### OOM en GPU
1. Reducir `batch_size`
2. Reducir `max_seq_len`
3. Activar `use_gradient_checkpointing = True`

### OOM en RAM (sistema)
1. Usar streaming dataset
2. Reducir workers de DataLoader
3. Modelo se carga en CPU antes de GPU - reducir tamaño

### Tokens fuera de rango
- Asegurar `vocab_size` en config == tokenizer.GetPieceSize()
- Típico: tokenizer tiene 32K, config dice 16K → error

## Checkpoints

```bash
# Ubicación
/workspace/PAMPAr-Coder/checkpoints/
├── best_model.pt      # Mejor val_loss
├── epoch_N.pt         # Por epoch
└── step_XXXX.pt       # Por steps

# Descargar a local
scp -P PORT root@IP:/workspace/PAMPAr-Coder/checkpoints/best_model.pt ./
```

## Costos estimados

| GPU | $/hora | 10 epochs (20K samples) |
|-----|--------|------------------------|
| A10 | $0.30 | ~$0.60 |
| A40 | $0.40 | ~$0.80 |
| A100 | $1.50 | ~$3.00 |
