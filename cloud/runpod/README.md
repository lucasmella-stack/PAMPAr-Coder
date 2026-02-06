# 🚀 PAMPAr-Coder: Entrenamiento en RunPod

## Guía Completa para Entrenar 3B de Manera Efectiva

---

## 📋 CHECKLIST PRE-ENTRENAMIENTO

### 1. Preparación de Datos ✅
- [ ] Datos limpios y bien formateados
- [ ] Sin duplicados
- [ ] Balance de tipos de código (Python, JS, etc.)
- [ ] Validación separada (5-10% del dataset)

### 2. Arquitectura ✅
- [ ] Configuración validada localmente
- [ ] Forward pass funciona sin errores
- [ ] Gradient checkpointing habilitado
- [ ] Mixed precision (FP16/BF16) configurado

### 3. Hiperparámetros ✅
- [ ] Learning rate con warmup
- [ ] Batch size óptimo para VRAM
- [ ] Gradient accumulation calculado
- [ ] Weight decay configurado

---

## 🎯 MEJORES PRÁCTICAS PARA 3B

### Learning Rate Schedule

```
Warmup: 2-5% de steps totales
Peak LR: 1e-4 a 3e-4
Decay: Cosine hasta 1e-5
```

### Batch Size Efectivo

```
Para 3B en A40 (48GB):
- Micro batch: 4-8
- Gradient accumulation: 8-16
- Effective batch: 32-128
```

### Checkpointing Frecuente

```
- Cada 1000 steps: checkpoint ligero
- Cada epoch: checkpoint completo
- Mantener top 3 mejores por val_loss
```

---

## ⚠️ ERRORES COMUNES Y CÓMO EVITARLOS

### 1. OOM (Out of Memory)
**Causa**: Batch size muy grande
**Solución**: 
- Reducir batch size
- Aumentar gradient accumulation
- Activar gradient checkpointing

### 2. Loss explota (NaN/Inf)
**Causa**: Learning rate muy alto
**Solución**:
- Usar warmup más largo
- Reducir LR inicial
- Gradient clipping (max_norm=1.0)

### 3. Loss no baja
**Causa**: LR muy bajo o datos mal formateados
**Solución**:
- Verificar formato de datos
- Aumentar LR gradualmente
- Revisar tokenización

### 4. Overfitting
**Causa**: Muy pocas epochs o datos repetitivos
**Solución**:
- Data augmentation
- Dropout
- Early stopping

### 5. Entrenamiento lento
**Causa**: I/O bottleneck
**Solución**:
- Datos pre-tokenizados
- DataLoader con num_workers
- Datos en SSD/NVMe

---

## 📊 MÉTRICAS A MONITOREAR

| Métrica | Valor Esperado | Acción si Mal |
|---------|----------------|---------------|
| Train Loss | Baja gradualmente | Verificar LR |
| Val Loss | Similar a train | Aumentar regularización |
| Gradient Norm | < 1.0 | Reducir LR |
| GPU Util | > 90% | Aumentar batch |
| Memory | < 95% VRAM | OK |

---

## 🔧 SETUP EN RUNPOD

### Paso 1: Crear Pod
1. Ir a RunPod.io
2. Seleccionar A40 (48GB) - Spot ($0.20/hr)
3. Template: PyTorch 2.0+
4. Disk: 50GB mínimo

### Paso 2: Conectar
```bash
ssh root@<pod-ip> -p <port>
```

### Paso 3: Setup
```bash
bash setup.sh
```

### Paso 4: Entrenar
```bash
bash train.sh
```

### Paso 5: Guardar
```bash
bash save_model.sh
```

---

## 💡 TIPS AVANZADOS

### 1. Monitoreo con wandb
```python
wandb.init(project="pampar-coder-3b")
wandb.log({"loss": loss, "lr": lr})
```

### 2. Checkpoint Recovery
```python
# Guardar estado completo
torch.save({
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'epoch': epoch,
    'step': step,
    'best_loss': best_loss,
}, 'checkpoint.pt')
```

### 3. Early Stopping
```python
patience = 3
no_improve = 0
for epoch in range(epochs):
    if val_loss < best_loss:
        best_loss = val_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            break
```

---

## 📁 ESTRUCTURA DE ARCHIVOS

```
cloud/runpod/
├── README.md          # Esta guía
├── setup.sh           # Setup del entorno
├── train.sh           # Script de entrenamiento
├── save_model.sh      # Guardar y subir modelo
├── config_3b.py       # Config para 3B
└── train_cloud.py     # Script principal
```
