# PAMPAr V4 — Despliegue en Alibaba Cloud GPU

Guía mínima para lanzar la ablación de Fase 6 en una instancia GPU
(ECS gn7i / gn7e / cualquiera con CUDA ≥ 12.1 y NVIDIA Container Toolkit).

## 1. Preparar la instancia

```bash
# En la instancia Alibaba (Ubuntu 22.04 con drivers NVIDIA preinstalados)
sudo apt-get update
sudo apt-get install -y docker.io git
sudo usermod -aG docker $USER
# logout / login

# NVIDIA Container Toolkit (si no viene con la imagen)
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verificar
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## 2. Build de la imagen

```bash
git clone https://github.com/lucasmella-stack/PAMPAr-Coder.git
cd PAMPAr-Coder
docker build -t pampar-v4:phase6 .
```

## 3. Lanzar una corrida

```bash
# Una sola variante / seed
docker run --rm --gpus all \
    -v "$(pwd)/ablation_results:/workspace/ablation_results" \
    pampar-v4:phase6 \
    --config configs/phase6_ablation/B_full.yaml --seed 42

# Las 5 variantes × 3 seeds (≈ 15 corridas, secuencial en una GPU)
docker run --rm --gpus all \
    -v "$(pwd)/ablation_results:/workspace/ablation_results" \
    --entrypoint bash \
    pampar-v4:phase6 \
    scripts/launch_phase6_ablation.sh
```

## 4. Recuperar resultados

```bash
# Los resultados quedan en ./ablation_results/phase6/ del host
# Generar el reporte:
docker run --rm \
    -v "$(pwd)/ablation_results:/workspace/ablation_results" \
    --entrypoint python \
    pampar-v4:phase6 \
    scripts/analyze_phase6_ablation.py

# Descargar a tu maquina local
scp -r alibaba_user@<ip>:~/PAMPAr-Coder/ablation_results/phase6 ./
```

## 5. Costos (referencia)

A 256-dim, 3 niveles, batch=8, seq=256, 2000 steps:

- GPU A10/A100: ≈ 5-10 min por corrida
- 15 corridas: ≈ 1.5-2.5 horas total
- VRAM: ≈ 4 GB (mucho margen para escalar a dim=512+ después)

## 6. Configuración recomendada para producción

Después de validar Fase 6 con el setup pequeño, subir:

```yaml
model:
  dim: 512
  n_levels: 5
  max_seq_len: 1024
training:
  max_steps: 10000
  batch_size: 16
  seq_len: 512
```

y usar A100 80GB (`ecs.gn7e-c12g1.6xlarge` o similar).
