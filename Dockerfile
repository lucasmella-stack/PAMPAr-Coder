# PAMPAr-Coder — Phase 6 ablation training image (Alibaba GPU ready)
#
# Base: PyTorch oficial con CUDA 12.1, runtime + cuDNN.
# Multi-stage para minimizar tamaño final.

# ── Stage 1: builder (instala deps) ────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS builder

WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime ───────────────────────────────────────────────────
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Non-root user
RUN groupadd -r pampar && useradd -r -g pampar -m -d /home/pampar pampar

# Copia paquetes Python ya instalados desde builder
COPY --from=builder /opt/conda/lib/python3.11/site-packages /opt/conda/lib/python3.11/site-packages
COPY --from=builder /opt/conda/bin /opt/conda/bin

WORKDIR /workspace
COPY --chown=pampar:pampar . /workspace

# El dataset y tokenizer se copian con el repo. Los outputs van a un
# volumen montado externamente (ej. /workspace/ablation_results).
RUN mkdir -p /workspace/ablation_results && chown -R pampar:pampar /workspace

USER pampar

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace \
    HF_HUB_DISABLE_TELEMETRY=1

# Healthcheck: model importable + CUDA ok
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import torch, pampar.coder.v4; assert torch.cuda.is_available()" || exit 1

# Default: lanza A_baseline. Override con `docker run ... <args>`.
ENTRYPOINT ["python", "scripts/train_v4_ablation.py"]
CMD ["--config", "configs/phase6_ablation/A_baseline.yaml", "--seed", "42"]
