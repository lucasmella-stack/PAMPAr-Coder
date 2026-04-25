# PAMPAr-Coder — Phase 6 ablation training image (Alibaba GPU ready)
#
# Base: PyTorch oficial con CUDA 12.1, runtime + cuDNN.
# Single-stage: la base ya es minimal (~6GB con CUDA), splitear en stages
# y copiar site-packages es frágil porque depende de la versión exacta
# de Python del base image (3.11.x → si cambia, los COPY rompen).

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Non-root user
RUN groupadd -r pampar && useradd -r -g pampar -m -d /home/pampar pampar

# Deps Python (cachea la layer si requirements.txt no cambia)
WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copia el repo (data/clean_sft.jsonl y tokenizer van adentro)
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
