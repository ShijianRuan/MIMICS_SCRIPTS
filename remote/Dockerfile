# One Linux CUDA runtime for both optional remote training backends.
# Model weights are intentionally mounted at /models and are not baked in.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MIMICS_AI_APP_ROOT=/app \
    MIMICS_REMOTE_MODELS_ROOT=/models \
    HF_HOME=/tmp/mimics-ai-cache/huggingface \
    TORCH_HOME=/tmp/mimics-ai-cache/torch \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    WANDB_MODE=offline \
    MIMICS_REMOTE_LOG_MAX_BYTES=33554432 \
    MIMICS_REMOTE_LOG_BACKUP_COUNT=3 \
    PYTHONPATH=/app:/app/tools:/app/integrations/nninteractive-finetune/src:/app/integrations/nnunet_segmentation_workflow

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY remote/requirements.txt /tmp/remote-requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/remote-requirements.txt

COPY . /app
RUN python -m pip install --no-deps -e /app/integrations/nninteractive-finetune

CMD ["python", "/app/tools/remote_worker.py", "preflight", "--models-dir", "/models"]
