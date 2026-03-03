# ── SDXL Inference Worker with Diffusers + Dual LoRA ────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# System deps (aligned with trainer)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Numpy first (trainer lesson)
RUN pip install --no-cache-dir "numpy==1.26.4"

# PyTorch — same as trainer
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Python deps — versions aligned with trainer where possible
RUN pip install --no-cache-dir \
    "diffusers[torch]==0.25.0" \
    "transformers==4.36.2" \
    "accelerate==0.25.0" \
    "safetensors==0.4.2" \
    "huggingface-hub==0.20.1" \
    "peft==0.10.0" \
    Pillow \
    requests \
    runpod

# Pre-download SDXL base + VAE at build time (fast cold start)
RUN python -c "\
from diffusers import StableDiffusionXLPipeline, AutoencoderKL; \
import torch; \
vae = AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=torch.float16); \
pipe = StableDiffusionXLPipeline.from_pretrained( \
    'stabilityai/stable-diffusion-xl-base-1.0', \
    vae=vae, \
    torch_dtype=torch.float16, \
    use_safetensors=True, \
    variant='fp16', \
); \
print('SDXL + VAE downloaded successfully') \
"

# Verify imports (trainer lesson)
RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
import diffusers; print(f'diffusers {diffusers.__version__}'); \
import peft; print(f'peft {peft.__version__}'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK')"

WORKDIR /app
COPY handler.py .

# python -u = unbuffered output for real-time logs
CMD ["python", "-u", "handler.py"]
