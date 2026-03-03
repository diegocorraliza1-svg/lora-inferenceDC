# ── SDXL Inference Worker with Diffusers + Dual LoRA + Face Restore ─────────

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git wget libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Numpy first
RUN pip install --no-cache-dir "numpy==1.26.4"

# PyTorch
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Python deps
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

# GFPGAN for face restore
RUN pip install --no-cache-dir gfpgan basicsr facexlib

# Pre-download SDXL base + VAE
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

# Pre-download GFPGAN weights (v1.4)
RUN python -c "\
import os, requests; \
os.makedirs('/app/weights', exist_ok=True); \
url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'; \
r = requests.get(url, allow_redirects=True); \
open('/app/weights/GFPGANv1.4.pth', 'wb').write(r.content); \
print(f'GFPGAN weights downloaded: {len(r.content)} bytes') \
"

# Verify imports
RUN python -c "\
import numpy; print(f'numpy {numpy.__version__}'); \
import torch; print(f'torch {torch.__version__}'); \
import diffusers; print(f'diffusers {diffusers.__version__}'); \
import peft; print(f'peft {peft.__version__}'); \
import gfpgan; print(f'gfpgan OK'); \
import runpod; print(f'runpod {runpod.__version__}'); \
print('ALL IMPORTS OK')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]
