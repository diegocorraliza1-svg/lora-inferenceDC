FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    diffusers[torch] \
    transformers \
    accelerate \
    safetensors \
    xformers \
    Pillow \
    requests

# Pre-download SDXL base model and VAE at build time (faster cold starts)
RUN python -c "\
from diffusers import StableDiffusionXLPipeline, AutoencoderKL; \
AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix'); \
StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', use_safetensors=True, variant='fp16')"

WORKDIR /app
COPY handler.py .

CMD ["python", "-u", "handler.py"]

