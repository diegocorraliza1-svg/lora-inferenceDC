FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod \
    diffusers[torch]==0.27.2 \
    transformers==4.40.2 \
    accelerate==0.29.3 \
    safetensors==0.4.3 \
    peft==0.10.0 \
    Pillow \
    requests \
    omegaconf

# Pre-download VAE
RUN python -c "from diffusers import AutoencoderKL; AutoencoderKL.from_pretrained('madebyollin/sdxl-vae-fp16-fix', torch_dtype=__import__('torch').float16)"

# Pre-download SDXL base (~6.5GB)
RUN mkdir -p /models/sdxl && \
    wget -q --show-progress -O /models/sdxl/sd_xl_base_1.0.safetensors \
    "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors"

COPY handler.py /app/handler.py
WORKDIR /app

CMD ["python", "-u", "handler.py"]
