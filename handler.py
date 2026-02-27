"""
RunPod Serverless handler for SDXL inference with dual LoRA support.
Supports txt2img and img2img modes.
"""

import os
import io
import base64
import time
import torch
import runpod
from PIL import Image
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from safetensors.torch import load_file
import requests

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_ID = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ── Global pipeline (loaded once at cold start) ────────────────────────────
print(f"[init] Loading VAE from {VAE_ID}...")
vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=DTYPE)

print(f"[init] Loading SDXL pipeline from {MODEL_ID}...")
txt2img_pipe = StableDiffusionXLPipeline.from_pretrained(
    MODEL_ID,
    vae=vae,
    torch_dtype=DTYPE,
    use_safetensors=True,
    variant="fp16" if DTYPE == torch.float16 else None,
)
txt2img_pipe.to(DEVICE)
txt2img_pipe.enable_xformers_memory_efficient_attention()

img2img_pipe = StableDiffusionXLImg2ImgPipeline(
    vae=txt2img_pipe.vae,
    text_encoder=txt2img_pipe.text_encoder,
    text_encoder_2=txt2img_pipe.text_encoder_2,
    tokenizer=txt2img_pipe.tokenizer,
    tokenizer_2=txt2img_pipe.tokenizer_2,
    unet=txt2img_pipe.unet,
    scheduler=txt2img_pipe.scheduler,
)
img2img_pipe.to(DEVICE)

print("[init] Pipelines ready.")

# ── LoRA cache ──────────────────────────────────────────────────────────────
_lora_cache: dict[str, str] = {}


def download_lora(storage_path: str) -> str:
    """Download a LoRA .safetensors file from Supabase Storage and return local path."""
    if storage_path in _lora_cache:
        return _lora_cache[storage_path]

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY must be set")

    local_path = f"/tmp/lora_{abs(hash(storage_path))}.safetensors"

    print(f"[lora] Downloading {storage_path}...")
    # Use REST API directly for reliable downloads
    url = f"{SUPABASE_URL}/storage/v1/object/loras/{storage_path}"
    r = requests.get(url, headers={
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    })
    r.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(r.content)

    _lora_cache[storage_path] = local_path
    print(f"[lora] Saved to {local_path} ({len(r.content) / 1024 / 1024:.1f} MB)")
    return local_path


def apply_loras(pipe, lora_configs: list[dict]):
    """Load and fuse multiple LoRAs into the pipeline."""
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    if not lora_configs:
        return

    adapter_names = []
    adapter_weights = []

    for cfg in lora_configs:
        local_path = download_lora(cfg["path"])
        adapter_name = cfg["adapter_name"]
        pipe.load_lora_weights(local_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
        adapter_weights.append(cfg["weight"])
        print(f"[lora] Loaded adapter '{adapter_name}' weight={cfg['weight']}")

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)


def image_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Handler ─────────────────────────────────────────────────────────────────
def handler(job):
    inp = job["input"]

    prompt = inp.get("prompt", "")
    negative_prompt = inp.get("negative_prompt", "blurry, low quality, deformed")
    width = inp.get("width", 1024)
    height = inp.get("height", 1024)
    steps = inp.get("steps", 28)
    cfg = inp.get("cfg", 7.0)
    seed = inp.get("seed", -1)
    mode = inp.get("mode", "txt2img")

    # LoRA config
    lora_url = inp.get("lora_url")
    lora_weight = inp.get("lora_weight", 0.75)
    trigger_word = inp.get("trigger_word", "")
    style_lora_url = inp.get("style_lora_url")
    style_lora_weight = inp.get("style_lora_weight", 0.5)

    # img2img
    init_image_base64 = inp.get("init_image_base64")
    strength = inp.get("strength", 0.65)

    # Inject trigger word
    if trigger_word and trigger_word not in prompt:
        prompt = f"{trigger_word}, {prompt}"

    # Seed
    if seed == -1:
        seed = int(time.time()) % 2**32
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Build LoRA list
    lora_configs = []
    if lora_url:
        lora_configs.append({
            "path": lora_url,
            "weight": lora_weight,
            "adapter_name": "subject",
        })
    if style_lora_url:
        lora_configs.append({
            "path": style_lora_url,
            "weight": style_lora_weight,
            "adapter_name": "style",
        })

    # Select pipeline
    if mode == "img2img" and init_image_base64:
        pipe = img2img_pipe
        apply_loras(pipe, lora_configs)

        init_bytes = base64.b64decode(init_image_base64)
        init_image = Image.open(io.BytesIO(init_bytes)).convert("RGB").resize((width, height))

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )
    else:
        pipe = txt2img_pipe
        apply_loras(pipe, lora_configs)

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            generator=generator,
        )

    output_image = result.images[0]
    img_b64 = image_to_base64(output_image)

    return {
        "images": [f"data:image/png;base64,{img_b64}"],
        "seed": seed,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "mode": mode,
            "lora_weight": lora_weight,
            "style_lora_weight": style_lora_weight if style_lora_url else 0,
            "strength": strength if mode == "img2img" else None,
        },
    }


runpod.serverless.start({"handler": handler})

