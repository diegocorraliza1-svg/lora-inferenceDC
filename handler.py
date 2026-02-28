"""
RunPod Serverless handler for SDXL inference with dual LoRA support.
Supports txt2img and img2img modes.
Images are uploaded to Supabase Storage instead of returned as base64.
"""

import os
import io
import base64
import time
import torch
import runpod
from PIL import Image
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLImg2ImgPipeline,
    AutoencoderKL,
)
import requests

# ── Config ──────────────────────────────────────────────────────────────────
MODEL_ID = os.getenv("BASE_MODEL_ID", "stabilityai/stable-diffusion-xl-base-1.0")
VAE_ID = os.getenv("VAE_ID", "madebyollin/sdxl-vae-fp16-fix")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

LORA_CACHE_DIR = "/tmp/loras"
os.makedirs(LORA_CACHE_DIR, exist_ok=True)

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

# SDPA is default in diffusers 0.27+; xformers is optional
try:
    txt2img_pipe.enable_xformers_memory_efficient_attention()
    print("[init] xformers enabled")
except Exception:
    print("[init] xformers not available, using SDPA (default)")

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


# ── Supabase helpers ────────────────────────────────────────────────────────
_lora_cache: dict[str, str] = {}


def _supabase_headers() -> dict:
    return {
        "apikey": SUPABASE_SERVICE_ROLE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    }


def download_lora(storage_path: str) -> str:
    """Download a LoRA .safetensors from Supabase public bucket, with local cache."""
    if storage_path in _lora_cache:
        local = _lora_cache[storage_path]
        if os.path.exists(local):
            print(f"[lora] Cache hit: {storage_path}")
            return local

    # Public bucket → no auth needed
    url = f"{SUPABASE_URL}/storage/v1/object/public/loras/{storage_path}"
    print(f"[lora] Downloading {url}...")
    r = requests.get(url, timeout=300)
    r.raise_for_status()

    local_path = os.path.join(LORA_CACHE_DIR, storage_path.replace("/", "_"))
    with open(local_path, "wb") as f:
        f.write(r.content)

    _lora_cache[storage_path] = local_path
    print(f"[lora] Saved {local_path} ({len(r.content) / 1024 / 1024:.1f} MB)")
    return local_path


def upload_to_supabase(
    data: bytes,
    storage_path: str,
    bucket: str = "generations",
    content_type: str = "image/png",
    max_retries: int = 3,
) -> str:
    """Upload bytes to Supabase Storage with retry + upsert."""
    url = f"{SUPABASE_URL}/storage/v1/object/{bucket}/{storage_path}"
    headers = {
        **_supabase_headers(),
        "Content-Type": content_type,
        "x-upsert": "true",
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, headers=headers, data=data, timeout=120)
            if r.ok:
                print(f"[upload] OK → {bucket}/{storage_path}")
                return storage_path
            print(f"[upload] Attempt {attempt} failed {r.status_code}: {r.text}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"[upload] Attempt {attempt} error: {e}")
            if attempt < max_retries:
                time.sleep(2 ** attempt)

    raise RuntimeError(f"Failed to upload {storage_path} after {max_retries} attempts")


# ── LoRA management ─────────────────────────────────────────────────────────
def apply_loras(pipe, lora_configs: list[dict]):
    """Load and set multiple LoRAs into the pipeline."""
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
        print(f"[lora] Loaded '{adapter_name}' weight={cfg['weight']}")

    pipe.set_adapters(adapter_names, adapter_weights=adapter_weights)


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
    num_images = inp.get("num_images", 1)
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

    # Context for upload paths
    project_id = inp.get("project_id", "unknown")
    user_id = inp.get("user_id", "unknown")
    job_id = job.get("id", f"job_{int(time.time())}")

    # Inject trigger word
    if trigger_word and trigger_word not in prompt:
        prompt = f"{trigger_word}, {prompt}"

    # Seed
    if seed < 0:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    # Build LoRA list
    lora_configs = []
    if lora_url:
        lora_configs.append({"path": lora_url, "weight": lora_weight, "adapter_name": "subject"})
    if style_lora_url:
        lora_configs.append({"path": style_lora_url, "weight": style_lora_weight, "adapter_name": "style"})

    # Generate
    common_args = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=cfg,
        generator=generator,
        num_images_per_prompt=num_images,
    )

    if mode == "img2img" and init_image_base64:
        pipe = img2img_pipe
        apply_loras(pipe, lora_configs)
        init_bytes = base64.b64decode(init_image_base64)
        init_image = Image.open(io.BytesIO(init_bytes)).convert("RGB").resize((width, height))
        result = pipe(image=init_image, strength=strength, **common_args)
    else:
        pipe = txt2img_pipe
        apply_loras(pipe, lora_configs)
        result = pipe(width=width, height=height, **common_args)

    # Upload to Storage instead of returning base64
    output_paths = []
    for i, img in enumerate(result.images):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        storage_path = f"{user_id}/{project_id}/{job_id}_{i}.png"
        upload_to_supabase(buf.getvalue(), storage_path)
        output_paths.append(storage_path)

    # Clean up LoRAs from VRAM
    try:
        pipe.unload_lora_weights()
    except Exception:
        pass

    return {
        "status": "completed",
        "images": output_paths,
        "seed": seed,
        "metadata": {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": steps,
            "cfg": cfg,
            "mode": mode,
            "width": width,
            "height": height,
            "lora_weight": lora_weight if lora_url else 0,
            "style_lora_weight": style_lora_weight if style_lora_url else 0,
            "strength": strength if mode == "img2img" else None,
        },
    }


runpod.serverless.start({"handler": handler})
