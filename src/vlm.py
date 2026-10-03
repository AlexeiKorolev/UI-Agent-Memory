"""Shared vLLM helpers: model paths, deterministic image resizing, chat batching."""
import math
import os
from pathlib import Path

from PIL import Image

# project root: $MG_PROJ if set, else the repo root (parent of src/)
PROJ = Path(os.environ.get("MG_PROJ", Path(__file__).resolve().parents[1]))
MODELS = {
    "controller": ("Qwen/Qwen3-VL-8B-Instruct", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"),
    "backbone": ("inclusionAI/UI-Venus-1.5-8B", "a06ff6c6f15a9eca210769dacc1603f73b4a500c"),
}

# Pixel budgets (we resize ourselves so every engine/processor sees identical pixels).
PX_CURRENT = 2_500_000      # current screenshot for the backbone (~2.4k visual tokens)
PX_CONTROLLER = 1_200_000   # current screenshot for the controller
PX_CROP = 300_000           # each memory crop
PX_HISTORY = 350_000        # each past screenshot in C2
FACTOR = 32                 # Qwen3-VL: 16px patches x 2x2 merge


def resize_to_budget(im, max_pixels, min_side=FACTOR):
    """Downscale (never upscale beyond rounding) to <= max_pixels, both sides multiples of 32."""
    im = im.convert("RGB")
    w, h = im.size
    scale = min(1.0, math.sqrt(max_pixels / (w * h)))
    nw = max(min_side, int(w * scale) // FACTOR * FACTOR)
    nh = max(min_side, int(h * scale) // FACTOR * FACTOR)
    if (nw, nh) != (w, h):
        im = im.resize((nw, nh), Image.BICUBIC)
    return im


def snapshot_path(role):
    from huggingface_hub import snapshot_download
    repo, rev = MODELS[role]
    return snapshot_download(repo, revision=rev, local_files_only=True)


def make_llm(role, max_model_len=24576, max_images=32, gpu_mem=0.90):
    from vllm import LLM
    return LLM(model=snapshot_path(role), dtype="bfloat16", max_model_len=max_model_len,
               limit_mm_per_prompt={"image": max_images, "video": 0}, gpu_memory_utilization=gpu_mem,
               seed=0, trust_remote_code=False, enable_prefix_caching=True,
               mm_processor_cache_gb=0)


def greedy(max_tokens):
    from vllm import SamplingParams
    return SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)


def to_messages(parts):
    """parts: list of str | PIL.Image  ->  single-user-turn chat message for vLLM."""
    content = []
    for p in parts:
        if isinstance(p, str):
            if content and content[-1]["type"] == "text":
                content[-1]["text"] += p
            else:
                content.append({"type": "text", "text": p})
        else:
            content.append({"type": "image_pil", "image_pil": p})
    return [{"role": "user", "content": content}]
