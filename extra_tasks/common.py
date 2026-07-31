"""
Shared vLLM-library plumbing for the "new CV areas" ordering experiments
(VQAv2, TallyQA, MMVP, BLINK, NExT-QA), reusing the exact S/T/I ordering
construction and coordinate-free JSON-parsing conventions established in
detpo_map/ordering_eval_vllm.py, generalized to accept a *list* of images at
the "I" position so the same code serves single-image tasks and multi-frame
video (NExT-QA) without duplication.

Engine split (same rationale as detpo_map, see detpo_map/PROMPTS.md):
  STI / SIT / STIT / SITIT -> vLLM library (pure input-layout changes)
  SITIT_rev                -> local HF + reverse_image_hooks (parked here for
                               now; add later per-task if requested)
"""
import base64
import io
import os
import sys

from PIL import Image

SUPP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SUPP)
from constants import SYSTEM_MESSAGE  # noqa: E402

# Model registry: tag -> (HF repo id, vLLM quantization arg or None).
# gemma-3-27b needs bitsandbytes 4-bit on a SINGLE GPU (bf16 needs ~54GB, doesn't
# fit one card; this box has no NVLink, so tensor-parallel sharding across both
# GPUs is either pathologically slow or -- with HF's naive device_map="auto" --
# produces NaN logits, verified separately). bnb 4-bit fits comfortably on one
# GPU (~16.7GB) and was confirmed to produce coherent output via vLLM.
MODEL_REGISTRY = {
    "qwen3-vl-8b": {"hf": "Qwen/Qwen3-VL-8B-Instruct", "quantization": None},
    "gemma-3-27b": {"hf": "google/gemma-3-27b-it", "quantization": "bitsandbytes"},
}
MODEL_HF = MODEL_REGISTRY["qwen3-vl-8b"]["hf"]
MODEL_TAG = "qwen3-vl-8b"

# Hook-free orderings, run on vLLM. SITIT_rev needs detpo_map/ordering_eval.py's
# reverse_image_hooks (local HF) and is intentionally not included here yet.
ORDER_LETTERS = {"STI": "STI", "SIT": "SIT", "STIT": "STIT", "SITIT": "SITIT"}
ORDER_LIST = ["STI", "SIT", "STIT", "SITIT"]


def downscale(pil, cap):
    w, h = pil.size
    s = cap / max(w, h)
    return pil if s >= 1.0 else pil.resize((max(1, int(w * s)), max(1, int(h * s))),
                                            Image.LANCZOS)


def data_uri(pil, quality=90):
    buf = io.BytesIO()
    pil.convert("RGB").save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_conversation(letters, task_text, image_uris, system_text=SYSTEM_MESSAGE):
    """image_uris: list of data-URI strings, ALL inserted (in order) at every 'I'
    in the ordering. A single-image task passes a 1-element list; SITIT with a
    K-frame video repeats the whole K-frame block at each 'I' (image echo)."""
    if isinstance(image_uris, str):
        image_uris = [image_uris]
    parts = []
    for c in letters:
        if c == "S":
            parts.append({"type": "text", "text": system_text})
        elif c == "T":
            parts.append({"type": "text", "text": task_text})
        elif c == "I":
            for uri in image_uris:
                parts.append({"type": "image_url", "image_url": {"url": uri}})
    return [{"role": "user", "content": parts}]


def make_llm(tp=2, max_model_len=24096, gpu_mem=0.85, limit_images=2,
            disable_mm_cache=False, model_tag="qwen3-vl-8b"):
    """disable_mm_cache: vLLM's multimodal-processor LRU cache can hit an
    internal AssertionError ("Expected a cached item for mm_hash=...") under
    large batches with many distinct images per request (observed on NExT-QA:
    1500 prompts x 6 frames). It exists to avoid reprocessing the SAME image
    seen in multiple prompts, which barely helps workloads where each prompt's
    images are mostly unique (e.g. one video's frames per prompt) -- disabling
    it there trades a little redundant preprocessing for correctness.

    model_tag: key into MODEL_REGISTRY. gemma-3-27b forces tp=1 (single GPU,
    bnb 4-bit -- see MODEL_REGISTRY comment); passing tp>1 for it is a bug."""
    cfg = MODEL_REGISTRY[model_tag]
    if model_tag == "gemma-3-27b":
        assert tp == 1, "gemma-3-27b must run tp=1 (single GPU, bnb 4-bit)"
    from vllm import LLM
    kwargs = dict(model=cfg["hf"], trust_remote_code=True,
                 max_model_len=max_model_len, tensor_parallel_size=tp,
                 gpu_memory_utilization=gpu_mem,
                 limit_mm_per_prompt={"image": limit_images},
                 disable_mm_preprocessor_cache=disable_mm_cache)
    if cfg["quantization"]:
        kwargs["quantization"] = cfg["quantization"]
        kwargs["load_format"] = cfg["quantization"]
    else:
        kwargs["dtype"] = "float16"
    return LLM(**kwargs)
