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

MODEL_HF = "Qwen/Qwen3-VL-8B-Instruct"
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


def make_llm(tp=2, max_model_len=24096, gpu_mem=0.85, limit_images=2):
    from vllm import LLM
    return LLM(model=MODEL_HF, trust_remote_code=True, dtype="float16",
              max_model_len=max_model_len, tensor_parallel_size=tp,
              gpu_memory_utilization=gpu_mem,
              limit_mm_per_prompt={"image": limit_images})
