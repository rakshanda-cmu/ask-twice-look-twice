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
# gemma-3-27b and gemma-4-31b both need bitsandbytes 4-bit on a SINGLE GPU
# (bf16 needs ~54-62GB, doesn't fit one card; this box has no NVLink, so
# tensor-parallel sharding across both GPUs is either pathologically slow or
# -- with HF's naive device_map="auto" -- produces NaN logits, verified
# separately for gemma-3-27b). bnb 4-bit fits comfortably on one GPU
# (~16-17GB) and was confirmed to produce coherent output via vLLM.
#
# gemma-4-31b's engine is "local_hf", not "vllm": vLLM 0.19.1 (the newest
# version whose PyPI wheel is still CUDA-12.x-compatible on this box's driver,
# see this repo's git log) crashes loading Gemma4ForConditionalGeneration --
# a real shape bug in vLLM's reimplementation of Gemma 4's heterogeneous
# attention head dims (head_dim=256 local / 512 global), reproduced in both
# compiled and eager mode, and matching multiple open upstream vLLM issues
# about this exact architecture. Local HF transformers' OWN Gemma4 modeling
# code is a separate implementation and does not hit this bug (verified: loads
# and generates correctly). Slower (no continuous batching), used only because
# vLLM is not viable for this model yet.
MODEL_REGISTRY = {
    "qwen3-vl-8b": {"hf": "Qwen/Qwen3-VL-8B-Instruct", "quantization": None,
                    "engine": "vllm"},
    "gemma-3-27b": {"hf": "google/gemma-3-27b-it", "quantization": "bitsandbytes",
                    "engine": "vllm"},
    "gemma-4-31b": {"hf": "google/gemma-4-31B-it", "quantization": "bitsandbytes",
                    "engine": "local_hf"},
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


class _HFOutItem:
    """Mimics one vLLM CompletionOutput -- only the .text attribute callers use."""
    def __init__(self, text):
        self.text = text


class _HFOut:
    """Mimics one vLLM RequestOutput -- only the .outputs[0].text access path
    every run_order()/run_dataset() in this codebase actually uses."""
    def __init__(self, text):
        self.outputs = [_HFOutItem(text)]


def _uri_to_pil(uri):
    header, b64 = uri.split(",", 1)
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


class HFChatEngine:
    """Drop-in replacement for vllm.LLM exposing just the .chat(convs, sp,
    use_tqdm=False) surface every harness in this repo calls, backed by local
    HF transformers .generate() instead of vLLM's batched engine. Used only
    for gemma-4-31b (see MODEL_REGISTRY's "engine" comment for why vLLM isn't
    viable for it). No continuous batching -- one generate() call per item --
    so this is much slower than vLLM; that's a known, accepted tradeoff for
    the one model that needs it, not something to "optimize" silently, since
    correctness under padding/batching for a brand-new architecture is a real
    risk this repo has been burned by before (see the coord_scale and
    max_tokens bugs found earlier for gemma-3-27b).

    convs: the SAME vLLM chat-format list this repo's build_conversation()
    already produces ([{"role": "user", "content": [{"type": "text", ...} |
    {"type": "image_url", "image_url": {"url": data_uri}}, ...]}]) -- converted
    here to HF's {"type": "image", "image": PIL.Image} message format."""

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def _to_hf_messages(self, conv):
        out = []
        for msg in conv:
            content = []
            for part in msg["content"]:
                if part["type"] == "text":
                    content.append({"type": "text", "text": part["text"]})
                elif part["type"] == "image_url":
                    content.append({"type": "image",
                                    "image": _uri_to_pil(part["image_url"]["url"])})
            out.append({"role": msg["role"], "content": content})
        return out

    def chat(self, convs, sp, use_tqdm=False, log_every=25):
        # log_every: this repo's run_order()/run_dataset() functions all call
        # llm.chat() ONCE with the full item list, then print their own
        # progress in a loop that only starts once chat() *returns* -- fine
        # for vLLM's batched engine (fast enough that the wait is short), but
        # this shim generates one item at a time, so without printing here a
        # multi-hour benchmark stage would show zero output until it's fully
        # done. Confirmed via a live run: BLINK (793 items) ran 90+ minutes
        # completely silent before this was added.
        import time
        import torch
        results = []
        t0 = time.time()
        for i, conv in enumerate(convs):
            messages = self._to_hf_messages(conv)
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(
                    **inputs, max_new_tokens=sp.max_tokens,
                    do_sample=sp.temperature > 0,
                    temperature=sp.temperature if sp.temperature > 0 else None)
            text = self.processor.tokenizer.decode(
                out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            results.append(_HFOut(text))
            if (i + 1) % log_every == 0 or (i + 1) == len(convs):
                dt = time.time() - t0
                rate = (i + 1) / dt if dt > 0 else 0.0
                print(f"    [local_hf] {i + 1}/{len(convs)} "
                      f"({rate:.2f} items/s, {dt:.0f}s elapsed)", flush=True)
        return results


def load_hf_chat_engine(model_tag):
    """Local-HF equivalent of make_llm() for models with engine=="local_hf"
    (currently only gemma-4-31b). Single GPU, bnb 4-bit -- same rationale as
    MODEL_REGISTRY's gemma-3-27b/gemma-4-31b comment; set CUDA_VISIBLE_DEVICES
    to pick the physical GPU."""
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig
    cfg = MODEL_REGISTRY[model_tag]
    assert cfg["engine"] == "local_hf", f"{model_tag} should use make_llm(), not this"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    if model_tag == "gemma-4-31b":
        from transformers import Gemma4ForConditionalGeneration as ModelCls
    else:
        raise ValueError(f"no local_hf loader wired up for {model_tag}")
    model = ModelCls.from_pretrained(cfg["hf"], device_map={"": 0},
                                     quantization_config=bnb)
    processor = AutoProcessor.from_pretrained(cfg["hf"])
    return HFChatEngine(model, processor)


def make_llm(tp=2, max_model_len=24096, gpu_mem=0.85, limit_images=2,
            disable_mm_cache=False, model_tag="qwen3-vl-8b"):
    """disable_mm_cache: vLLM's multimodal-processor LRU cache can hit an
    internal AssertionError ("Expected a cached item for mm_hash=...") under
    large batches with many distinct images per request (observed on NExT-QA:
    1500 prompts x 6 frames). It exists to avoid reprocessing the SAME image
    seen in multiple prompts, which barely helps workloads where each prompt's
    images are mostly unique (e.g. one video's frames per prompt) -- disabling
    it there trades a little redundant preprocessing for correctness.

    model_tag: key into MODEL_REGISTRY. gemma-3-27b/gemma-4-31b force tp=1
    (single GPU, bnb 4-bit -- see MODEL_REGISTRY comment); passing tp>1 for
    either is a bug. gemma-4-31b returns an HFChatEngine instead of a vllm.LLM
    (see MODEL_REGISTRY's "engine" comment) -- same .chat() call surface, so
    callers don't need to know which one they got."""
    cfg = MODEL_REGISTRY[model_tag]
    if model_tag in ("gemma-3-27b", "gemma-4-31b"):
        assert tp == 1, f"{model_tag} must run tp=1 (single GPU, bnb 4-bit)"
    if cfg["engine"] == "local_hf":
        return load_hf_chat_engine(model_tag)
    from vllm import LLM
    kwargs = dict(model=cfg["hf"], trust_remote_code=True,
                 max_model_len=max_model_len, tensor_parallel_size=tp,
                 gpu_memory_utilization=gpu_mem,
                 limit_mm_per_prompt={"image": limit_images})
    # disable_mm_preprocessor_cache was removed in vLLM 0.19; mm_processor_cache_gb=0
    # is its replacement (0 == cache disabled).
    if disable_mm_cache:
        kwargs["mm_processor_cache_gb"] = 0
    if cfg["quantization"]:
        kwargs["quantization"] = cfg["quantization"]
        kwargs["load_format"] = cfg["quantization"]
    else:
        kwargs["dtype"] = "float16"
    return LLM(**kwargs)
