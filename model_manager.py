'''
Modified from: https://github.com/LALBJ/PAI/blob/master/model_loader.py
'''

import os
import torch
from constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    IMAGE_TOKEN_INDEX,
    IMAGE_TOKEN_LENGTH,
    INSTRUCTION_TEMPLATE,
    SYSTEM_MESSAGE,
)

# ── model path registry ───────────────────────────────────────────────────────
MODEL_PATHS = {
    "llava-1.5":     "./llava-v1.5-7b",
    "qwen2.5-vl-7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen3-vl-8b":   "Qwen/Qwen3-VL-8B-Instruct",
    "gemma-3-27b":   "google/gemma-3-27b-it",
    "gemma-3-12b":   "google/gemma-3-12b-it",
    "gemma-3-4b":    "google/gemma-3-4b-it",
    "internvl3-8b":  "OpenGVLab/InternVL3-8B-hf",
}

QWEN_MODELS = ("qwen2.5-vl-7b", "qwen3-vl-8b")
GEMMA_MODELS = ("gemma-3-27b", "gemma-3-12b", "gemma-3-4b")
INTERNVL_MODELS = ("internvl3-8b",)

# section_info schema:
#   {
#     'order': 'SIT',
#     'before_img': [('system', n_tokens), ('task', n_tokens), ...],
#     'after_img':  [('task', n_tokens), ...],
#   }
# Each entry is a (section_name, approximate_token_count) pair in prompt order.


def _count_tokens(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _parse_order(order):
    """Normalise and validate order string; ensure I is present."""
    order = "".join(c for c in order.upper() if c in "SIT")
    if "I" not in order:
        order = "I" + order
    return order


# ═══════════════════════════════════════════════════════════════════════════════
#  LLaVA-1.5
# ═══════════════════════════════════════════════════════════════════════════════

def load_llava_model(model_path):
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, False, False, device=device
    )
    return tokenizer, model, image_processor, model


def _build_llava_template(system_text, order):
    """
    Construct the LLaVA prompt template string for a given ordering.

    ``order`` letters: S=system text, I=image, T=task/question
    Template placeholders: <ImageHere>  (image),  <question>  (task text)
    """
    order = _parse_order(order)
    part = {"S": system_text, "I": "<ImageHere>", "T": "<question>"}
    inner = "\n".join(part[c] for c in order if part.get(c))
    return f"USER: {inner} ASSISTANT:"


def _llava_section_info(system_text, task_text, order, tokenizer):
    """
    Compute (approximate) per-section token counts for the LLaVA prompt.

    Returns a section_info dict.
    """
    order = _parse_order(order)
    img_pos = order.index("I")
    before_chars = [c for c in order[:img_pos]  if c in "ST"]
    after_chars  = [c for c in order[img_pos+1:] if c in "ST"]

    texts = {"S": system_text, "T": task_text}

    before_img = [
        ("system" if c == "S" else "task", _count_tokens(tokenizer, texts[c]))
        for c in before_chars if texts.get(c)
    ]
    after_img = [
        ("system" if c == "S" else "task", _count_tokens(tokenizer, texts[c]))
        for c in after_chars if texts.get(c)
    ]
    return {"order": order, "before_img": before_img, "after_img": after_img}


def prepare_llava_inputs(template, query, image_tensor, tokenizer):
    qu = [template.replace("<question>", q) for q in query]
    batch_size = len(query)

    chunks      = [q.split("<ImageHere>") for q in qu]
    chunk_before = [chunk[0] for chunk in chunks]
    chunk_after  = [chunk[1] for chunk in chunks]

    token_before = (
        tokenizer(chunk_before, return_tensors="pt", padding="longest",
                  add_special_tokens=False).to("cuda").input_ids
    )
    token_after = (
        tokenizer(chunk_after,  return_tensors="pt", padding="longest",
                  add_special_tokens=False).to("cuda").input_ids
    )
    bos = torch.ones([batch_size, 1], dtype=torch.int64, device="cuda") * tokenizer.bos_token_id
    image_token = torch.ones([batch_size, 1], dtype=torch.int64, device="cuda") * IMAGE_TOKEN_INDEX

    img_start_idx = len(token_before[0]) + 1       # +1 for BOS
    img_end_idx   = img_start_idx + IMAGE_TOKEN_LENGTH

    input_ids = torch.cat([bos, token_before, image_token, token_after], dim=1)
    kwargs = {"images": image_tensor.half()}
    return qu, input_ids, img_start_idx, img_end_idx, kwargs


# ═══════════════════════════════════════════════════════════════════════════════
#  Qwen2.5-VL / Qwen3-VL
# ═══════════════════════════════════════════════════════════════════════════════

def load_qwen_vl_model(model_path, model_name, attn_implementation=None):
    from transformers import AutoProcessor

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if model_name == "qwen2.5-vl-7b":
        from transformers import Qwen2_5_VLForConditionalGeneration as MC
    else:
        from transformers import Qwen3VLForConditionalGeneration as MC

    kw = dict(torch_dtype=torch.float16, device_map=device)
    if attn_implementation:   # e.g. "eager" so output_attentions works
        kw["attn_implementation"] = attn_implementation
    model     = MC.from_pretrained(model_path, **kw)
    processor = AutoProcessor.from_pretrained(model_path)
    return processor.tokenizer, model, processor, model


def _build_qwen_messages(system_text, task_text, pil_image, order, image_copies=1,
                         task2_text=None):
    """Build Qwen messages list with content items in the requested order.

    image_copies>1 repeats the SAME image N times at the image position — used to
    increase the number of image tokens between the question and the answer
    (distance sweep) without removing information.

    task2_text (if given) is used for the SECOND+ Task occurrence (e.g. order
    'STIT'): lets the post-image repeat be a short cue instead of the full
    question, to test whether the question *content* (not just a post-image
    prompt) drives the STIT gain.
    """
    order = _parse_order(order)
    content = []
    t_seen = 0
    for c in order:
        if c == "S" and system_text:
            content.append({"type": "text", "text": system_text})
        elif c == "T":
            t_seen += 1
            txt = task_text if (t_seen == 1 or task2_text is None) else task2_text
            content.append({"type": "text", "text": txt})
        elif c == "I":
            for _ in range(max(1, int(image_copies))):
                content.append({"type": "image", "image": pil_image})
    return [{"role": "user", "content": content}]


def _qwen_section_info(system_text, task_text, order, tokenizer):
    """Approximate per-section token counts for the Qwen prompt."""
    order = _parse_order(order)
    img_pos = order.index("I")
    before_chars = [c for c in order[:img_pos]  if c in "ST"]
    after_chars  = [c for c in order[img_pos+1:] if c in "ST"]

    texts = {"S": system_text, "T": task_text}
    before_img = [
        ("system" if c == "S" else "task", _count_tokens(tokenizer, texts[c]))
        for c in before_chars if texts.get(c)
    ]
    after_img = [
        ("system" if c == "S" else "task", _count_tokens(tokenizer, texts[c]))
        for c in after_chars if texts.get(c)
    ]
    return {"order": order, "before_img": before_img, "after_img": after_img}


def prepare_qwen_vl_inputs(model, processor, model_name, query_list, pil_image,
                            system_text="", order="IT", image_copies=1,
                            task2_text=None, enable_thinking=False):
    from qwen_vl_utils import process_vision_info

    query    = query_list[0]
    messages = _build_qwen_messages(system_text, query, pil_image, order,
                                    image_copies=image_copies, task2_text=task2_text)

    tmpl_kwargs = {}
    if model_name == "qwen3-vl-8b":
        tmpl_kwargs["enable_thinking"] = bool(enable_thinking)

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, **tmpl_kwargs
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # ── locate vision token span ──────────────────────────────────────────────
    ids            = inputs.input_ids[0]
    vision_start_id = model.config.vision_start_token_id   # 151652
    vision_end_id   = model.config.vision_end_token_id     # 151653

    img_start_idx   = (ids == vision_start_id).nonzero(as_tuple=True)[0][0].item() + 1
    img_end_idx     = (ids == vision_end_id  ).nonzero(as_tuple=True)[0][0].item()
    n_vision_tokens = img_end_idx - img_start_idx

    # ── grid shape ────────────────────────────────────────────────────────────
    merge_size = model.config.vision_config.spatial_merge_size
    if "image_grid_thw" in inputs:
        t, h, w = inputs["image_grid_thw"][0]
        h, w = h.item(), w.item()
        grid_h, grid_w = (h, w) if h * w == n_vision_tokens else (h // merge_size, w // merge_size)
    else:
        side = int(n_vision_tokens ** 0.5)
        grid_h = grid_w = side

    input_ids = inputs.input_ids
    kwargs    = {k: v for k, v in inputs.items() if k != "input_ids"}
    return [query], input_ids, img_start_idx, img_end_idx, grid_h, grid_w, kwargs


# ═══════════════════════════════════════════════════════════════════════════════
#  Gemma 3 (multimodal)
# ═══════════════════════════════════════════════════════════════════════════════

def load_gemma_model(model_path, model_name, attn_implementation=None):
    from transformers import (AutoProcessor, Gemma3ForConditionalGeneration,
                              BitsAndBytesConfig)

    # IMPORTANT: load on a SINGLE GPU (device_map={"":0}), never multi-GPU shard.
    # Splitting Gemma-3-27B across 2 GPUs with device_map="auto" on this no-NVLink
    # box produces NaN logits (garbage output) AND ~0.1 tok/s — verified by isolating
    # single-GPU (correct + ~9 tok/s) vs sharded (NaN). Set CUDA_VISIBLE_DEVICES to
    # pick which physical GPU is "0".
    #
    # 27B in bf16 (~54GB) does not fit one 48GB GPU, so quantize it to 4-bit nf4
    # (~16GB). Smaller Gemmas (12B ~24GB, 4B ~8GB) fit in bf16 → load full precision.
    big = "27b" in model_name.lower()
    kw = dict(device_map={"": 0})
    if big:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    else:
        kw["torch_dtype"] = torch.bfloat16
    if attn_implementation:           # e.g. "eager" so output_attentions works
        kw["attn_implementation"] = attn_implementation
    model     = Gemma3ForConditionalGeneration.from_pretrained(model_path, **kw)
    processor = AutoProcessor.from_pretrained(model_path)
    return processor.tokenizer, model, processor, model


def _gemma_image_token_id(model, processor):
    """Best-effort lookup of Gemma's image placeholder token id."""
    for attr in ("image_token_index", "image_token_id"):
        v = getattr(model.config, attr, None)
        if v is not None:
            return v
    tok = getattr(processor, "tokenizer", None)
    if tok is not None:
        for name in ("<image_soft_token>", "<start_of_image>"):
            tid = tok.convert_tokens_to_ids(name)
            if tid is not None and tid >= 0:
                return tid
    return None


def prepare_gemma_inputs(model, processor, model_name, query_list, pil_image,
                          system_text="", order="IT", image_copies=1,
                          task2_text=None, enable_thinking=False):
    """Gemma 3 input prep. Reuses the same ordered content layout as Qwen (system
    text lives as a text item in the user turn) so the S/I/T positional semantics
    are identical across models — keeping the IST/STI/STIT comparison apples-to-
    apples. enable_thinking is accepted for signature parity (Gemma has no such
    flag) and ignored."""
    query    = query_list[0]
    messages = _build_qwen_messages(system_text, query, pil_image, order,
                                    image_copies=image_copies, task2_text=task2_text)

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    ids = inputs["input_ids"][0]
    img_tok = _gemma_image_token_id(model, processor)
    pos = ((ids == img_tok).nonzero(as_tuple=True)[0]
           if img_tok is not None else torch.tensor([], dtype=torch.long))
    if len(pos):
        img_start_idx = int(pos[0].item())
        img_end_idx   = int(pos[-1].item()) + 1
    else:
        img_start_idx = img_end_idx = 0
    n_vis = max(1, img_end_idx - img_start_idx)
    side  = int(n_vis ** 0.5)
    grid_h = grid_w = side if side * side == n_vis else side

    input_ids = inputs["input_ids"]
    kwargs    = {k: v for k, v in inputs.items() if k != "input_ids"}
    return [query], input_ids, img_start_idx, img_end_idx, grid_h, grid_w, kwargs


# ═══════════════════════════════════════════════════════════════════════════════
#  InternVL3 (native transformers support, HF-format checkpoint)
# ═══════════════════════════════════════════════════════════════════════════════

def load_internvl_model(model_path, model_name, attn_implementation=None):
    from transformers import InternVLForConditionalGeneration, AutoProcessor
    kw = dict(torch_dtype=torch.bfloat16, device_map={"": 0})
    if attn_implementation:
        kw["attn_implementation"] = attn_implementation
    model     = InternVLForConditionalGeneration.from_pretrained(model_path, **kw)
    processor = AutoProcessor.from_pretrained(model_path)
    return processor.tokenizer, model, processor, model


def prepare_internvl_inputs(model, processor, model_name, query_list, pil_image,
                            system_text="", order="IT", image_copies=1,
                            task2_text=None, enable_thinking=False):
    """InternVL3 input prep. Reuses the shared ordered content layout
    (_build_qwen_messages) so the S/I/T positional semantics match the other
    models. enable_thinking is accepted for parity and ignored."""
    query    = query_list[0]
    messages = _build_qwen_messages(system_text, query, pil_image, order,
                                    image_copies=image_copies, task2_text=task2_text)
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    ).to(model.device)

    ids = inputs["input_ids"][0]
    img_tok = getattr(model.config, "image_token_id",
                      getattr(model.config, "image_token_index", None))
    pos = ((ids == img_tok).nonzero(as_tuple=True)[0]
           if img_tok is not None else torch.tensor([], dtype=torch.long))
    if len(pos):
        img_start_idx = int(pos[0].item())
        img_end_idx   = int(pos[-1].item()) + 1
    else:
        img_start_idx = img_end_idx = 0
    n_vis  = max(1, img_end_idx - img_start_idx)
    side   = int(n_vis ** 0.5)
    grid_h = grid_w = side

    input_ids = inputs["input_ids"]
    kwargs    = {k: v for k, v in inputs.items() if k != "input_ids"}
    return [query], input_ids, img_start_idx, img_end_idx, grid_h, grid_w, kwargs


# ═══════════════════════════════════════════════════════════════════════════════
#  ModelManager
# ═══════════════════════════════════════════════════════════════════════════════

class ModelManager:
    def __init__(self, model_name, attn_implementation=None):
        self.model_name     = model_name.lower()
        self.attn_implementation = attn_implementation
        self.tokenizer      = None
        self.vlm_model      = None
        self.llm_model      = None
        self.image_processor = None
        # set by prepare_inputs_from_pil
        self.img_start_idx  = None
        self.img_end_idx    = None
        self.grid_h         = None
        self.grid_w         = None
        self.num_layers     = None
        self.section_info   = None   # populated by prepare_inputs_from_pil
        self.load_model()

    # ── model loading ─────────────────────────────────────────────────────────

    def load_model(self):
        if self.model_name == "llava-1.5":
            model_path = os.path.expanduser(MODEL_PATHS["llava-1.5"])
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_llava_model(model_path)
            )
            self.grid_h = self.grid_w = 24
            self.num_layers = self.llm_model.config.num_hidden_layers

        elif self.model_name in QWEN_MODELS:
            model_path = MODEL_PATHS[self.model_name]
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_qwen_vl_model(model_path, self.model_name,
                                   attn_implementation=self.attn_implementation)
            )
            cfg = self.llm_model.config
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
                self.num_layers = cfg.text_config.num_hidden_layers
            else:
                self.num_layers = cfg.num_hidden_layers

        elif self.model_name in GEMMA_MODELS:
            model_path = MODEL_PATHS[self.model_name]
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_gemma_model(model_path, self.model_name,
                                 attn_implementation=self.attn_implementation)
            )
            cfg = self.llm_model.config
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
                self.num_layers = cfg.text_config.num_hidden_layers
            else:
                self.num_layers = cfg.num_hidden_layers

        elif self.model_name in INTERNVL_MODELS:
            model_path = MODEL_PATHS[self.model_name]
            self.tokenizer, self.vlm_model, self.image_processor, self.llm_model = (
                load_internvl_model(model_path, self.model_name,
                                    attn_implementation=self.attn_implementation)
            )
            cfg = self.llm_model.config
            if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "num_hidden_layers"):
                self.num_layers = cfg.text_config.num_hidden_layers
            else:
                self.num_layers = getattr(cfg, "num_hidden_layers", None)
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    # ── unified input preparation ─────────────────────────────────────────────

    def prepare_inputs_from_pil(self, query_list, pil_image,
                                  system_prompt="", order=None, image_copies=1,
                                  task2_text=None, enable_thinking=False):
        """
        Prepare model inputs from a PIL image.

        Parameters
        ----------
        query_list    : list[str]  – task queries (usually length 1)
        pil_image     : PIL.Image
        system_prompt : str        – optional system text
        order         : str        – e.g. "SIT", "IST".  Defaults to "SIT" for
                                     LLaVA (S before image) and "IT" for Qwen.

        Sets: img_start_idx, img_end_idx, grid_h, grid_w, section_info.
        Returns (questions, input_ids, kwargs).
        """
        query = query_list[0]

        if self.model_name == "llava-1.5":
            if order is None:
                order = "SIT"
            sys_text = system_prompt if system_prompt else SYSTEM_MESSAGE
            template = _build_llava_template(sys_text, order)

            from llava.mm_utils import process_images
            images_tensor = process_images(
                [pil_image], self.image_processor, self.llm_model.config,
            ).to(self.llm_model.device, dtype=torch.float16)

            questions, input_ids, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
                template, query_list, images_tensor, self.tokenizer
            )
            self.img_start_idx = img_start_idx
            self.img_end_idx   = img_end_idx
            self.grid_h = self.grid_w = 24
            self.section_info  = _llava_section_info(sys_text, query, order, self.tokenizer)
            return questions, input_ids, kwargs

        elif self.model_name in QWEN_MODELS:
            if order is None:
                order = "IT"
            sys_text = system_prompt or ""
            questions, input_ids, img_start_idx, img_end_idx, grid_h, grid_w, kwargs = (
                prepare_qwen_vl_inputs(
                    self.llm_model, self.image_processor, self.model_name,
                    query_list, pil_image,
                    system_text=sys_text, order=order, image_copies=image_copies,
                    task2_text=task2_text, enable_thinking=enable_thinking,
                )
            )
            self.img_start_idx = img_start_idx
            self.img_end_idx   = img_end_idx
            self.grid_h        = grid_h
            self.grid_w        = grid_w
            self.section_info  = _qwen_section_info(sys_text, query, order, self.tokenizer)
            return questions, input_ids, kwargs

        elif self.model_name in GEMMA_MODELS:
            if order is None:
                order = "IT"
            sys_text = system_prompt or ""
            questions, input_ids, img_start_idx, img_end_idx, grid_h, grid_w, kwargs = (
                prepare_gemma_inputs(
                    self.llm_model, self.image_processor, self.model_name,
                    query_list, pil_image,
                    system_text=sys_text, order=order, image_copies=image_copies,
                    task2_text=task2_text, enable_thinking=enable_thinking,
                )
            )
            self.img_start_idx = img_start_idx
            self.img_end_idx   = img_end_idx
            self.grid_h        = grid_h
            self.grid_w        = grid_w
            self.section_info  = _qwen_section_info(sys_text, query, order, self.tokenizer)
            return questions, input_ids, kwargs

        elif self.model_name in INTERNVL_MODELS:
            if order is None:
                order = "IT"
            sys_text = system_prompt or ""
            questions, input_ids, img_start_idx, img_end_idx, grid_h, grid_w, kwargs = (
                prepare_internvl_inputs(
                    self.llm_model, self.image_processor, self.model_name,
                    query_list, pil_image,
                    system_text=sys_text, order=order, image_copies=image_copies,
                    task2_text=task2_text, enable_thinking=enable_thinking,
                )
            )
            self.img_start_idx = img_start_idx
            self.img_end_idx   = img_end_idx
            self.grid_h        = grid_h
            self.grid_w        = grid_w
            self.section_info  = _qwen_section_info(sys_text, query, order, self.tokenizer)
            return questions, input_ids, kwargs

        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    # ── space-token augmented inputs ──────────────────────────────────────────

    def prepare_inputs_with_spaces(self, query_list, pil_image,
                                     system_prompt="", order=None, n_spaces=20):
        """
        Same as prepare_inputs_from_pil, but inserts exactly ``n_spaces`` space
        tokens at the end of the user content — i.e. right before the
        user-turn-closing token. For IST (Image·System·Task) that is *after the
        Task prompt*; for STI (System·Task·Image) that is *after the image*.

        The image span is left unchanged (insertion happens after it in both
        orderings), so vision-token logit-lens indexing is unaffected.

        Sets the same attributes as prepare_inputs_from_pil. Returns
        (questions, input_ids, kwargs).
        """
        questions, input_ids, kwargs = self.prepare_inputs_from_pil(
            query_list, pil_image, system_prompt=system_prompt, order=order,
        )
        if n_spaces <= 0:
            return questions, input_ids, kwargs

        space_id = self.tokenizer(" ", add_special_tokens=False).input_ids[-1]
        ids = input_ids[0]

        if self.model_name in QWEN_MODELS:
            im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
            im_end   = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
            starts = (ids == im_start).nonzero().flatten()
            a = starts[-1].item() if len(starts) else ids.shape[0]
            ends = (ids[:a] == im_end).nonzero().flatten()
            insert_at = ends[-1].item() if len(ends) else a
        else:
            # LLaVA: prompt ends with " ASSISTANT:"; append before generation.
            insert_at = ids.shape[0]

        sp = torch.full((n_spaces,), space_id, dtype=ids.dtype, device=ids.device)
        input_ids = torch.cat([ids[:insert_at], sp, ids[insert_at:]]).unsqueeze(0)

        kwargs = dict(kwargs)
        # Extend any per-token kwargs so get_rope_index / attention stay aligned.
        for key, fill in (("attention_mask", 1), ("mm_token_type_ids", 0)):
            if key in kwargs and kwargs[key] is not None and kwargs[key].dim() == 2:
                t = kwargs[key][0]
                pad = torch.full((n_spaces,), fill, dtype=t.dtype, device=t.device)
                kwargs[key] = torch.cat([t[:insert_at], pad, t[insert_at:]]).unsqueeze(0)

        # Make the inserted spaces visible in the text-token grid: append a
        # 'spaces' segment after the existing after-image sections (the spaces sit
        # right after Task for IST / after the image for STI, i.e. first/last in
        # after_img). Without this the renderer's segmentation drops them.
        si = dict(self.section_info)
        after = list(si.get("after_img", []))
        after.append(("spaces", n_spaces))
        si["after_img"] = after
        self.section_info = si

        return questions, input_ids, kwargs

    # ── legacy LLaVA-only helpers (notebook compatibility) ────────────────────

    def construct_template(self):
        if self.model_name == "llava-1.5":
            return SYSTEM_MESSAGE + " " + INSTRUCTION_TEMPLATE[self.model_name]
        raise ValueError(f"construct_template not supported for {self.model_name}")

    def prepare_inputs_for_model(self, query, image, use_dataloader=False):
        """LLaVA-1.5 only legacy path.  Prefer prepare_inputs_from_pil()."""
        if self.model_name != "llava-1.5":
            raise ValueError(
                f"prepare_inputs_for_model only supports llava-1.5; "
                f"use prepare_inputs_from_pil() for {self.model_name}"
            )
        template = self.construct_template()
        images_tensor = image["pixel_values"][0] if use_dataloader else image
        questions, input_ids, img_start_idx, img_end_idx, kwargs = prepare_llava_inputs(
            template, query, images_tensor, self.tokenizer
        )
        self.img_start_idx = img_start_idx
        self.img_end_idx   = img_end_idx
        return questions, input_ids, kwargs

    def decode(self, output_ids):
        if self.model_name == "llava-1.5":
            output_ids = output_ids.clone()
            output_ids[output_ids == IMAGE_TOKEN_INDEX] = torch.tensor(
                0, dtype=output_ids.dtype, device=output_ids.device
            )
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            return [t.split("ASSISTANT:")[-1].strip() for t in output_text]
        elif self.model_name in QWEN_MODELS + GEMMA_MODELS:
            return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        raise ValueError(f"Unknown model: {self.model_name}")
