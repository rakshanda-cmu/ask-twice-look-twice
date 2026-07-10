"""
Logit Lens overlay visualization for vision tokens in LLaVA.

For each layer, projects vision-token AND text-token hidden states through the
LM head to show what vocabulary item each token position "represents" at that
depth.  Outputs an animated GIF (and optionally individual PNGs).

Usage:
    python logit_lens_overlay.py
    python logit_lens_overlay.py --img-path ./COCO/val2014/COCO_val2014_000000499775.jpg
    python logit_lens_overlay.py --img-path <path> --query "Describe the image."
                                 --output-gif my_output.gif --save-frames
"""

import argparse
import os

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from model_manager import ModelManager, QWEN_MODELS
from utils import setup_seeds, disable_torch_init, logitLens_of_vision_tokens

# ── constants ────────────────────────────────────────────────────────────────
# LLaVA-1.5 default (24×24 = 576).  Qwen models use dynamic grids; the actual
# grid_h / grid_w are passed explicitly to every rendering function.
GRID_SIZE   = 24
PATCH_TOTAL = GRID_SIZE * GRID_SIZE


def parse_args():
    parser = argparse.ArgumentParser(description="Logit Lens overlay on image patches.")
    parser.add_argument("--model",     type=str,   default="llava-1.5")
    parser.add_argument("--data-path", type=str,   default="./COCO/val2014/")
    parser.add_argument("--img-path",  type=str,   default=None,
                        help="Direct path to image (overrides --data-path + --image-id).")
    parser.add_argument("--image-id",  type=int,   default=499775,
                        help="COCO val2014 image id.")
    parser.add_argument("--query",     type=str,
                        default="Please help me describe the image in detail.")
    parser.add_argument("--beam",      type=int,   default=1)
    parser.add_argument("--max-tokens",type=int,   default=512)
    parser.add_argument("--top-k",     type=int,   default=50)
    parser.add_argument("--layer-range", type=int, nargs="+",
                        default=[0, 5, 7, 10, 12, 15, 17, 18, 19, 20, 21, 22, 24, 26, 27, 28, 30, 31])
    parser.add_argument("--output-gif",  type=str, default="logit_lens_overlay.gif")
    parser.add_argument("--fps",         type=float, default=1.5)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frames-dir",  type=str,  default="logit_lens_frames")
    parser.add_argument("--alpha",       type=float, default=0.55,
                        help="Heatmap blend alpha (0 = image only, 1 = heatmap only).")
    parser.add_argument("--resolution",  type=int,  default=1152,
                        help="Square output frame resolution in pixels.")
    parser.add_argument("--system-prompt", type=str, default="",
                        help="System prompt text (empty = model default).")
    parser.add_argument("--order",       type=str,  default=None,
                        help="Prompt section order: e.g. SIT, IST, IT.  "
                             "Default: SIT for LLaVA, IT for Qwen.")
    parser.add_argument("--no-text-lens", action="store_true",
                        help="Skip text-token logit lens panel.")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
#  Vision-token logit lens
# ═══════════════════════════════════════════════════════════════════════════════

def logit_lens_all_vision_tokens(model, tokenizer, input_ids, outputs,
                                  vision_token_start, layer_range,
                                  logits_warper, logits_processor,
                                  grid_h=GRID_SIZE, grid_w=GRID_SIZE):
    """
    Returns
    -------
    all_probs : np.ndarray  shape (len(layer_range), grid_h*grid_w)
        Index 0  → layer_range[0]  (forward order — bug-fixed vs original).
    all_words : list[list[str]]  shape (len(layer_range), grid_h*grid_w)
    grid_h, grid_w : int  (passed through unchanged for convenience)
    """
    patch_total = grid_h * grid_w
    token_range = [vision_token_start, vision_token_start + patch_total]
    layer_max_prob, layer_words = logitLens_of_vision_tokens(
        model, tokenizer, input_ids, outputs,
        token_range, layer_range,
        logits_warper, logits_processor,
    )
    # logitLens_of_vision_tokens prepends rows → layer_max_prob[0] = LAST layer.
    # Reverse so index i → layer_range[i], matching layer_words order.
    probs_np = layer_max_prob.numpy()[::-1].copy()
    return probs_np, layer_words


# ═══════════════════════════════════════════════════════════════════════════════
#  Text-token logit lens
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_lm_head(model, input_ids, hidden_2d, logits_warper, logits_processor):
    """
    Run the LM head on hidden_2d (n_tokens, hidden_dim) and return
    (probs np.ndarray (n_tokens,), words list[str]).
    hidden_2d must already be on the correct device/dtype.
    """
    logits = model.lm_head(hidden_2d).cpu().float()
    logits = F.log_softmax(logits, dim=-1)
    logits = logits_processor(input_ids, logits)
    logits = logits_warper(input_ids, logits)
    probs  = F.softmax(logits, dim=-1)
    vals, ids = probs.max(dim=-1)
    words  = [model.config.vocab_size and
              "□" or "□"]   # placeholder — overwritten below
    tokenizer = None        # resolved in compute_text_logit_lens
    return vals.detach().numpy(), ids.detach()


def compute_text_logit_lens(
        model, tokenizer, input_ids, outputs,
        vision_token_start, vision_token_end,
        layer_range, logits_warper, logits_processor,
):
    """
    Compute logit lens for input text tokens (before/after image) and generated tokens.

    Returns
    -------
    dict with keys 'before', 'after', 'generated', each:
        probs  : np.ndarray  (len(layer_range), n_tokens)  — index 0 = layer_range[0]
        words  : list[list[str]]  (len(layer_range) × n_tokens)
        labels : list[str]  actual decoded tokens (x-axis labels)
    """
    def _tok_label(tid):
        s = tokenizer.decode([int(tid)], skip_special_tokens=False)
        return (s.replace("\n", "↵").replace("\t", "→").strip() or "□")[:10]

    def _run_lm(hidden_2d):
        """hidden_2d: (n, hidden_dim) on GPU, model dtype → (probs np, words list)."""
        h = hidden_2d.clone().detach()
        logits = model.lm_head(h).cpu().float()
        logits = F.log_softmax(logits, dim=-1)
        logits = logits_processor(input_ids, logits)
        logits = logits_warper(input_ids, logits)
        probs  = F.softmax(logits, dim=-1)
        vals, ids = probs.max(dim=-1)
        words  = [tokenizer.decode([i.item()], skip_special_tokens=True) or "□" for i in ids]
        return vals.detach().numpy(), words

    result    = {}
    n_layers  = len(layer_range)
    lm_dtype  = next(model.lm_head.parameters()).dtype
    lm_device = next(model.lm_head.parameters()).device

    # ── input tokens before image ─────────────────────────────────────────────
    n_before_total = vision_token_start
    before_start   = 0
    before_ids     = input_ids[0, before_start:n_before_total].cpu()
    before_labels  = [_tok_label(t) for t in before_ids]
    n_before       = len(before_ids)

    before_probs = np.zeros((n_layers, n_before))
    before_words = []
    for li, layer_i in enumerate(layer_range):
        hs = outputs['hidden_states'][0][layer_i + 1].squeeze(0)   # (seq_len, hidden)
        sl = hs[before_start:n_before_total].to(device=lm_device, dtype=lm_dtype)
        p, w = _run_lm(sl)
        before_probs[li] = p
        before_words.append(w)
    result['before'] = {'probs': before_probs, 'words': before_words, 'labels': before_labels}

    # ── input tokens after image ──────────────────────────────────────────────
    # For LLaVA a single -200 placeholder expands to 576 hidden-state tokens,
    # so hs is longer than input_ids.  For Qwen there is no expansion (1-to-1).
    # Derive the correct after-image start in input_ids from the hs length so
    # the token labels and hidden-state slices always correspond 1-to-1.
    #
    # For Qwen, vision_token_end points to the vision_end special token (151653)
    # itself — the real text tokens begin one position later.  Detect this by
    # checking whether hs length == input_ids length (no expansion → Qwen).
    hs0_len         = outputs['hidden_states'][0][layer_range[0] + 1].shape[1]
    ids_len         = input_ids.shape[1]
    # +1 to skip the vision_end special token for Qwen (no-expansion models)
    hs_after_offset = vision_token_end + (1 if hs0_len == ids_len else 0)
    n_after_in_hs   = hs0_len - hs_after_offset          # tokens available after vision block
    after_ids_start = ids_len - n_after_in_hs             # corresponding position in input_ids

    n_after      = n_after_in_hs
    after_ids    = input_ids[0, after_ids_start:after_ids_start + n_after].cpu()
    after_labels = [_tok_label(t) for t in after_ids]

    after_probs = np.zeros((n_layers, n_after))
    after_words = []
    for li, layer_i in enumerate(layer_range):
        hs = outputs['hidden_states'][0][layer_i + 1].squeeze(0)
        sl = hs[hs_after_offset:hs_after_offset + n_after].to(device=lm_device, dtype=lm_dtype)
        p, w = _run_lm(sl)
        after_probs[li] = p
        after_words.append(w)
    result['after'] = {'probs': after_probs, 'words': after_words, 'labels': after_labels}

    # ── generated tokens ──────────────────────────────────────────────────────
    # hidden_states[t] at step t:
    #   t=0 → processes all input tokens, last position predicts generated token 0
    #   t≥1 → processes single token (generated token t-1), predicts generated token t
    # We show "logit lens when generating token t":
    #   t=0: hidden_states[0][layer+1][:, -1, :]   (last input position)
    #   t≥1: hidden_states[t][layer+1][:, 0, :]    (the new-token position)
    original_input_len = input_ids.shape[1]
    gen_ids     = outputs['sequences'][0, original_input_len:].cpu()
    n_steps     = len(outputs['hidden_states'])   # includes step 0
    n_gen       = min(len(gen_ids), n_steps)
    gen_labels  = [_tok_label(t) for t in gen_ids[:n_gen]]

    gen_probs = np.zeros((n_layers, n_gen))
    gen_words = []
    for li, layer_i in enumerate(layer_range):
        step_p, step_w = [], []
        for t in range(n_gen):
            if t == 0:
                hs = outputs['hidden_states'][0][layer_i + 1].squeeze(0)  # (seq, hidden)
                sl = hs[-1:].to(device=lm_device, dtype=lm_dtype)         # last input pos
            else:
                hs = outputs['hidden_states'][t][layer_i + 1].squeeze(0)  # (1, hidden) or (hidden,)
                if hs.dim() == 1:
                    hs = hs.unsqueeze(0)
                sl = hs[:1].to(device=lm_device, dtype=lm_dtype)
            p, w = _run_lm(sl)
            step_p.append(float(p[0]))
            step_w.append(w[0])
        gen_probs[li] = np.array(step_p)
        gen_words.append(step_w)
    result['generated'] = {'probs': gen_probs, 'words': gen_words, 'labels': gen_labels}

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Frame rendering helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _colormap_heatmap(prob_grid, cmap_name="plasma"):
    """(H, W) float32 in [0,1]  →  (H, W, 3) uint8."""
    rgba = plt.get_cmap(cmap_name)(prob_grid)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


# Font candidates in priority order. Noto Sans CJK covers Chinese/Japanese/Korean
# (the usual source of "tofu" □ boxes from a multilingual model's vocab) plus
# Latin/Cyrillic/Greek; DejaVu is the Latin-only fallback.
_FONT_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", None),
]


def _load_font(size):
    for path, idx in _FONT_CANDIDATES:
        try:
            if idx is None:
                return ImageFont.truetype(path, size)
            return ImageFont.truetype(path, size, index=idx)
        except Exception:
            continue
    return ImageFont.load_default()


def render_vision_frame(image: Image.Image, probs_1d: np.ndarray,
                         words_1d: list, layer_idx: int,
                         alpha: float = 0.55,
                         grid_h: int = GRID_SIZE,
                         grid_w: int = GRID_SIZE) -> Image.Image:
    """Render one layer's vision-token logit-lens predictions overlaid on the image."""
    W, H = image.size
    pw, ph = W // grid_w, H // grid_h

    prob_grid    = probs_1d.reshape(grid_h, grid_w).astype(np.float32)
    heatmap_full = cv2.resize(_colormap_heatmap(prob_grid), (W, H),
                              interpolation=cv2.INTER_NEAREST)
    img_np   = np.array(image).astype(np.float32)
    blended  = np.clip((1 - alpha) * img_np + alpha * heatmap_full, 0, 255).astype(np.uint8)
    frame    = Image.fromarray(blended)
    draw     = ImageDraw.Draw(frame)

    # grid lines
    for gx in range(0, W, pw):
        draw.line([(gx, 0), (gx, H)], fill=(80, 80, 80), width=1)
    for gy in range(0, H, ph):
        draw.line([(0, gy), (W, gy)], fill=(80, 80, 80), width=1)

    # patch token labels
    font_size = max(6, min(pw, ph) // 4)
    font = _load_font(font_size)
    for row in range(grid_h):
        for col in range(grid_w):
            idx  = row * grid_w + col
            word = words_1d[idx].strip() or "·"
            cx   = col * pw + pw // 2
            cy   = row * ph + ph // 2
            draw.text((cx - 1, cy - 1), word, fill=(0, 0, 0), font=font, anchor="mm")
            draw.text((cx, cy),         word, fill=(255, 255, 255), font=font, anchor="mm")

    # title bar
    title_h    = max(40, H // 14)
    title_bar  = Image.new("RGB", (W, title_h), color=(20, 20, 20))
    td         = ImageDraw.Draw(title_bar)
    td.text((W // 2, title_h // 2),
            f"Logit Lens — Layer {layer_idx}",
            fill=(255, 255, 255), font=_load_font(title_h // 2), anchor="mm")
    out = Image.new("RGB", (W, H + title_h))
    out.paste(title_bar, (0, 0))
    out.paste(frame,     (0, title_h))
    return out


# ── shared palette ────────────────────────────────────────────────────────────
_SEC_BG = {
    "system":    np.array([  0,  55,  55], np.float32),
    "task":      np.array([  0,  30, 110], np.float32),
    "generated": np.array([ 80,  20,   0], np.float32),
    "spaces":    np.array([ 70,   0,  90], np.float32),
}
_HDR_BG = {
    "system":    ( 0, 100,  90),
    "task":      ( 0,  55, 155),
    "generated": (130,  40,   0),
    "spaces":    (120,   0, 150),
}
_SEC_LABELS = {
    "system":    "SYSTEM PROMPT",
    "task":      "TASK PROMPT",
    "generated": "GENERATED TOKENS",
    "spaces":    "SPACE TOKENS (inserted)",
}
_GRID_LINE = (50, 50, 50)


def _render_token_sections(
        sections: list,
        grid_w: int,
        cell_px: int,
        cmap,
) -> Image.Image | None:
    """
    Render a list of (sec_name, probs, words, labels) tuples as a 2D cell grid.

    Returns None if all sections are empty.
    """
    sections = [(n, p, w, l) for n, p, w, l in sections if len(l) > 0]
    if not sections:
        return None

    hdr_h   = max(18, cell_px // 2)
    total_w = grid_w * cell_px
    lbl_h   = max(7, cell_px // 6)

    def _rows(n):
        return max(1, (n + grid_w - 1) // grid_w)

    total_h = sum(hdr_h + _rows(len(l)) * cell_px for _, _, _, l in sections)

    canvas = Image.new("RGB", (total_w, total_h), color=(18, 18, 18))
    draw   = ImageDraw.Draw(canvas)
    font_pred = _load_font(max(6, cell_px // 4))
    font_lbl  = _load_font(lbl_h)
    font_hdr  = _load_font(max(8, hdr_h * 2 // 3))

    y = 0
    for sec_name, probs, words, labels in sections:
        n   = len(labels)
        hbg = _HDR_BG.get(sec_name, (60, 60, 60))

        # header
        draw.rectangle([0, y, total_w - 1, y + hdr_h - 1], fill=hbg)
        draw.text((total_w // 2, y + hdr_h // 2),
                  _SEC_LABELS.get(sec_name, sec_name.upper()),
                  fill=(240, 240, 240), font=font_hdr, anchor="mm")
        y += hdr_h

        # cells
        sec_tint = _SEC_BG.get(sec_name, np.array([40, 40, 40], np.float32))
        for i, lbl in enumerate(labels):
            row_i = i // grid_w
            col_i = i % grid_w
            cx    = col_i * cell_px
            cy    = y + row_i * cell_px

            prob = float(probs[i]) if i < len(probs) else 0.0
            rgba = cmap(prob)
            base = np.array([rgba[0]*255, rgba[1]*255, rgba[2]*255], np.float32)
            tint = np.clip(base * 0.72 + sec_tint * 0.28, 0, 255).astype(np.uint8)

            draw.rectangle([cx, cy, cx+cell_px-1, cy+cell_px-1],
                           fill=tuple(int(v) for v in tint))
            draw.rectangle([cx, cy, cx+cell_px-1, cy+cell_px-1],
                           outline=_GRID_LINE, width=1)

            pred = (words[i] if i < len(words) else "").strip() or "·"
            draw.text((cx+cell_px//2, cy+cell_px//2 - lbl_h//2 - 1),
                      pred[:8], fill=(255, 255, 255), font=font_pred, anchor="mm")
            draw.text((cx+cell_px//2, cy+cell_px-2),
                      str(lbl)[:6], fill=(190, 190, 190), font=font_lbl, anchor="mb")

        y += _rows(n) * cell_px

    return canvas


def _extract_sections(text_data, layer_fi, section_info):
    """
    Return (before_segs, after_segs, gen_seg) as lists of
    (sec_name, probs, words, labels) tuples for the given layer.
    """
    def _get(key):
        sec = text_data.get(key, {})
        p = sec.get("probs", np.zeros((1, 0)))
        w = sec.get("words", [[]])
        l = sec.get("labels", [])
        if p.shape[0] == 0 or layer_fi >= p.shape[0]:
            return np.array([]), [], []
        return p[layer_fi], (w[layer_fi] if layer_fi < len(w) else []), l

    bp, bw, bl = _get("before")
    ap, aw, al = _get("after")
    gp, gw_, gl = _get("generated")

    def _segment(probs, words, labels, named_sizes):
        out, pos = [], 0
        for name, size in named_sizes:
            end = min(pos + size, len(labels))
            p = probs[pos:end] if len(probs) else np.array([])
            w = list(words[pos:end]) if words else []
            out.append((name, p, w, list(labels[pos:end])))
            pos = end
        return out

    before_segs = _segment(bp, bw, bl, section_info.get("before_img", []))
    after_segs  = _segment(ap, aw, al, section_info.get("after_img",  []))
    gen_seg     = [("generated", gp, gw_, gl)]
    return before_segs, after_segs, gen_seg


def render_text_grid(
        text_data: dict,
        vision_probs_1d: np.ndarray,
        vision_words: list,
        layer_fi: int,
        section_info: dict,
        grid_h: int,
        grid_w: int,
        cell_px: int,
        cmap_name: str = "plasma",
) -> Image.Image | None:
    """Public helper kept for callers that want the full grid in one image."""
    cmap = plt.get_cmap(cmap_name)
    before_segs, after_segs, gen_seg = _extract_sections(text_data, layer_fi, section_info)
    all_segs = before_segs + after_segs + gen_seg
    return _render_token_sections(all_segs, grid_w, cell_px, cmap)


# ── kept for backward compatibility (CLI --no-text-lens path not used here) ──
def render_text_strip(text_data: dict, layer_fi: int,
                      width_px: int, strip_height_px: int,
                      cmap_name: str = "plasma") -> Image.Image:
    """
    Render a horizontal strip showing text-token logit-lens for ONE layer.

    Sections: [Input before image] | [Input after image] | [Generated tokens]
    Cell color = max-token probability, cell text = predicted token,
    x-tick label = actual token.
    """
    before = text_data['before']
    after  = text_data['after']
    gen    = text_data['generated']

    # pull out this layer's slice
    def _layer_slice(section):
        return section['probs'][layer_fi], section['words'][layer_fi], section['labels']

    bp, bw, bl = _layer_slice(before)
    ap, aw, al = _layer_slice(after)
    gp, gw, gl = _layer_slice(gen)

    nb, na, ng = len(bl), len(al), len(gl)
    GAP        = 1
    total_cols = nb + GAP + na + GAP + ng

    # build combined arrays with gap columns (prob=0, word='')
    all_probs  = np.zeros((1, total_cols))
    all_pred   = [''] * total_cols
    all_labels = [''] * total_cols

    def _fill(offset, probs, words, labels):
        all_probs[0, offset:offset + len(probs)] = probs
        all_pred[offset:offset + len(words)]     = words
        all_labels[offset:offset + len(labels)]  = labels

    _fill(0,              bp, bw, bl)
    _fill(nb + GAP,       ap, aw, al)
    _fill(nb + GAP + na + GAP, gp, gw, gl)

    # section boundary column ranges (for background shading)
    sec_ranges = [
        (0,          nb,            "#003333", "Input\n(before img)", "cyan"),
        (nb + GAP,   nb + GAP + na, "#002244", "Input\n(after img)",  "cyan"),
        (nb + GAP + na + GAP, nb + GAP + na + GAP + ng, "#332200", "Generated", "yellow"),
    ]

    # ── matplotlib figure ────────────────────────────────────────────────────
    DPI   = 150
    fig_w = width_px  / DPI
    fig_h = strip_height_px / DPI
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("#111111")
    ax.set_facecolor("#111111")

    # ── colored background bands (drawn first, behind heatmap) ───────────────
    import matplotlib.patches as mpatches
    band_colors = {"cyan": "#003333", "yellow": "#332200"}
    for (c0, c1, bg, _, _) in sec_ranges:
        ax.axvspan(c0 - 0.5, c1 - 0.5, ymin=0, ymax=1,
                   facecolor=bg, alpha=0.5, zorder=0)

    ax.imshow(all_probs, aspect="auto", cmap=cmap_name, vmin=0, vmax=1,
              extent=[-0.5, total_cols - 0.5, 0.5, -0.5], zorder=1)

    # predicted-token text inside cells
    fs_cell = max(4, min(9, int(0.55 * width_px / total_cols)))
    for ci in range(total_cols):
        w = all_pred[ci]
        if w:
            ax.text(ci, 0, w, ha="center", va="center",
                    color="white", fontsize=fs_cell, fontweight="bold",
                    clip_on=True, zorder=2)

    # ── section separator lines ───────────────────────────────────────────────
    # Draw on both sides of each gap column so sections are clearly bounded
    ax.axvline(nb - 0.5,                    color="cyan",   lw=2, linestyle="--", alpha=0.9, zorder=3)
    ax.axvline(nb + GAP - 0.5,              color="cyan",   lw=2, linestyle="--", alpha=0.9, zorder=3)
    ax.axvline(nb + GAP + na - 0.5,         color="yellow", lw=2, linestyle="--", alpha=0.9, zorder=3)
    ax.axvline(nb + GAP + na + GAP - 0.5,   color="yellow", lw=2, linestyle="--", alpha=0.9, zorder=3)

    # ── section header labels (inside the top of each band) ──────────────────
    fs_hdr = max(5, fs_cell + 1)
    for (c0, c1, _, label, color) in sec_ranges:
        mid = (c0 + c1 - 1) / 2
        ax.text(mid, -0.45, label, ha="center", va="bottom",
                color=color, fontsize=fs_hdr, fontweight="bold",
                linespacing=1.1, zorder=4)

    # ── x-axis: actual token labels (vertical, stay within their column) ─────
    ax.set_xticks(range(total_cols))
    fs_tick = max(4, fs_cell - 1)
    ax.set_xticklabels(all_labels, fontsize=fs_tick, rotation=90,
                       ha="center", va="top", color="white")
    ax.tick_params(axis="x", colors="white", pad=1, length=2)

    # y-axis: hidden
    ax.set_yticks([])
    ax.set_ylim(0.55, -0.80)   # headroom for section headers above
    ax.set_xlim(-0.5, total_cols - 0.5)

    for sp in ax.spines.values():
        sp.set_visible(False)

    plt.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf).resize((width_px, strip_height_px), Image.LANCZOS)


def _make_colorbar_frame(width, height, cmap_name="plasma"):
    """Horizontal colorbar legend with confidence tick labels → PIL image."""
    bar_h = max(64, height // 12)
    fig, ax = plt.subplots(figsize=(width / 150, bar_h / 150), dpi=150)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.70, bottom=0.40)
    sm = plt.cm.ScalarMappable(cmap=cmap_name,
                               norm=mcolors.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=1.0, pad=0)
    # Explicit confidence ticks so each colour's probability is readable.
    ticks = np.linspace(0.0, 1.0, 11)
    cb.set_ticks(ticks)
    cb.set_ticklabels([f"{t:.1f}" for t in ticks])
    cb.set_label("Max token probability (confidence)", color="white", fontsize=8)
    cb.ax.xaxis.set_tick_params(color="white", labelsize=7)
    plt.setp(cb.ax.xaxis.get_ticklabels(), color="white")
    fig.patch.set_facecolor("#141414")
    ax.set_visible(False)
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)
    return Image.fromarray(buf).resize((width, bar_h), Image.LANCZOS)


def render_combined_frame(
        disp_image: Image.Image,
        probs_1d: np.ndarray, words_1d: list,
        layer_idx: int, layer_fi: int,
        text_data: dict | None,
        alpha: float,
        section_info: dict | None = None,
        grid_h: int = GRID_SIZE,
        grid_w: int = GRID_SIZE,
) -> Image.Image:
    """
    Compose one GIF frame stacked in prompt order (top → bottom):

        [before-image text tokens]   ← system / task (if before image)
        [image spatial overlay]      ← vision logit-lens heatmap
        [after-image text tokens]    ← task / system (if after image)
        [generated tokens]
    """
    W       = disp_image.width
    cell_px = W // grid_w           # same pixel size as one image patch
    cmap    = plt.get_cmap("plasma")
    si      = section_info or {"order": "IT", "before_img": [], "after_img": [("task", 999)]}

    # ── build text section panels ─────────────────────────────────────────────
    if text_data is not None:
        before_segs, after_segs, gen_seg = _extract_sections(text_data, layer_fi, si)
        before_grid = _render_token_sections(before_segs, grid_w, cell_px, cmap)
        after_grid  = _render_token_sections(after_segs,  grid_w, cell_px, cmap)
        gen_grid    = _render_token_sections(gen_seg,      grid_w, cell_px, cmap)
    else:
        before_grid = after_grid = gen_grid = None

    # ── image overlay (centre of the stack) ──────────────────────────────────
    vision = render_vision_frame(disp_image, probs_1d, words_1d, layer_idx, alpha,
                                 grid_h=grid_h, grid_w=grid_w)

    # ── stack panels in prompt order ──────────────────────────────────────────
    def _maybe_resize(img):
        if img is not None and img.width != W:
            return img.resize((W, img.height), Image.NEAREST)
        return img

    before_grid = _maybe_resize(before_grid)
    after_grid  = _maybe_resize(after_grid)
    gen_grid    = _maybe_resize(gen_grid)

    divider = Image.new("RGB", (W, 6), color=(60, 60, 80))

    panels = []
    if before_grid:
        panels += [before_grid, divider]
    panels += [vision]
    if after_grid:
        panels += [divider, after_grid]
    if gen_grid:
        panels += [divider, gen_grid]

    total_h = sum(p.height for p in panels)
    out     = Image.new("RGB", (W, total_h), color=(20, 20, 20))
    y = 0
    for p in panels:
        out.paste(p, (0, y))
        y += p.height
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    setup_seeds()
    disable_torch_init()

    print("Loading model …")
    model_manager = ModelManager(args.model)

    img_path = args.img_path or os.path.join(
        args.data_path, f"COCO_val2014_{str(args.image_id).zfill(12)}.jpg"
    )
    print(f"Image: {img_path}")
    img = Image.open(img_path).convert("RGB")

    _, input_ids, kwargs = model_manager.prepare_inputs_from_pil(
        [args.query], img,
        system_prompt=args.system_prompt,
        order=args.order,
    )
    grid_h, grid_w   = model_manager.grid_h, model_manager.grid_w
    section_info      = model_manager.section_info

    print("Running inference …")
    with torch.inference_mode():
        outputs = model_manager.llm_model.generate(
            input_ids,
            do_sample=False,
            num_beams=args.beam,
            max_new_tokens=args.max_tokens,
            use_cache=True,
            output_hidden_states=True,
            return_dict_in_generate=True,
            **kwargs,
        )

    answer = model_manager.tokenizer.batch_decode(
        outputs["sequences"], skip_special_tokens=True
    )[0].strip()
    print(f"Generated: {answer}\n")

    logits_warper    = TopKLogitsWarper(top_k=args.top_k, filter_value=float("-inf"))
    logits_processor = LogitsProcessorList([])

    print(f"Computing vision logit lens ({len(args.layer_range)} layers) …")
    all_probs, all_words = logit_lens_all_vision_tokens(
        model_manager.llm_model, model_manager.tokenizer,
        input_ids, outputs,
        model_manager.img_start_idx,
        args.layer_range,
        logits_warper, logits_processor,
        grid_h=grid_h, grid_w=grid_w,
    )

    text_data = None
    if not args.no_text_lens:
        print("Computing text logit lens …")
        text_data = compute_text_logit_lens(
            model_manager.llm_model, model_manager.tokenizer,
            input_ids, outputs,
            model_manager.img_start_idx, model_manager.img_end_idx,
            args.layer_range, logits_warper, logits_processor,
        )

    disp_image = img.resize((args.resolution, args.resolution), Image.LANCZOS)
    colorbar   = _make_colorbar_frame(args.resolution, args.resolution)

    if args.save_frames:
        os.makedirs(args.frames_dir, exist_ok=True)

    frames = []
    for fi, layer_idx in enumerate(args.layer_range):
        frame = render_combined_frame(
            disp_image, all_probs[fi], all_words[fi],
            layer_idx, fi, text_data, args.alpha,
            section_info=section_info,
            grid_h=grid_h, grid_w=grid_w,
        )
        # colorbar at bottom
        full = Image.new("RGB", (frame.width, frame.height + colorbar.height))
        full.paste(frame,    (0, 0))
        full.paste(colorbar, (0, frame.height))
        frames.append(full)
        print(f"  Rendered layer {layer_idx} ({fi + 1}/{len(args.layer_range)})")

        if args.save_frames:
            full.save(os.path.join(args.frames_dir, f"layer_{layer_idx:03d}.png"))

    duration_ms = int(1000 / args.fps)
    imageio.mimsave(
        args.output_gif,
        [np.array(f) for f in frames],
        format="GIF", duration=duration_ms, loop=0,
    )
    print(f"\nSaved GIF  → {args.output_gif}  ({len(frames)} frames @ {args.fps} fps)")
    if args.save_frames:
        print(f"Saved PNGs → {args.frames_dir}/")


if __name__ == "__main__":
    main()
