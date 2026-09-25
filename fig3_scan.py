#!/usr/bin/env python3
"""
Scan high-resolution NaturalBench scenes for the new Fig. 3.

The new Fig. 3 has to carry two claims at once:
  (a) LOCALIZATION -- with no question, each region's patches decode, under the
      logit lens, to that region's own object (this is the claim Sec. 4.1 cites);
  (b) STEERING -- with a question placed BEFORE the image (STI), the same patches
      decode to question-relevant words instead, and differently for two different
      questions.

Only some scenes make that legible: we need several LARGE, well-separated regions
whose patches decode to clean, distinct English nouns. This script measures
exactly that on the image-only pass and ranks candidates, so the figure's own
metric picks the scene rather than taste alone.

Per image: one image-only forward pass -> logit lens at LAYER -> patches grouped
by decoded word stem -> "concepts" = word clusters that are big enough and
spatially compact. Score rewards many concepts, large clusters, tight clusters,
well-separated centroids, and clean English decodings.

    CUDA_VISIBLE_DEVICES=1 python fig3_scan.py --num 400
"""
import argparse, glob, json, os, re
from collections import defaultdict

import numpy as np, torch
from PIL import Image
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper

from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from logit_lens_overlay import logit_lens_all_vision_tokens

NB_IMAGES = "/home/grg/Research/middle_layers_indicating_hallucinations/naturalbench/images"
LAYER = 28
LONG_SIDE = 896

# Words that decode everywhere and name nothing in the scene: they inflate the
# cluster count without giving the figure a callout worth drawing.
GENERIC = set("""
background backdrop scene image photo picture view color colour colors gray grey
white black blue green red yellow brown dark light bright blur blurry blurred
texture pattern surface area region part parts side left right top bottom center
centre middle front back near far above below inside outside something someone
thing things stuff other others more most less least very much many few some any
what which where when who how why the and but for with from into onto over under
looks look looking seems seem appears appear shows show showing being been having
""".split())

WORD_RE = re.compile(r"^[a-z]{3,}$")


def stem(w):
    for suf in ("ing", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            return w[: -len(suf)]
    return w


def clean(word):
    w = word.strip().lower()
    return w if WORD_RE.fullmatch(w) and w not in GENERIC else None


def concepts(words, gh, gw, min_patches=5, max_spread=0.26):
    """Word clusters that are big and spatially compact -> callout candidates."""
    pos = defaultdict(list)
    for i, raw in enumerate(words):
        w = clean(raw)
        if w:
            pos[stem(w)].append((i // gw, i % gw))
    out = []
    for s, ps in pos.items():
        if len(ps) < min_patches:
            continue
        a = np.asarray(ps, float)
        a[:, 0] /= gh
        a[:, 1] /= gw
        c = a.mean(0)
        spread = float(np.sqrt(((a - c) ** 2).sum(1)).mean())
        if spread > max_spread:
            continue                      # word sprayed over the whole image
        # surface form: the most common raw spelling in this cluster
        forms = [clean(words[r * gw + cc]) for r, cc in ps]
        form = max(set(forms), key=forms.count)
        out.append({"word": form, "stem": s, "n": len(ps),
                    "spread": round(spread, 3),
                    "cy": round(float(c[0]), 3), "cx": round(float(c[1]), 3)})
    out.sort(key=lambda d: -(d["n"] / (0.12 + d["spread"])))
    return out


def score(cs, en_frac):
    """Reward several big, tight, well-separated concepts on a clean-English pass."""
    top = cs[:5]
    if len(top) < 3:
        return -1.0, 0.0
    n = np.array([c["n"] for c in top], float)
    sp = np.array([c["spread"] for c in top], float)
    ctr = np.array([[c["cy"], c["cx"]] for c in top])
    d = [np.linalg.norm(ctr[i] - ctr[j])
         for i in range(len(top)) for j in range(i + 1, len(top))]
    min_sep = float(min(d))
    s = (2.0 * len(top)
         + 0.10 * n.sum()
         - 6.0 * sp.mean()
         + 6.0 * min_sep
         + 4.0 * en_frac)
    return float(s), min_sep


def render(pil, words, gh, gw, cs, path, title):
    """Word grid over the image + a marker at each concept centroid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(gw * 0.42, gh * 0.42), dpi=70)
    ax.imshow(pil.resize((gw * 40, gh * 40)), extent=[0, gw, gh, 0])
    keep = {c["stem"] for c in cs[:5]}
    for i, raw in enumerate(words):
        w = clean(raw)
        if not w:
            continue
        r, c = i // gw, i % gw
        hot = stem(w) in keep
        ax.text(c + 0.5, r + 0.5, w[:9], ha="center", va="center", fontsize=5.5,
                color="yellow" if hot else "white",
                fontweight="bold" if hot else "normal",
                bbox=dict(fc="black", alpha=0.55 if hot else 0.25, pad=0.4, lw=0))
    ax.set_xlim(0, gw); ax.set_ylim(gh, 0); ax.axis("off")
    ax.set_title(title, fontsize=9)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=400)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--pool-file", default=None, dest="pool_file",
                    help="newline-separated image paths to scan instead of NaturalBench")
    ap.add_argument("--top", type=int, default=16)
    ap.add_argument("--out", default="fig3_scan")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    if args.pool_file:
        pool = [l.strip() for l in open(args.pool_file) if l.strip()]
    else:
        pool = []
        for f in sorted(glob.glob(os.path.join(NB_IMAGES, "*.jpg"))):
            try:
                w, h = Image.open(f).size
            except Exception:
                continue
            if min(w, h) >= 1000 and 1.15 <= w / h <= 1.8:
                pool.append(f)
    pool = pool[:: args.stride][: args.num]
    print(f"[scan] {len(pool)} candidate scenes", flush=True)

    mm = ModelManager("qwen3-vl-8b")
    warper, proc = TopKLogitsWarper(top_k=50, filter_value=float("-inf")), LogitsProcessorList([])

    rows = []
    for k, f in enumerate(pool):
        pil = Image.open(f).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            _, input_ids, kwargs = mm.prepare_inputs_from_pil([""], pil, system_prompt="", order="I")
            with torch.inference_mode():
                out = mm.llm_model(input_ids, output_hidden_states=True, use_cache=False, **kwargs)
            _, words = logit_lens_all_vision_tokens(
                mm.llm_model, mm.tokenizer, input_ids,
                {"hidden_states": (out.hidden_states,)}, mm.img_start_idx,
                [LAYER], warper, proc, grid_h=mm.grid_h, grid_w=mm.grid_w)
        except Exception as e:
            print(f"  [skip] {os.path.basename(f)}: {e}", flush=True)
            continue
        w0 = words[0]
        en_frac = sum(1 for x in w0 if clean(x)) / len(w0)
        cs = concepts(w0, mm.grid_h, mm.grid_w)
        sc, sep = score(cs, en_frac)
        rows.append({"file": f, "score": round(sc, 3), "en": round(en_frac, 3),
                     "sep": round(sep, 3), "grid": [mm.grid_h, mm.grid_w],
                     "concepts": cs[:6], "words": w0})
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(pool)}] best={max(r['score'] for r in rows):.2f}", flush=True)
        del out
    rows.sort(key=lambda r: -r["score"])

    os.makedirs(args.out, exist_ok=True)
    json.dump([{k: v for k, v in r.items() if k != "words"} for r in rows[:60]],
              open(os.path.join(args.out, "ranked.json"), "w"), indent=2)
    print("\n[scan] top scenes:")
    for i, r in enumerate(rows[: args.top]):
        names = ", ".join(f"{c['word']}({c['n']})" for c in r["concepts"][:5])
        print(f" {i:2d}. {os.path.basename(r['file']):18s} score={r['score']:6.2f} "
              f"en={r['en']:.2f} sep={r['sep']:.2f}  {names}")
        pil = Image.open(r["file"]).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        render(pil, r["words"], r["grid"][0], r["grid"][1], r["concepts"],
               os.path.join(args.out, f"cand{i:02d}.png"),
               f"{i}: {os.path.basename(r['file'])}  |  {names}")
    print(f"[scan] renders -> {args.out}/cand*.png")


if __name__ == "__main__":
    main()
