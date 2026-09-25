#!/usr/bin/env python3
"""
Show WHICH words question-first adds, per candidate scene.

"The steered words make sense about the question" is a claim about vocabulary, so
print it: for each question, the words that gain the most patches relative to the
image-first read-out of the same scene. If the diagnosis is right, asking about a
couch should add couch/sofa/cushion words, and asking about a plant should add
plant/foliage words.

    CUDA_VISIBLE_DEVICES=1 python fig3_words.py --files a.jpg,b.jpg
"""
import argparse, json, os
from collections import Counter

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import clean
from fig3_probe import topk_patch_tokens
from fig3_rescan import IMGDIR, LONG_SIDE, LAYER


def counts(words):
    return Counter(clean(t[0]) for t in words if clean(t[0]))


def added(base, new, k=10):
    b, n = counts(base), counts(new)
    gains = [(n[w] - b.get(w, 0), w, n[w], b.get(w, 0)) for w in n]
    gains = [g for g in gains if g[0] > 0]
    gains.sort(reverse=True)
    return gains[:k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uptake", default="fig3_uptake.json")
    ap.add_argument("--files", required=True)
    ap.add_argument("--layer", type=int, default=LAYER)
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    rows = {r["file"]: r for r in json.load(open(args.uptake))}
    mm = ModelManager("qwen3-vl-8b")

    for f in [x.strip() for x in args.files.split(",")]:
        r = rows[f]
        pil = Image.open(os.path.join(IMGDIR, f)).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        base = topk_patch_tokens(mm, pil, args.layer, r["q1"], "SIT", SYSTEM_MESSAGE, english=True)[0]
        s1 = topk_patch_tokens(mm, pil, args.layer, r["q1"], "STI", SYSTEM_MESSAGE, english=True)[0]
        s2 = topk_patch_tokens(mm, pil, args.layer, r["q2"], "STI", SYSTEM_MESSAGE, english=True)[0]
        print(f"\n=== {f}  loc={r['localized']}/{r['n_anchors']}  "
              f"regions={[a['name'] for a in r['anchors']]}")
        for tag, q, w in (("q1 " + r["A"], r["q1"], s1), ("q2 " + r["B"], r["q2"], s2)):
            got = added(base, w)
            print(f"  {tag:22s} {q}")
            print("      + " + ", ".join(f"{w_}({g:+d}→{n})" for g, w_, n, _ in got))


if __name__ == "__main__":
    main()
