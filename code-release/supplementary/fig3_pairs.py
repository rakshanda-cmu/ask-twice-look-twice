#!/usr/bin/env python3
"""
For ONE scene, try every question pair and report which pair makes the steered
words most clearly about the question.

For each object X in the scene we ask "What does the {X} look like?" and count,
for every object Y, how many patches decode to a word in Y's vocabulary. The
diagonal of that matrix is what the figure needs to show: asking about X should
raise X's own words above the image-first baseline, and above what the OTHER
question produces for X.

    CUDA_VISIBLE_DEVICES=1 python fig3_pairs.py \
        --file COCO_val2014_000000253665.jpg --objects "laptop,couch,cat"
"""
import argparse, itertools, os

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names
from fig3_probe import topk_patch_tokens
from fig3_rescan import IMGDIR, LONG_SIDE, LAYER, phrase
from fig3_uptake import uptake


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--objects", required=True)
    ap.add_argument("--layer", type=int, default=LAYER)
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    objs = [o.strip() for o in args.objects.split(",")]
    pil = Image.open(os.path.join(IMGDIR, args.file)).convert("RGB")
    s = LONG_SIDE / max(pil.size)
    pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)

    mm = ModelManager("qwen3-vl-8b")
    qs = {o: f"What does the {phrase(o)} look like?" for o in objs}

    base = topk_patch_tokens(mm, pil, args.layer, qs[objs[0]], "SIT",
                             SYSTEM_MESSAGE, english=True)[0]
    sti = {o: topk_patch_tokens(mm, pil, args.layer, qs[o], "STI",
                                SYSTEM_MESSAGE, english=True)[0] for o in objs}

    print(f"\n{args.file}  layer {args.layer}")
    print("uptake matrix: rows = question asked, cols = whose words appear\n")
    hdr = f"{'asked':>16s} | " + " ".join(f"{o:>13s}" for o in objs)
    print(hdr); print("-" * len(hdr))
    print(f"{'(image-first)':>16s} | " + " ".join(f"{uptake(base, o):13d}" for o in objs))
    for a in objs:
        print(f"{'STI: ' + a:>16s} | " + " ".join(f"{uptake(sti[a], o):13d}" for o in objs))

    print("\nbest question pairs (gain over image-first, and specificity vs the other question):")
    scored = []
    for A, B in itertools.combinations(objs, 2):
        gA = uptake(sti[A], A) - uptake(base, A)
        gB = uptake(sti[B], B) - uptake(base, B)
        sA = uptake(sti[A], A) - uptake(sti[B], A)
        sB = uptake(sti[B], B) - uptake(sti[A], B)
        scored.append((min(gA, gB) * 2 + min(sA, sB) * 2 + gA + gB, A, B, gA, gB, sA, sB))
    for sc, A, B, gA, gB, sA, sB in sorted(scored, reverse=True):
        print(f"  score={sc:3d}  q1={A:12s} q2={B:12s} "
              f"gain({A})={gA:+3d} gain({B})={gB:+3d} "
              f"spec({A})={sA:+3d} spec({B})={sB:+3d}")


if __name__ == "__main__":
    main()
