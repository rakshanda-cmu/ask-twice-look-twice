#!/usr/bin/env python3
"""
Find the Fig. 3 scene where the STEERED WORDS ARE ABOUT THE QUESTION.

Earlier rankings scored how MANY patches change when the question changes. That
says the read-out moved, not that it moved somewhere meaningful. Here we score
whether it moves toward the thing that was asked about:

    uptake(order, q, X) = # patches whose decoded word belongs to object X's
                          vocabulary, under `order` with question q

A perfect example has, for two questions q1 (about A) and q2 (about B):

    gain_A = uptake(STI, q1, A) - uptake(SIT, A)      # asking about A surfaces A
    gain_B = uptake(STI, q2, B) - uptake(SIT, B)      # asking about B surfaces B
    spec_A = uptake(STI, q1, A) - uptake(STI, q2, A)  # ...and A-words are specific
    spec_B = uptake(STI, q2, B) - uptake(STI, q1, B)  #    to the question that asked

Image-first needs no q1/q2 split: the causal mask makes its read-out identical for
both questions, which every scan so far has confirmed at exactly 0%.

Note this is a claim about WHICH WORDS appear, not about WHERE they appear. The
paper's Sec. G shows the rewrite is spatially diffuse; nothing here contradicts
that, and nothing here should be drawn as a spotlight on the queried object.

    CUDA_VISIBLE_DEVICES=1 python fig3_uptake.py --num 150
"""
import argparse, json, os

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names
from fig3_probe import topk_patch_tokens
from fig3_rescan import IMGDIR, LONG_SIDE, LAYER


def uptake(words, obj):
    """# patches whose top-1 decoded word belongs to `obj`'s vocabulary."""
    return sum(1 for t in words if names(obj, [t[0]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rescan", default="fig3_rescan2.json")
    ap.add_argument("--num", type=int, default=150)
    ap.add_argument("--top", type=int, default=14)
    ap.add_argument("--long-side", type=int, default=LONG_SIDE, dest="long_side",
                    help="render/measure resolution; the word map needs a coarse grid")
    ap.add_argument("--out", default="fig3_uptake.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pool = json.load(open(args.rescan))[: args.num]
    print(f"[uptake] {len(pool)} scenes", flush=True)
    mm = ModelManager("qwen3-vl-8b")

    rows = []
    for k, r in enumerate(pool):
        A, B, q1, q2 = r["A"], r["B"], r["q1"], r["q2"]
        if A == B:
            continue
        pil = Image.open(os.path.join(IMGDIR, r["file"])).convert("RGB")
        s = args.long_side / max(pil.size)
        if s < 1:
            pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            sit = topk_patch_tokens(mm, pil, LAYER, q1, "SIT", SYSTEM_MESSAGE, english=True)[0]
            s1 = topk_patch_tokens(mm, pil, LAYER, q1, "STI", SYSTEM_MESSAGE, english=True)[0]
            s2 = topk_patch_tokens(mm, pil, LAYER, q2, "STI", SYSTEM_MESSAGE, english=True)[0]
        except Exception as e:
            print(f"  [skip] {r['file']}: {e}", flush=True)
            continue

        base_A, base_B = uptake(sit, A), uptake(sit, B)
        a1, a2 = uptake(s1, A), uptake(s2, A)
        b1, b2 = uptake(s1, B), uptake(s2, B)
        gain_A, gain_B = a1 - base_A, b2 - base_B
        spec_A, spec_B = a1 - a2, b2 - b1
        # both questions must work, and each must be specific to its own object
        score = min(gain_A, gain_B) * 2 + min(spec_A, spec_B) * 2 + (gain_A + gain_B)

        rows.append({"file": r["file"], "A": A, "B": B, "q1": q1, "q2": q2,
                     "grid": r["grid"], "anchors": r["anchors"],
                     "base": [base_A, base_B], "sti_q1": [a1, b1], "sti_q2": [a2, b2],
                     "gain": [gain_A, gain_B], "spec": [spec_A, spec_B],
                     "score": score, "steered": r["steered"],
                     "localized": r["localized"], "n_anchors": r["n_anchors"]})
        if (k + 1) % 25 == 0:
            print(f"  [{k+1}/{len(pool)}] best={max(x['score'] for x in rows)}", flush=True)

    rows.sort(key=lambda x: -x["score"])
    json.dump(rows, open(args.out, "w"), indent=2)
    print("\n===== steered words that are ABOUT the question =====")
    print("     (base = image-first uptake; STI q1/q2 = question-first uptake)")
    for x in rows[: args.top]:
        print(f"  {x['file']}  score={x['score']:3d}  loc={x['localized']}/{x['n_anchors']}")
        print(f"      {x['A']:12s}: base {x['base'][0]:3d} -> q1 {x['sti_q1'][0]:3d} "
              f"(q2 {x['sti_q2'][0]:3d})   gain {x['gain'][0]:+d}  spec {x['spec'][0]:+d}")
        print(f"      {x['B']:12s}: base {x['base'][1]:3d} -> q2 {x['sti_q2'][1]:3d} "
              f"(q1 {x['sti_q1'][1]:3d})   gain {x['gain'][1]:+d}  spec {x['spec'][1]:+d}")


if __name__ == "__main__":
    main()
