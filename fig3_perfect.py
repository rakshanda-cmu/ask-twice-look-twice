#!/usr/bin/env python3
"""
Find the "perfect" Fig. 3 patches: ones that answer to whichever object is asked about.

A patch p qualifies when, at the same layer, on the same image:

    image-first        p decodes to something generic (neither A's nor B's words)
    question-first q1  p decodes to A's vocabulary      <- asked about A, says A
    question-first q2  p decodes to B's vocabulary      <- asked about B, says B

That is the cleanest possible statement of "the steered words make sense about the
question": one patch, two questions, each time the read-out lands on the thing that
was asked about, with the image and the layer held fixed.

Scenes are ranked by how many such patches they contain. The qualifying patches
become the figure's callouts, which the caption must state plainly: they are
selected as the clearest instances, illustrative of the mechanism, in the same
spirit as the paper's Fig. 13.

    CUDA_VISIBLE_DEVICES=1 python fig3_perfect.py --num 40
"""
import argparse, json, os

from PIL import Image

from constants import SYSTEM_MESSAGE
from utils import setup_seeds, disable_torch_init
from model_manager import ModelManager
from fig3_anchors import names, clean
from fig3_probe import topk_patch_tokens
from fig3_rescan import IMGDIR, LONG_SIDE, LAYER


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uptake", default="fig3_uptake.json")
    ap.add_argument("--num", type=int, default=40)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--out", default="fig3_perfect.json")
    args = ap.parse_args()

    setup_seeds(); disable_torch_init()
    from transformers.utils import logging as hl; hl.set_verbosity_error()

    pool = json.load(open(args.uptake))[: args.num]
    print(f"[perfect] {len(pool)} scenes", flush=True)
    mm = ModelManager("qwen3-vl-8b")

    rows = []
    for k, r in enumerate(pool):
        A, B = r["A"], r["B"]
        pil = Image.open(os.path.join(IMGDIR, r["file"])).convert("RGB")
        s = LONG_SIDE / max(pil.size)
        pil = pil.resize((int(pil.width * s), int(pil.height * s)), Image.LANCZOS)
        try:
            sit, _, gh, gw = topk_patch_tokens(mm, pil, LAYER, r["q1"], "SIT",
                                               SYSTEM_MESSAGE, english=True)
            s1 = topk_patch_tokens(mm, pil, LAYER, r["q1"], "STI", SYSTEM_MESSAGE, english=True)[0]
            s2 = topk_patch_tokens(mm, pil, LAYER, r["q2"], "STI", SYSTEM_MESSAGE, english=True)[0]
        except Exception as e:
            print(f"  [skip] {r['file']}: {e}", flush=True)
            continue

        hits = []
        for i in range(gh * gw):
            w0, w1, w2 = sit[i][0], s1[i][0], s2[i][0]
            if not (clean(w0) and clean(w1) and clean(w2)):
                continue
            if names(A, [w0]) or names(B, [w0]):
                continue                        # must start out generic
            if names(A, [w1]) and names(B, [w2]) and w1 != w2:
                hits.append({"rc": [i // gw, i % gw],
                             "sit": sit[i], "sti1": s1[i], "sti2": s2[i]})
        rows.append({"file": r["file"], "A": A, "B": B, "q1": r["q1"], "q2": r["q2"],
                     "grid": [gh, gw], "n_perfect": len(hits), "hits": hits,
                     "localized": r["localized"], "n_anchors": r["n_anchors"],
                     "anchors": r["anchors"], "steered": r["steered"]})
        print(f"[{k+1}/{len(pool)}] {r['file'][-10:]} perfect={len(hits):3d} "
              f"loc={r['localized']}/{r['n_anchors']} | {A} vs {B}", flush=True)

    rows.sort(key=lambda x: (x["n_perfect"], x["localized"] == x["n_anchors"]), reverse=True)
    json.dump(rows, open(args.out, "w"), indent=2)
    print("\n===== scenes with patches that answer to whichever object is asked =====")
    for x in rows[: args.top]:
        if not x["n_perfect"]:
            continue
        print(f"  {x['file']}  perfect={x['n_perfect']}  loc={x['localized']}/{x['n_anchors']}"
              f"  | q1={x['A']}  q2={x['B']}")
        for h in x["hits"][:4]:
            print(f"      @{str(h['rc']):9s} image-first {h['sit'][0]:>12s}"
                  f"   ->q1 {h['sti1'][0]:>12s}   ->q2 {h['sti2'][0]:>12s}")


if __name__ == "__main__":
    main()
