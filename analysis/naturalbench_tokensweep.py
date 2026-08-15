#!/usr/bin/env python3
"""
Token-count scaling law for the question-first gap (Qwen3-VL-8B, NaturalBench).

We sweep the image resolution, which sets the number of vision tokens between the
question and the answer under STI (question-first). Prediction from the read-out
mechanism: the STI-vs-SIT group-accuracy gap should GROW with vision-token count,
because more image tokens push the question further from the answer position and it
is read less. SIT (question-last) places the question adjacent to the answer, so its
accuracy should be roughly flat in token count. A monotone gap-vs-tokens curve turns
the diagnosis into a predictive theory.

Loads the model once and reuses `run_experiment` (resumable per-resolution
checkpoints). Saves `naturalbench/tokensweep.json`; plot with `make_tokencount_fig.py`.

Usage:
  CUDA_VISIBLE_DEVICES=0 python naturalbench_tokensweep.py \
      --num-groups 400 --res 224,336,448,560,672
"""
import argparse
import json
import os
from PIL import Image


def probe_tokens(mm, pil, R):
    """Number of vision tokens the model actually produces for an R x R image."""
    img = pil.resize((R, R), Image.LANCZOS)
    mm.prepare_inputs_from_pil(["Is this a test?"], img, system_prompt="", order="IT")
    return int(mm.img_end_idx - mm.img_start_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-8b")
    ap.add_argument("--num-groups", type=int, default=400, dest="num_groups")
    ap.add_argument("--res", default="224,336,448,560,672",
                    help="comma-separated square resolutions to sweep")
    ap.add_argument("--min-side", type=int, default=0, dest="min_side",
                    help="keep only groups where BOTH images have min(w,h) >= this "
                         "(so every sweep resolution is a downscale; removes the "
                         "upscaling confound in the high-token regime)")
    ap.add_argument("--tag", default="", help="extra tag on output files, e.g. 'big'")
    ap.add_argument("--out-json", default="tokensweep.json", dest="out_json")
    ap.add_argument("--nb-dir", default="./naturalbench", dest="nb_dir")
    ap.add_argument("--out-dir", default="./naturalbench/results", dest="out_dir")
    args = ap.parse_args()

    from core.model_manager import ModelManager
    from core.utils import setup_seeds, disable_torch_init
    from transformers.utils import logging as hl
    hl.set_verbosity_error()
    from core.constants import SYSTEM_MESSAGE
    from benchmarks.naturalbench_eval import run_experiment, load_groups
    setup_seeds()
    disable_torch_init()

    resolutions = [int(x) for x in args.res.split(",")]
    groups = load_groups(args.nb_dir)
    if args.min_side > 0:
        def big(g):
            for k in ("image_0", "image_1"):
                with Image.open(os.path.join(args.nb_dir, g[k])) as im:
                    if min(im.size) < args.min_side:
                        return False
            return True
        n0 = len(groups)
        groups = [g for g in groups if big(g)]
        print(f"[filter] min-side>={args.min_side}: kept {len(groups)}/{n0} groups "
              f"(all downscales, no upscaling)", flush=True)
    groups = groups[:args.num_groups]
    print(f"[data] {len(groups)} groups; resolutions {resolutions}", flush=True)
    mm = ModelManager(args.model)

    sample = Image.open(os.path.join(args.nb_dir, groups[0]["image_0"])).convert("RGB")
    rows = []
    for R in resolutions:
        tok = probe_tokens(mm, sample, R)
        row = {"res": R, "tokens": tok}
        for order in ("STI", "SIT"):
            meta, _ = run_experiment(
                mm, groups, order=order, system_prompt=SYSTEM_MESSAGE,
                nb_dir=args.nb_dir, out_dir=args.out_dir, checkpoint_every=100,
                resize=(R, R), resize_mode="exact", tag_suffix=f"res{R}{args.tag}")
            row[order] = meta["g_acc"]
        row["gap"] = row["SIT"] - row["STI"]
        rows.append(row)
        print(f"  R={R:4d} tokens={tok:4d}  STI={row['STI']:.3f} "
              f"SIT={row['SIT']:.3f} gap={row['gap']:+.3f}", flush=True)

    out = {"model": args.model, "num_groups": len(groups),
           "min_side": args.min_side, "rows": rows}
    json.dump(out, open(os.path.join(args.nb_dir, args.out_json), "w"), indent=2)
    print(f"[saved] naturalbench/{args.out_json}", flush=True)


if __name__ == "__main__":
    main()
