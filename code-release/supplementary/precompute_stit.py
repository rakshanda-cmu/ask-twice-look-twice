"""
Render the STIT ordering (System·Task·Image·Task — question repeated after the
image) logit lens for every example already in the Middle-Layer manifest, so the
UI can show STIT-vs-STI and STIT-vs-IST per-layer comparisons.

Only the 'normal' variant / STIT order is added; the existing IST & STI renders
are untouched. Resumable.

Run:
    CUDA_VISIBLE_DEVICES=0 python precompute_stit.py
"""

import os
import time

import imageio
import numpy as np
from PIL import Image

from model_manager import ModelManager
from utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hf_logging
from constants import SYSTEM_MESSAGE
from logit_lens_runner import run_logit_lens
from naturalbench_eval import answer_suffix
import midlayer_core as mc

ORDER = "STIT"
VARIANT = "normal"


def _parse(code):
    g, i, q = code.split("_")
    return int(g[1:]), int(i[1:]), int(q[1:])


def _done(code, num_layers):
    d = mc._variant_dir(code, VARIANT, ORDER)
    if not os.path.isdir(d):
        return False
    layers = [f for f in os.listdir(d)
              if f.startswith("layer_") and f.endswith(".png")]
    if len(layers) < num_layers:
        return False
    return mc._img_ok(os.path.join(d, "final.png")) and \
        mc._img_ok(os.path.join(d, "anim.gif"))


def main():
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    man = mc.load_manifest()
    exs = man["examples"]
    print(f"[load] manifest with {len(exs)} examples")

    mm = ModelManager("qwen3-vl-8b")
    layer_range = mc.get_layer_range(mm)
    t0 = time.time()

    for n, e in enumerate(exs, 1):
        code = e["code"]
        # make sure the manifest references the STIT dir
        e.setdefault("dirs", {}).setdefault(VARIANT, {})[ORDER] = \
            mc._variant_dir(code, VARIANT, ORDER)

        if _done(code, mm.num_layers):
            mc.save_manifest(man)
            print(f"[{n}/{len(exs)}] {code} STIT already done — skip")
            continue

        g, i, q = _parse(code)
        rec = mc.build_example_record("qwen3-vl-8b", g, i, q)
        img = Image.open(os.path.join(mc.NB_DIR, rec["image_rel"])).convert("RGB")
        query = rec["question"] + answer_suffix(rec["question_type"])

        ans, frames = run_logit_lens(
            mm, img, query, system_prompt=SYSTEM_MESSAGE, order=ORDER,
            layer_range=layer_range, top_k=50, max_tokens=16,
            resolution=640, alpha=0.55, show_text_lens=True, n_spaces=0,
        )
        odir = mc._variant_dir(code, VARIANT, ORDER)
        os.makedirs(odir, exist_ok=True)
        for li, fr in zip(layer_range, frames):
            fr.save(os.path.join(odir, f"layer_{li:02d}.png"))
        frames[-1].save(os.path.join(odir, "final.png"))
        imageio.mimsave(os.path.join(odir, "anim.gif"),
                        [np.array(f) for f in frames],
                        format="GIF", duration=400, loop=0)
        e.setdefault("answers_generated", {}).setdefault(VARIANT, {})[ORDER] = ans
        mc.save_manifest(man)
        print(f"[{n}/{len(exs)}] {code} STIT: {ans!r}")

    print(f"\n[done] STIT rendered ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
