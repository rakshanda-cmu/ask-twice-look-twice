"""
Re-render ONLY the 'space' variant for every example in the manifest, after the
fix that makes the 20 inserted space tokens visible as a labeled SPACE TOKENS
section in the token grid. The 'normal' variant is untouched.

Resumable: skips examples whose space renders are already complete. Updates the
manifest in place (merging the new space dirs/answers).

Run:
    CUDA_VISIBLE_DEVICES=1 python precompute_space.py
"""

import os
import time

from model_manager import ModelManager
from utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hf_logging

import midlayer_core as mc


def _parse(code):
    g, i, q = code.split("_")
    return int(g[1:]), int(i[1:]), int(q[1:])


def main():
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    man = mc.load_manifest()
    exs = man["examples"]
    print(f"[load] manifest with {len(exs)} examples")

    mm = ModelManager("qwen3-vl-8b")
    t0 = time.time()
    for n, e in enumerate(exs, 1):
        code = e["code"]
        g, i, q = _parse(code)
        rec = mc.build_example_record("qwen3-vl-8b", g, i, q)

        # Always (re)capture the exact passed prompt for both variants — cheap.
        e["prompt_info"] = mc.capture_prompt_info(mm, rec)

        if mc.is_example_complete(code, mm.num_layers, variants=("space",)):
            mc.save_manifest(man)
            print(f"[{n}/{len(exs)}] {code} space render already done — "
                  f"prompt_info captured")
            continue

        rec2 = mc.process_example(
            mm, rec, resolution=640, variants=("space",),
            progress_cb=lambda msg: print("   ", msg), ts=time.time(),
        )
        e.setdefault("dirs", {})["space"] = rec2["dirs"]["space"]
        e.setdefault("answers_generated", {})["space"] = rec2["answers_generated"]["space"]
        mc.save_manifest(man)
        print(f"[{n}/{len(exs)}] {code} space re-rendered: "
              f"{rec2['answers_generated']['space']}")

    print(f"\n[done] space variant re-rendered ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
