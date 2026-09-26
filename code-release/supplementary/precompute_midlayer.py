"""
Batch-run the curated 10 NaturalBench examples (5 IST✓/STI✗ + 5 STI✓/IST✗)
through the logit lens under both orderings and populate the Middle-Layer
Analysis manifest.

Run:
    CUDA_VISIBLE_DEVICES=1 python precompute_midlayer.py
"""

import time

from model_manager import ModelManager
from utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hf_logging

import midlayer_core as mc

MODEL = "qwen3-vl-8b"

# Curated picks as (group_index, image_index, question_index)
# Chosen for clear, non-debatable answers with close-up subjects (zoomed-out
# landscapes excluded so the per-patch token overlay stays readable).
# Scenario A — IST✓/STI✗  (40, close-up & non-debatable, vision-tokens <= 300)
CURATED_A = [
    (13, 1, 1), (22, 0, 1), (28, 1, 0), (30, 0, 0), (38, 0, 0),
    (49, 0, 0), (57, 1, 0), (82, 1, 1), (87, 1, 1), (118, 1, 1),
    (129, 1, 1), (148, 1, 0), (149, 0, 1), (150, 1, 0), (183, 1, 0),
    (185, 1, 0), (199, 0, 0), (202, 0, 0), (217, 1, 0), (244, 0, 0),
    (245, 0, 0), (245, 1, 0), (263, 1, 0), (270, 1, 0), (297, 1, 0),
    (300, 0, 0), (311, 1, 1), (342, 0, 0), (357, 1, 0), (374, 1, 1),
    (391, 0, 1), (408, 1, 0), (427, 1, 1), (428, 0, 1), (434, 0, 0),
    (453, 0, 1), (478, 0, 1), (518, 1, 0), (533, 0, 0), (555, 1, 0),
]
# Scenario B — STI✓/IST✗  (40, close-up & non-debatable, vision-tokens <= 300)
CURATED_B = [
    (22, 0, 0), (27, 0, 1), (38, 1, 0), (57, 0, 1), (93, 0, 0),
    (106, 1, 1), (131, 0, 1), (134, 0, 1), (138, 1, 1), (149, 0, 0),
    (168, 1, 0), (189, 0, 0), (203, 0, 0), (203, 1, 1), (204, 1, 0),
    (223, 0, 1), (238, 0, 1), (263, 0, 0), (267, 0, 1), (300, 1, 1),
    (307, 1, 0), (311, 0, 1), (354, 1, 1), (357, 1, 1), (391, 0, 0),
    (395, 1, 1), (405, 1, 1), (410, 0, 0), (427, 1, 0), (547, 1, 1),
    (584, 1, 1), (596, 0, 0), (638, 1, 1), (652, 1, 0), (682, 0, 1),
    (698, 1, 0), (698, 0, 1), (737, 0, 0), (738, 1, 0), (765, 0, 1),
]


def main():
    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()

    print(f"[load] model = {MODEL}")
    mm = ModelManager(MODEL)

    picks = [("A", p) for p in CURATED_A] + [("B", p) for p in CURATED_B]
    print(f"[run] {len(picks)} examples × {len(mc.ORDERS)} orderings, "
          f"{mm.num_layers} layers each")

    man = {"examples": []}   # fresh manifest for the curated set
    t0 = time.time()
    for n, (scen, (g, i, q)) in enumerate(picks, 1):
        rec = mc.build_example_record(MODEL, g, i, q)
        print(f"\n[{n}/{len(picks)}] {rec['code']}  scenario={rec['scenario']}  "
              f"Q={rec['question'][:60]!r}")
        if mc.is_example_complete(rec["code"], mm.num_layers):
            rec = mc.attach_existing(rec, mm.num_layers, ts=time.time())
            print("     skip (already rendered) ->", rec["dirs"])
        else:
            rec = mc.process_example(
                mm, rec, resolution=640,
                progress_cb=lambda msg: print("   ", msg),
                ts=time.time(),
            )
            print(f"     saved -> {rec['dirs']}")
            print(f"     generated answers: {rec['answers_generated']}")
        man = mc.add_example(man, rec)

    mc.save_manifest(man)
    dt = time.time() - t0
    print(f"\n[done] {len(man['examples'])} examples in manifest "
          f"({dt/60:.1f} min). Manifest: {mc.MANIFEST}")


if __name__ == "__main__":
    main()
