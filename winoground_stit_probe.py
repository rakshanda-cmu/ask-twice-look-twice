"""
Mechanism Probe on WINOGROUND (Qwen3-VL-8B) — whole dataset (all 1600 image-caption
pairs), for the STIT ordering (and STI / IST for contrast). For every pair we run one
eager forward and measure, per layer:
  - a_q       : attention mass the answer position puts on the QUESTION (caption) span
  - a_img     : attention mass the answer position puts on the IMAGE span
  - p_correct : logit-lens P(correct Yes/No token) from the pre-answer hidden state
then average over the entire dataset and plot the three curves per ordering.

This is the Winoground counterpart of mechanism_probe.py (NaturalBench). It reuses
mechanism_probe.probe_pair unchanged; Winoground is already a yes/no task
(caption-matches-image), so no new probe logic is needed.

Output (local; naturalbench/ is gitignored):
  naturalbench/probe/winoground_stit_probe.json   aggregated per-layer curves + acc
  naturalbench/probe/winoground_stit_probe.png    3-panel figure

Run (one GPU):  CUDA_VISIBLE_DEVICES=0 python winoground_stit_probe.py
"""

import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_manager import ModelManager
from mechanism_probe import probe_pair
from winoground_eval import load_winoground_samples, _question

OUT_DIR = "naturalbench/probe"
ORDERS = ["STI", "STIT", "IST"]
OCOLOR = {"STI": "#f58518", "STIT": "#54a24b", "IST": "#4c78a8"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-examples", type=int, default=None, dest="max_examples",
                    help="Winoground examples (default: all 400 → 1600 pairs).")
    ap.add_argument("--thumb", type=int, default=448,
                    help="Resize images to <= this longest side (bounds seq/mem).")
    ap.add_argument("--orders", default=",".join(ORDERS),
                    help="Comma-separated orderings to probe.")
    args = ap.parse_args()

    from utils import setup_seeds, disable_torch_init
    setup_seeds(); disable_torch_init()
    orders = [o.strip().upper() for o in args.orders.split(",") if o.strip()]
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[load] qwen3-vl-8b (eager attention)", flush=True)
    mm = ModelManager("qwen3-vl-8b", attn_implementation="eager")

    samples = load_winoground_samples(max_examples=args.max_examples)
    print(f"[data] {len(samples)} Winoground pairs · orders {orders}", flush=True)

    def thumb(pil):
        w, h = pil.size
        s = args.thumb / max(w, h)
        return pil.resize((max(1, int(w * s)), max(1, int(h * s)))) if s < 1 else pil

    agg = {o: {"a_q": [], "a_img": [], "p_gt": [], "correct": []} for o in orders}
    for n, s in enumerate(samples, 1):
        img = thumb(s["image"].convert("RGB"))
        question = _question(s["caption"])
        for o in orders:
            r = probe_pair(mm, img, question, s["gt"], o)
            if r is None:
                continue
            agg[o]["a_q"].append(r["a_q"])
            agg[o]["a_img"].append(r["a_img"])
            agg[o]["p_gt"].append(r["p_gt"])
            agg[o]["correct"].append(r["correct"])
        if n % 100 == 0 or n == len(samples):
            acc = {o: (np.mean(agg[o]["correct"]) if agg[o]["correct"] else 0) for o in orders}
            print(f"  [{n}/{len(samples)}] acc " +
                  " ".join(f"{o}={acc[o]:.3f}" for o in orders), flush=True)

    result = {"n_pairs": len(samples), "orders": {}}
    for o in orders:
        d = agg[o]
        result["orders"][o] = {
            "acc": float(np.mean(d["correct"])) if d["correct"] else 0.0,
            "a_q": np.mean(d["a_q"], 0).tolist() if d["a_q"] else [],
            "a_img": np.mean(d["a_img"], 0).tolist() if d["a_img"] else [],
            "p_gt": np.mean(d["p_gt"], 0).tolist() if d["p_gt"] else [],
        }
    json.dump(result, open(f"{OUT_DIR}/winoground_stit_probe.json", "w"), indent=2)

    # ── figure: 3 panels (answer→question, answer→image, p_correct) ────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    panels = [("a_q", "answer → question (caption) attention"),
              ("a_img", "answer → image attention"),
              ("p_gt", "logit-lens P(correct Yes/No)")]
    for j, (key, title) in enumerate(panels):
        for o in orders:
            y = result["orders"][o][key]
            if not y:
                continue
            lw = 3.0 if o == "STIT" else 1.6
            ax[j].plot(range(len(y)), y, label=f"{o} (acc {result['orders'][o]['acc']*100:.1f}%)",
                       color=OCOLOR.get(o), linewidth=lw)
        ax[j].set_title(title, fontsize=11)
        ax[j].set_xlabel("layer")
        ax[j].grid(alpha=0.25)
        ax[j].legend(fontsize=8)
    fig.suptitle(f"Winoground Mechanism Probe — Qwen3-VL-8B, whole dataset "
                 f"(n={len(samples)} pairs)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{OUT_DIR}/winoground_stit_probe.png", dpi=130)
    print(f"[saved] {OUT_DIR}/winoground_stit_probe.png", flush=True)
    for o in orders:
        oo = result["orders"][o]
        aq = np.array(oo["a_q"]); pg = np.array(oo["p_gt"])
        print(f"  {o}: acc={oo['acc']*100:.1f}%  mid a_q peak={aq[16:26].max():.3f}  "
              f"final P(correct)={pg[-1]:.3f}", flush=True)


if __name__ == "__main__":
    main()
