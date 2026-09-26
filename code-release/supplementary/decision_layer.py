"""
Decision-layer analysis for STI / STIT / IST.

Question: *when*, across depth, does the model commit to its yes/no answer — and
does STI commit too early to an image-anchored (often wrong) answer before the
question is integrated?

For each yes/no (image, question) pair under each ordering we run one forward
pass and read, at the answer position, the logit-lens distribution restricted to
the two candidate tokens {Yes, No} at every layer. From the per-layer
two-candidate probability P(correct | {correct, wrong}) we derive:

  • p_corr2[layer]   — mean P(correct among the two) by layer (the emergence curve)
  • commit layer     — earliest layer from which the argmax stays equal to the
                       FINAL answer (when the decision locks in, right or wrong)
  • correct-onset    — earliest layer from which the correct token stays top-1
                       (only defined when the final answer is correct)
  • flip class       — committed-correct / late-correct / committed-wrong /
                       flipped-to-wrong (correct mid-stack, wrong at the end)

Hypothesis: on STI-wrong/IST-right pairs, STI's p_corr2 stays < 0.5 with an
EARLY commit to the wrong token, while IST/STIT cross above 0.5 — the correct
answer wins only once a question representation sits next to the answer.

Run:
    CUDA_VISIBLE_DEVICES=0 python decision_layer.py --set neutral --num-pairs 250
    CUDA_VISIBLE_DEVICES=0 python decision_layer.py --set disagreement --num-pairs 300
Outputs: naturalbench/probe/decision_{set}.json  (+ decision_{set}.png)
"""

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from PIL import Image

from model_manager import ModelManager
from utils import setup_seeds, disable_torch_init
from transformers.utils import logging as hf_logging
from constants import SYSTEM_MESSAGE
from naturalbench_eval import answer_suffix, judge_pair
from mechanism_probe import disagreement_pairs, neutral_pairs, PROBE_DIR
# Decision probe stays on the stock orderings (no reversed-image variant).
ORDERS = ("STI", "SIT", "STIT", "SITIT")


@torch.no_grad()
def decision_pair(mm, img, question, gt, order, yes_id, no_id):
    """Per-layer two-candidate trajectory for one pair under one ordering."""
    qtext = question + answer_suffix("yes_no")
    _, input_ids, kwargs = mm.prepare_inputs_from_pil(
        [qtext], img, system_prompt=SYSTEM_MESSAGE, order=order)
    out = mm.llm_model(input_ids, output_hidden_states=True, **kwargs)
    hs = out.hidden_states                       # tuple: embed + n_layers

    gt_yes = gt.strip().lower() == "yes"
    corr_id, wrong_id = (yes_id, no_id) if gt_yes else (no_id, yes_id)
    W = mm.llm_model.lm_head.weight
    wc = W[corr_id].float()
    ww = W[wrong_id].float()

    p_corr2, top_corr = [], []
    for l in range(1, len(hs)):                   # skip embedding layer
        h = hs[l][0, -1].float()
        lc, lw = float(h @ wc), float(h @ ww)
        m = max(lc, lw)
        ec, ew = math.exp(lc - m), math.exp(lw - m)
        p_corr2.append(ec / (ec + ew))
        top_corr.append(lc > lw)

    final_correct = top_corr[-1]
    n = len(top_corr)

    # commit layer: earliest L from which argmax stays == final answer
    commit = n - 1
    for L in range(n - 1, -1, -1):
        if top_corr[L] == final_correct:
            commit = L
        else:
            break
    # correct-onset: earliest L from which correct is top-1 to the end
    onset = None
    if final_correct:
        onset = commit                            # same as commit when final-correct

    # flip class
    mid = n // 2
    if final_correct:
        cls = "committed_correct" if commit <= mid else "late_correct"
    else:
        was_corr_mid = any(top_corr[:mid + 1])
        cls = "flipped_to_wrong" if was_corr_mid else "committed_wrong"

    return {"p_corr2": p_corr2, "final_correct": final_correct,
            "commit": commit, "onset": onset, "cls": cls, "n_layers": n}


def _make_fig(summary, setname):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"STI": "#d62728", "SIT": "#1f77b4", "STIT": "#2ca02c",
              "SITIT": "#9467bd", "SITIT_rev": "#8c564b"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    for o in ORDERS:
        y = summary["orders"][o]["p_corr2"]
        ax.plot(range(len(y)), y, color=colors[o], lw=2,
                label=f"{o} (acc {summary['orders'][o]['acc']:.2f})")
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6)
    ax.set_title("P(correct | {Yes,No}) by layer", fontsize=11)
    ax.set_xlabel("layer"); ax.set_ylabel("P(correct among the two)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    for o in ORDERS:
        cl = summary["orders"][o]["commit_layers"]
        if cl:
            ax.hist(cl, bins=18, histtype="step", lw=2, color=colors[o],
                    label=f"{o} (median {np.median(cl):.0f})")
    ax.set_title("Commitment layer (decision locks in)", fontsize=11)
    ax.set_xlabel("layer"); ax.set_ylabel("# pairs")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"Decision-layer analysis — {setname} set (n={summary['n_pairs']})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(PROBE_DIR, f"decision_{setname}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=250, dest="num_pairs")
    ap.add_argument("--set", default="neutral", dest="pset",
                    choices=["disagreement", "neutral"])
    ap.add_argument("--img-cap", type=int, default=640, dest="img_cap")
    args = ap.parse_args()

    hf_logging.set_verbosity_error()
    setup_seeds()
    disable_torch_init()
    os.makedirs(PROBE_DIR, exist_ok=True)

    pairs = (neutral_pairs(args.num_pairs) if args.pset == "neutral"
             else disagreement_pairs(args.num_pairs))
    print(f"[data] {len(pairs)} {args.pset} yes/no pairs")
    print("[load] qwen3-vl-8b")
    mm = ModelManager("qwen3-vl-8b")
    yes_id = mm.tokenizer("Yes", add_special_tokens=False).input_ids[0]
    no_id = mm.tokenizer("No", add_special_tokens=False).input_ids[0]

    agg = {o: {"p_corr2": [], "commit": [], "onset": [], "cls": [], "correct": []}
           for o in ORDERS}
    t0 = time.time()
    n_ok = 0
    for n, (g, i, q, imgrel, question, gt) in enumerate(pairs, 1):
        img = Image.open(os.path.join("naturalbench", imgrel)).convert("RGB")
        img.thumbnail((args.img_cap, args.img_cap), Image.LANCZOS)
        try:
            res = {o: decision_pair(mm, img, question, gt, o, yes_id, no_id)
                   for o in ORDERS}
        except Exception as e:
            print(f"  [{n}] err: {type(e).__name__} {str(e)[:80]}")
            continue
        n_ok += 1
        for o in ORDERS:
            r = res[o]
            agg[o]["p_corr2"].append(r["p_corr2"])
            agg[o]["commit"].append(r["commit"])
            if r["onset"] is not None:
                agg[o]["onset"].append(r["onset"])
            agg[o]["cls"].append(r["cls"])
            agg[o]["correct"].append(r["final_correct"])
        if n % 10 == 0:
            print(f"  [{n}/{len(pairs)}] ok={n_ok} ({n/(time.time()-t0):.2f}/s)")

    summary = {"n_pairs": n_ok, "pset": args.pset, "orders": {}}
    for o in ORDERS:
        d = agg[o]
        classes = d["cls"]
        cls_frac = {c: round(classes.count(c) / max(1, len(classes)), 4)
                    for c in ("committed_correct", "late_correct",
                              "committed_wrong", "flipped_to_wrong")}
        summary["orders"][o] = {
            "p_corr2": np.mean(d["p_corr2"], 0).tolist() if d["p_corr2"] else [],
            "acc": float(np.mean(d["correct"])) if d["correct"] else 0.0,
            "commit_layers": d["commit"],
            "commit_median": float(np.median(d["commit"])) if d["commit"] else None,
            "commit_mean": float(np.mean(d["commit"])) if d["commit"] else None,
            "onset_median": float(np.median(d["onset"])) if d["onset"] else None,
            "cls_frac": cls_frac,
        }
    json.dump(summary, open(os.path.join(PROBE_DIR, f"decision_{args.pset}.json"), "w"),
              indent=2)
    _make_fig(summary, args.pset)

    print(f"\n[done] {n_ok} pairs ({(time.time()-t0)/60:.1f} min)")
    for o in ORDERS:
        s = summary["orders"][o]
        print(f"  {o}: acc {s['acc']:.2f} · commit median L{s['commit_median']} · "
              f"final P(corr2) {s['p_corr2'][-1]:.2f} · "
              f"committed_wrong {s['cls_frac']['committed_wrong']:.2f}")


if __name__ == "__main__":
    main()
