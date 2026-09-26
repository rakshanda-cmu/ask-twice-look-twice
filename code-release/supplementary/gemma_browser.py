"""
Streamlit page: Gemma 3 — IST / STI / STIT results and a cross-model comparison
against Qwen3-VL-8B. Purely additive; reads gemma-3-27b__<order>__results.json
written by gemma_eval.py --model gemma-3-27b.
"""

import json
import os

import streamlit as st

RESULTS_DIR = "./naturalbench/results"
GEMMA_MODEL = "gemma-3-27b"
REF_MODEL = "qwen3-vl-8b"
ORDERS = [
    ("IST", "IST — Image · System · Task"),
    ("STI", "STI — System · Task · Image"),
    ("STIT", "STIT — System · Task · Image · Task"),
]
RUN_CMD = ("CUDA_VISIBLE_DEVICES=1 python gemma_eval.py "
           "--model gemma-3-27b --order IST,STI,STIT --num-groups 1900")


def _meta(model, tag):
    p = os.path.join(RESULTS_DIR, f"{model}__{tag}__results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["meta"]
    except Exception:
        return None


def _metric_row(meta):
    return {
        "G-Acc (all 4)": f"{meta.get('g_acc', 0)*100:.1f}%",
        "Q-Acc (per question)": f"{meta.get('q_acc', 0)*100:.1f}%",
        "I-Acc (per image)": f"{meta.get('i_acc', 0)*100:.1f}%",
        "Pair-Acc": f"{meta.get('pair_acc', 0)*100:.1f}%",
        "# groups": str(meta.get("num_groups", 0)),
    }


def _gap_caption(metas, model_label):
    if metas.get("IST") and metas.get("STI"):
        d = (metas["IST"]["g_acc"] - metas["STI"]["g_acc"]) * 100
        better = "IST" if d > 0 else ("STI" if d < 0 else "tie")
        msg = (f"**{model_label}** Group-accuracy gap (IST − STI): **{d:+.1f} pts** "
               f"→ {'**' + better + '** higher' if better != 'tie' else 'tie'}")
        if metas.get("STIT"):
            g = metas["STIT"]["g_acc"] * 100
            parts = []
            if metas.get("STI"):
                parts.append(f"vs STI **{g - metas['STI']['g_acc']*100:+.1f}**")
            if metas.get("IST"):
                parts.append(f"vs IST **{g - metas['IST']['g_acc']*100:+.1f}**")
            msg += f". STIT Group-Acc **{g:.1f}%** (" + " · ".join(parts) + ")"
        st.caption(msg)


def render_gemma_page():
    st.subheader("🔷 Gemma 3 (27B) — IST / STI / STIT")
    st.caption(
        "Does the prompt-ordering effect replicate on a second model? Same "
        "NaturalBench groups and the same S/I/T content layout as the Qwen runs, "
        "so any IST-vs-STI gap (and the STIT recovery) can be compared head-to-"
        "head across architectures."
    )

    gem = {tag: _meta(GEMMA_MODEL, tag) for tag, _ in ORDERS}
    if not any(gem.values()):
        st.info(
            "**No Gemma 3 results yet.** `google/gemma-3-27b-it` is a gated repo — "
            "once access is granted to the HF token, run (resumable, single GPU, "
            f"loaded 4-bit nf4 — multi-GPU sharding gives NaN logits here):\n\n`{RUN_CMD}`"
        )
        return

    # ── Gemma table ────────────────────────────────────────────────────────────
    st.markdown("#### Gemma 3 (27B)")
    rows = {label: _metric_row(gem[tag]) for tag, label in ORDERS if gem.get(tag)}
    st.table(rows)
    _gap_caption(gem, "Gemma 3 (27B)")

    # ── cross-model comparison vs Qwen3-VL-8B ──────────────────────────────────
    ref = {tag: _meta(REF_MODEL, tag) for tag, _ in ORDERS}
    if any(ref.values()):
        st.markdown("#### Cross-model — Group accuracy by ordering")
        comp = {}
        for tag, label in ORDERS:
            short = tag
            comp[short] = {
                "Gemma 3 (27B)": f"{gem[tag]['g_acc']*100:.1f}%" if gem.get(tag) else "—",
                "Qwen3-VL (8B)": f"{ref[tag]['g_acc']*100:.1f}%" if ref.get(tag) else "—",
            }
        st.table(comp)
        # does the IST-vs-STI direction agree across models?
        if all(gem.get(o) for o in ("IST", "STI")) and all(ref.get(o) for o in ("IST", "STI")):
            dg = (gem["IST"]["g_acc"] - gem["STI"]["g_acc"]) * 100
            dr = (ref["IST"]["g_acc"] - ref["STI"]["g_acc"]) * 100
            agree = (dg > 0) == (dr > 0)
            st.caption(
                f"IST − STI gap: Gemma **{dg:+.1f} pts** · Qwen **{dr:+.1f} pts** — "
                + ("**same direction** → the ordering effect replicates across "
                   "architectures." if agree else
                   "**opposite direction** → the effect is model-specific.")
            )
        _gap_caption(ref, "Qwen3-VL (8B)")

    st.markdown("---")
    st.caption(f"Re-run / extend with:  `{RUN_CMD}`")
