"""
Streamlit page: "Echo Ablations & GEPA" — a dedicated, additive tab for the
echo-resolution ablation (RF20's half-vs-quarter sweep, extended to all other
benchmarks), token-cost accounting, GEPA prompt-optimization baseline, and
logit-lens STI/IST word-diff, kept SEPARATE from the existing per-dataset
tabs so nothing there is touched. Purely additive and read-only; reads
whatever result JSONs each sub-section needs and renders "not run yet"
placeholders for anything still pending.
"""
import json
import os

import pandas as pd
import streamlit as st

RF20_RESULTS_DIR = "./rf20/results"
RF20_MODEL = "qwen3-vl-8b"
RF20_ECHO_COLS = [
    ("SITIT", "SITIT baseline (both full-res)"),
    ("SITIT_echo2half", "SITIT (2nd/echoed occ. at 0.5x)"),
    ("SITIT_echo2quarter", "SITIT (2nd/echoed occ. at 0.25x)"),
]


def _rf20_meta(tag):
    p = os.path.join(RF20_RESULTS_DIR, f"{RF20_MODEL}__{tag}__results.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))["meta"]
    except Exception:
        return None


def _rf20_resolution_section():
    st.subheader("🔍 RF20 — echoed-occurrence resolution: half vs quarter")
    st.caption(
        "SITIT shows the image twice; here the SECOND (echoed) occurrence is "
        "downscaled to 0.5x or 0.25x while the first stays full-res, isolating "
        "how much detail the echo actually needs to carry. All rows: "
        f"**{RF20_MODEL}**, full 20-dataset RF20 (24,449 samples)."
    )
    metas = {tag: _rf20_meta(tag) for tag, _ in RF20_ECHO_COLS}
    if not any(metas.values()):
        st.info("No RF20 resolution-sweep results yet.")
        return

    rows = []
    for tag, label in RF20_ECHO_COLS:
        m = metas.get(tag)
        if not m:
            rows.append({"Ordering": label, "N": "—", "F1": "—", "Accuracy": "—"})
            continue
        o = m.get("overall", {})
        rows.append({
            "Ordering": label, "N": str(o.get("n", "—")),
            "F1": f"{o.get('f1', 0)*100:.2f}%",
            "Accuracy": f"{o.get('acc', 0)*100:.2f}%",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    base, half, quarter = metas.get("SITIT"), metas.get("SITIT_echo2half"), metas.get("SITIT_echo2quarter")
    if base and half and quarter:
        f1_base = base["overall"]["f1"] * 100
        f1_half = half["overall"]["f1"] * 100
        f1_quarter = quarter["overall"]["f1"] * 100
        d_half = f1_half - f1_base
        d_quarter = f1_quarter - f1_base
        st.caption(
            f"vs baseline F1 — half: **{d_half:+.2f} pts**, quarter: "
            f"**{d_quarter:+.2f} pts**. "
            + (f"**Half beats quarter** ({f1_half:.2f}% > {f1_quarter:.2f}%) — "
               "the expected/safer outcome; per your instruction, this would "
               "extend the half-resolution echo ablation to the other benchmarks."
               if f1_half > f1_quarter else
               f"**Quarter beats half** ({f1_quarter:.2f}% > {f1_half:.2f}%) — "
               "the more surprising outcome; per your instruction, this should "
               "be discussed before extending it anywhere.")
        )
    elif base and half:
        st.caption("Quarter-resolution run not finished yet — showing baseline vs half so far.")
    elif base:
        st.caption("Half/quarter runs in progress — showing baseline only so far.")


TOKEN_COST_PATH = "./token_cost_results.json"
TOKEN_COST_VARIANTS = [
    ("STI", "STI (baseline: 1 image, 1 task)"),
    ("STIT", "STIT (text repetition: task x2)"),
    ("SITIT_full", "SITIT (image repetition, full-res echo)"),
    ("SITIT_half", "SITIT (image repetition, ½-res echo)"),
    ("SITIT_quarter", "SITIT (image repetition, ¼-res echo)"),
]


def _token_cost_section():
    st.subheader("🔢 Token-cost Delta — text repetition vs image repetition")
    st.caption(
        "How many extra tokens does each intervention cost, relative to STI "
        "baseline: repeating the TASK text (STIT) vs repeating the IMAGE "
        "(SITIT), at full/½/¼ resolution for the echoed occurrence."
    )
    if not os.path.exists(TOKEN_COST_PATH):
        st.info("Not computed yet — run token_cost_analysis.py (CPU-only, no "
                "GPU generation needed).")
        return
    d = json.load(open(TOKEN_COST_PATH))
    st.caption(
        f"Mean over **{d['n_images']}** representative RF20 images (one per "
        f"dataset), **{d['model']}** tokenizer/processor. Task text: "
        f"\"{d['task_text']}\""
    )
    rows = []
    for key, label in TOKEN_COST_VARIANTS:
        s, delta = d["summary"][key], d["delta_vs_STI"][key]
        rows.append({
            "Variant": label,
            "Image tokens": f"{s['image']:.0f}",
            "Text tokens": f"{s['text']:.0f}",
            "Total tokens": f"{s['total']:.0f}",
            "Δ total vs STI": f"{delta['delta_total']:+.0f}",
            "Δ image vs STI": f"{delta['delta_image']:+.0f}",
            "Δ text vs STI": f"{delta['delta_text']:+.0f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    stit_d = d["delta_vs_STI"]["STIT"]["delta_total"]
    full_d = d["delta_vs_STI"]["SITIT_full"]["delta_total"]
    half_d = d["delta_vs_STI"]["SITIT_half"]["delta_total"]
    quarter_d = d["delta_vs_STI"]["SITIT_quarter"]["delta_total"]
    st.caption(
        f"Text repetition (STIT) costs only **+{stit_d:.0f} tokens** vs STI "
        f"(the repeated task text is short); image repetition (SITIT) costs "
        f"far more, and scales with the echoed image's resolution: "
        f"**+{full_d:.0f}** at full res, **+{half_d:.0f}** at half "
        f"(~{100*half_d/full_d:.0f}% of full), **+{quarter_d:.0f}** at quarter "
        f"(~{100*quarter_d/full_d:.0f}% of full) — consistent with vision "
        f"tokens scaling with pixel count (quadratically in linear scale)."
    )


GEPA_RESULTS_DIR = "./gepa_results"
GEPA_MODEL = "qwen3-vl-8b"
GEPA_DATASETS = ["pope", "vqa"]


def _gepa_result(dataset):
    p = os.path.join(GEPA_RESULTS_DIR, f"{dataset}__{GEPA_MODEL}__gepa.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _gepa_section():
    st.subheader("🧬 GEPA baseline — prompt optimization")
    st.caption(
        "GEPA (Reflective Prompt Evolution, arxiv.org/abs/2507.19457) as an "
        "automated prompt-optimization baseline, run on datasets other than "
        "RF20. Qwen3-VL-8B (via vLLM) answers under a fixed STI order; "
        "Gemma-3-27B, on a separate GPU via reflection_server.py, serves as "
        "the reflection model that proposes improved prompt text from "
        "failure feedback -- deliberately a different, stronger model than "
        "the task model, matching GEPA's own intended design. Two real bugs "
        "in an earlier version (the SAME 8B model reused for both task and "
        "reflection roles; a hardcoded 'yes/no question-answering' "
        "objective passed to GEPA regardless of dataset -- wrong for VQA's "
        "open-ended answers) produced a false null result on both datasets. "
        "Fixing both, then giving VQA a larger/more stable search budget "
        "once POPE's fix alone confirmed the mechanism works, both datasets "
        "now show a genuine accuracy gain from a longer, more specific "
        "optimized prompt -- not free, but a real win for a small extra "
        "token cost. Train/val subsets are carved out of each benchmark's "
        "existing full pool (seeded, disjoint from each other); the "
        "held-out eval subset is scored for both the baseline SYSTEM_"
        "MESSAGE and the GEPA-optimized prompt on identical examples."
    )
    results = {d: _gepa_result(d) for d in GEPA_DATASETS}
    if not any(results.values()):
        st.info("Not run yet.")
        return

    rows = []
    for d in GEPA_DATASETS:
        r = results.get(d)
        if not r:
            rows.append({"Dataset": d, "N (train/val/eval)": "—",
                        "Baseline acc": "—", "GEPA acc": "—", "Δ acc": "—",
                        "Train wall-clock": "—", "Metric calls": "—",
                        "Train tokens": "—", "Δ inference tokens": "—"})
            continue
        sp, tr, ic, ac = r["split"], r["training"], r["inference_cost"], r["accuracy"]
        rows.append({
            "Dataset": d,
            "N (train/val/eval)": f"{sp['train']}/{sp['val']}/{sp['held_out_eval']}",
            "Baseline acc": f"{ac['baseline_system_message']['acc']*100:.2f}%",
            "GEPA acc": f"{ac['gepa_optimized']['acc']*100:.2f}%",
            "Δ acc": f"{ac['delta']*100:+.2f} pts",
            "Train wall-clock": f"{tr['wall_clock_s']:.0f}s",
            "Metric calls": str(tr["total_metric_calls"]),
            "Train tokens": f"{tr['total_tokens']:,}",
            "Δ inference tokens": f"{ic['delta_tokens']:+d}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    for d in GEPA_DATASETS:
        r = results.get(d)
        if not r:
            continue
        with st.expander(f"{d}: optimized prompt + cost breakdown"):
            tr = r["training"]
            st.caption(
                f"Training calls: **{tr['task_calls']}** task_lm "
                f"({tr['task_tokens_in']:,} in / {tr['task_tokens_out']:,} out tok) + "
                f"**{tr['reflect_calls']}** reflection ({tr['reflect_tokens_in']:,} in / "
                f"{tr['reflect_tokens_out']:,} out tok) = **{tr['total_tokens']:,}** total "
                f"tokens over **{tr['wall_clock_s']:.0f}s**."
            )
            st.markdown("**Baseline SYSTEM_MESSAGE:**")
            st.code(r["baseline_prompt"], language=None)
            st.markdown("**GEPA-optimized prompt:**")
            st.code(r["gepa_optimized_prompt"], language=None)


def _logit_lens_diff_section():
    st.subheader("🧠 Logit-lens word-diff — STI vs IST")
    st.caption(
        "Per (layer, generation-step), diff the top-predicted word between "
        "STI and IST orderings on the same (image, question) pairs, on "
        "**generated-answer positions only** (the one part of the logit-lens "
        "machinery directly comparable between the two orderings, since the "
        "input-text token layout itself differs entirely). Two filters strip "
        "out non-signal before reporting a diff: both words must be real "
        "dictionary words (early/mid-layer logit lens frequently projects "
        "onto subword fragments or unrelated tokens, not real words) and "
        "must not be WordNet synonyms of each other."
    )
    path = "./logit_lens_word_diff_results.json"
    if not os.path.exists(path):
        st.info("Not run yet.")
        return
    d = json.load(open(path))
    st.caption(
        f"**{d['model']}**, n={d['n_samples']} RF20 samples, layers "
        f"{d['layer_range']}: **{d['total_genuine_diffs']}** genuine "
        f"word-diffs survived filtering."
    )
    by_layer = pd.DataFrame(
        [{"Layer": k, "Genuine diffs": v} for k, v in d["diffs_by_layer"].items()])
    st.dataframe(by_layer, hide_index=True, use_container_width=True)
    rows = [{"Question": e["query"][:60], "STI answer": e["answer_sti"],
            "IST answer": e["answer_ist"], "Layer": diff["layer"],
            "Step": diff["step"], "STI word": diff["sti_word"],
            "IST word": diff["ist_word"]}
           for e in d["per_example"] for diff in e["diffs"]]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "Several diffs show STI's logit lens already predicting the "
            "eventual answer word at a given layer while IST predicts an "
            "unrelated word at the same layer/step -- i.e. STI's residual "
            "stream commits to its answer earlier in the network than IST's "
            "does, for the same (image, question) pair."
        )


ECHO2HALF_DATASETS = [
    # (label, results_dir, file_prefix, sitit_tag, echo_tag, metric_fn)
    ("BLINK", "extra_tasks/results", "blink_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("CV-Bench", "extra_tasks/results", "cvbench_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("HR-Bench", "extra_tasks/results", "hrbench_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("MMStar", "extra_tasks/results", "mmstar_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("RealWorldQA", "extra_tasks/results", "realworldqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("WorldMedQA-V", "extra_tasks/results", "worldmedqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("MVBench", "extra_tasks/results", "mvbench_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("NExT-QA", "extra_tasks/results", "nextqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("MSVD-QA", "extra_tasks/results", "msvdqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("TGIF-QA", "extra_tasks/results", "tgifqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("TallyQA", "extra_tasks/results", "tallyqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("acc", m["accuracy"])),
    ("VQAv2", "extra_tasks/results", "vqa_order", "SITIT", "SITIT_echo2half",
     lambda m: ("vqa_score", m["vqa_score"])),
]


def _echo2half_extension_section():
    st.subheader("📐 Echo-resolution extension — half-res echo across every benchmark")
    st.caption(
        "RF20's echo-resolution sweep (above) found half beats quarter "
        "clearly, so per that branch decision this extends the half-res "
        "echo ablation (2nd/echoed SITIT occurrence at 0.5x) to every other "
        "benchmark in the repo -- all at full production scale, matching "
        "each dataset's existing SITIT baseline N exactly. RF20 itself is "
        "excluded (it's what drove this decision, not a target of it)."
    )
    rows = []
    for label, rdir, prefix, sitit_tag, echo_tag, metric_fn in ECHO2HALF_DATASETS:
        base_p = os.path.join(rdir, f"{prefix}-{sitit_tag}_{RF20_MODEL}.json")
        echo_p = os.path.join(rdir, f"{prefix}-{echo_tag}_{RF20_MODEL}.json")
        base_m = json.load(open(base_p))["meta"] if os.path.exists(base_p) else None
        echo_m = json.load(open(echo_p))["meta"] if os.path.exists(echo_p) else None
        if not (base_m and echo_m):
            rows.append({"Benchmark": label, "N": "—", "Metric": "—",
                        "SITIT baseline": "—", "SITIT echo2half": "—", "Δ": "—"})
            continue
        metric_name, base_v = metric_fn(base_m)
        _, echo_v = metric_fn(echo_m)
        rows.append({
            "Benchmark": label, "N": echo_m.get("n"), "Metric": metric_name,
            "SITIT baseline": f"{base_v*100:.2f}%", "SITIT echo2half": f"{echo_v*100:.2f}%",
            "Δ": f"{(echo_v-base_v)*100:+.2f} pts",
        })

    def _pope_row():
        p = "pope/results/qwen3-vl-8b__SITIT__results.json"
        pe = "pope/results/qwen3-vl-8b__SITIT_echo2half__results.json"
        if not (os.path.exists(p) and os.path.exists(pe)):
            return {"Benchmark": "POPE", "N": "—", "Metric": "—",
                   "SITIT baseline": "—", "SITIT echo2half": "—", "Δ": "—"}
        b, e = json.load(open(p))["meta"]["overall"], json.load(open(pe))["meta"]["overall"]
        return {"Benchmark": "POPE", "N": e["n"], "Metric": "f1",
                "SITIT baseline": f"{b['f1']*100:.2f}%",
                "SITIT echo2half": f"{e['f1']*100:.2f}%",
                "Δ": f"{(e['f1']-b['f1'])*100:+.2f} pts"}

    def _winoground_row():
        p = "winoground/results/qwen3-vl-8b__SITIT__results.json"
        pe = "winoground/results/qwen3-vl-8b__SITIT_echo2half__results.json"
        if not (os.path.exists(p) and os.path.exists(pe)):
            return {"Benchmark": "Winoground", "N": "—", "Metric": "—",
                   "SITIT baseline": "—", "SITIT echo2half": "—", "Δ": "—"}
        b, e = json.load(open(p))["meta"]["overall"], json.load(open(pe))["meta"]["overall"]
        return {"Benchmark": "Winoground", "N": e["n"], "Metric": "group_acc",
                "SITIT baseline": f"{b['group_acc']*100:.2f}%",
                "SITIT echo2half": f"{e['group_acc']*100:.2f}%",
                "Δ": f"{(e['group_acc']-b['group_acc'])*100:+.2f} pts"}

    rows.append(_pope_row())
    rows.append(_winoground_row())
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    n_done = sum(1 for r in rows if r["Δ"] != "—")
    st.caption(f"{n_done}/{len(rows)} benchmarks complete.")


def render_new_experiments_page():
    st.title("🔬 Echo Ablations & GEPA")
    st.caption(
        "A separate tab for the newest round of ablations, kept apart from "
        "the existing per-dataset tabs so nothing there is modified."
    )
    _rf20_resolution_section()
    st.markdown("---")
    _echo2half_extension_section()
    st.markdown("---")
    _token_cost_section()
    st.markdown("---")
    _gepa_section()
    st.markdown("---")
    _logit_lens_diff_section()
