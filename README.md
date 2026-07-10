# Ask Twice, Look Twice — Supplementary Code & Results

Supplementary material for the paper *"Ask Twice, Look Twice: Prompt Echoing
Resolves the Question-First Paradox in Vision-Language Models."*

This repository contains the analysis code, the interactive interpretability UI,
and the benchmark result files needed to reproduce the paper's tables, figures,
and mechanistic analyses. Model weights and raw image datasets are **not**
included (they are large and publicly available); see [Setup](#setup).

---

## Prompt-ordering notation

A prompt is composed of three sections, written in token order:

| Letter | Section |
|--------|---------|
| **S**  | System message |
| **I**  | Image (expands to many visual tokens) |
| **T**  | Task / question |

The six orderings studied here (the *core* set distributed in this repo):

| Ordering | Sequence | Role |
|----------|----------|------|
| `IST`  | Image · System · Task | image first |
| `SIT`  | System · Image · Task | **question-last** (baseline) |
| `STI`  | System · Task · Image | **question-first** (the paradox) |
| `STIT` | System · Task · Image · Task | **question echoing** (ours) |
| `SITIT`| System · Image · Task · Image · Task | **image echoing** (ours, best) |
| `SITIT_rev` | System · Image · Task · Ī · Task | image echo with reversed 2nd copy |

Exploratory variants (padding/`_space`, resolution sweep/`_res`, `_copies`,
`StIT`, `STSIT`, `SITI`, `STITI`, `_perm`, …) are intentionally omitted.

---

## Repository layout

```
.
├── logit_lens_app.py          # Streamlit UI entry point (all analysis tabs)
├── *_browser.py               # per-analysis UI pages (read-only viewers)
│
├── model_manager.py           # order-aware input builder, shared across models
├── constants.py, utils.py     # system prompt, seeding, helpers
├── reverse_image_hooks.py     # 2nd-image reversal hooks (SITIT_rev), both models
│
├── *_eval.py                  # benchmark runners (see map below)
├── *_sitit_reverse.py         # SITIT_rev runners per benchmark
│
├── mechanism_probe.py         # per-layer read-out probe (answer→question/image attn)
├── decision_layer.py          # {Yes,No} decision-layer / P(correct) by depth
├── causal_knockout.py         # attention-knockout double dissociation
├── modify_attention.py        # attention-edge severing utility
├── logit_lens_overlay.py      # logit-lens rendering (vision heatmap + token grid)
├── scratch_patch_cosine*.py   # per-patch cosine-to-image-only (steering figure)
│
├── *_gen.py / make_*.py       # figure generators (see map below)
│
├── naturalbench/results/      # result JSONs (core orderings only)
├── pope/results/
├── winoground/results/
├── rf20/results/
└── patch_cosine_*.json        # per-patch cosine data for the steering figure
```

## Code → paper map

| Paper element | Scripts |
|---------------|---------|
| Position ladder / paradox (NaturalBench) | `naturalbench_eval.py` (Qwen), `gemma_eval.py` (Gemma) |
| POPE / Winoground / RF20 | `pope_eval.py`, `winoground_eval.py`, `rf20_eval.py`, `rf20_map_eval.py` |
| SITIT_rev (image-reversal) | `*_sitit_reverse.py` + `reverse_image_hooks.py` |
| Steering (logit-lens patch crops) | `make_steering_fig.py`, `logit_lens_overlay.py`, `logit_lens_runner.py` |
| Steering, measured (per-patch cosine) | `scratch_patch_cosine.py`, `scratch_patch_cosine_sit.py` → `patch_cosine_*.json` |
| Read-out probe (attention + emergence) | `mechanism_probe.py`, `decision_layer.py`, `make_probe_figs.py` |
| Causal attention knockout | `causal_knockout.py`, `modify_attention.py` |
| Gap vs. question–answer distance | `naturalbench_tokensweep.py`, `make_tokencount_fig.py` |
| Significance (McNemar / bootstrap) | `make_significance.py` |
| Open-ended VQA | `vqa_eval.py` |
| Supplementary: is the rewrite localized? | `qq_cosine_gen.py`, `patchcos_overlay_gen.py`, `logitlens_demo_gen.py`, `attn_heatmap_gen.py`, `sitit_stit_gif_gen.py` |
| Interactive viewer for all of the above | `logit_lens_app.py` + `*_browser.py` |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Models** (downloaded from Hugging Face on first use, not shipped here):
Qwen3-VL-8B, Gemma-3-27B (loaded 4-bit, single GPU), Qwen2.5-VL-7B, InternVL3-8B,
LLaVA-1.5-7B. Gemma-3-27B must be loaded on a single GPU in 4-bit.

**Datasets** (download separately into the sibling folders):
NaturalBench, POPE, Winoground, RF20, VQAv2 (see `download_naturalbench.py`,
`download_vqa.py`). Only the per-run **result JSONs** are included here.

## Interactive UI

```bash
streamlit run logit_lens_app.py
```

Tabs cover the cross-dataset summary, the visual-patch perturbation (cosine)
analysis, the SITIT-vs-STIT logit-lens comparison, the per-layer logit lens on a
single image, the mechanism probe, and the decision-layer view. The result-file
viewers work from the included JSONs; the figure tabs that animate GIFs require
regenerating the artifacts locally with the `*_gen.py` scripts.

## Reproducing a benchmark run

```bash
# NaturalBench, Qwen3-VL-8B, each core ordering:
python naturalbench_eval.py --order STI     # question-first
python naturalbench_eval.py --order SIT     # question-last
python naturalbench_eval.py --order STIT    # question echoing
python naturalbench_eval.py --order SITIT   # image echoing

# SITIT with reversed 2nd image:
python naturalbench_sitit_reverse.py --model qwen3-vl-8b --order SITIT --reverse
```

Result files are written as
`<dataset>/results/<model>__<ordering>__results.json`.

---

*Weights and datasets are external; this repository ships code plus the result
JSONs for the six core prompt orderings.*
