# Position, Not Content: What Repairs Question-First Prompting in Vision-Language Models

Supplementary code for an anonymous ICLR submission. Paper under double-blind
review.

This release is **code only**. It contains every script that produced the
paper's tables, figures and mechanistic analyses, and nothing else: no result
files, no model weights, no datasets, no figure assets. Each script writes its
own result JSON on the first run, and the interactive browser reads those files
once they exist, so the numbers in the paper are reproduced by running the code
rather than by reading a shipped artifact.

Absolute paths have been replaced by repository-relative placeholders
(`datasets/`, `hf_cache/`, `third_party/`). Point them at your own copies, or
set the matching environment variable, before running anything.

---

## Prompt-ordering notation

A prompt is three sections, written in token order:

| Letter | Section |
|--------|---------|
| **S** | System message |
| **I** | Image, which expands to many visual tokens |
| **T** | Task, the question text |

| Ordering | Sequence | Role in the paper |
|----------|----------|-------------------|
| `IST` | Image · System · Task | image first |
| `SIT` | System · Image · Task | question-last, the baseline |
| `STI` | System · Task · Image | question-first, the paradox |
| `STIT` | System · Task · Image · Task | question echoing |
| `SITIT` | System · Image · Task · Image · Task | image echoing |
| `SITIT_rev` | System · Image · Task · Ī · Task | image echo, second copy reversed |
| `SITIT_echo2half` | as `SITIT`, second image at half resolution | the Echo ½ column |

`SITIT_rev` comes from `--reverse`; the echo-resolution variants come from
`--echo-scale 0.5 --echo-which second`, which tags the run `SITIT_echo2half`.

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` lists the packages the analysis and figure code needs. The
benchmark sweeps additionally need `vllm` and `qwen-vl-utils`, the GEPA baseline
needs `gepa` and an OpenAI-compatible client for the reflection server, and
running LLaVA-1.5 needs the `llava` package from its own repository. These are
grouped at the end of `requirements.txt`.

## Models

Downloaded from Hugging Face on first use. Qwen3-VL-8B is the primary model;
Qwen2.5-VL-7B, InternVL3-8B, LLaVA-1.5-7B, Gemma-3-27B, Gemma-3-12B, Gemma-3-4B
and Gemma-4-31B appear in the cross-model tables. Gemma-3-27B must be loaded
4-bit on a single GPU. Set `HF_HOME` to wherever you want the cache, which the
scripts refer to as `hf_cache`.

## Datasets

Download separately and point the script constants at them. `download_naturalbench.py`
and `download_vqa.py` fetch two of them. The rest are the public releases of
NaturalBench, POPE, Winoground, RF20-VL, VQAv2, BLINK, CV-Bench, HR-Bench,
MMStar, MMVP, RealWorldQA, WorldMedQA-V, TallyQA, MVBench, NExT-QA, MSVD-QA and
TGIF-QA. The RF20-VL detection experiments also need the DetPO instruction files
under `third_party/DetPO/data_instr/default/`.

---

## Layout

```
.
├── logit_lens_app.py        Streamlit entry point for the interactive browser
├── *_browser.py             one read-only viewer page each
│
├── model_manager.py         order-aware input builder, shared by every model
├── constants.py, utils.py   system prompt, seeding, helpers
├── reverse_image_hooks.py   second-image reversal hooks for SITIT_rev
│
├── *_eval.py                benchmark runners
├── *_sitit_reverse.py       SITIT_rev runners, one per benchmark
├── extra_tasks/             the 13 additional benchmarks, vLLM harnesses
├── detpo_map/               RF20-VL detection under each ordering, plus PROMPTS.md
├── refcoco_gaze/            exploratory Grad-CAM view, not used in the paper
│
├── mechanism_probe.py       read-out probe, answer to question and image attention
├── decision_layer.py        {Yes,No} decision layer and P(correct) by depth
├── causal_knockout.py       attention-knockout double dissociation
├── modify_attention.py      attention-edge severing utility
├── logit_lens_overlay.py    logit-lens rendering, vision heatmap and token grid
├── scratch_patch_cosine*.py per-patch cosine to an image-only forward pass
│
├── fig2_*.py, fig3_*.py     the systematic search behind Figures 2 and 3
└── make_*.py, *_gen.py      figure generators
```

## Code to paper map

| Paper element | Scripts |
|---------------|---------|
| Ordering ladder on NaturalBench, the paradox | `naturalbench_eval.py`, `gemma_eval.py` |
| POPE and Winoground ladders | `pope_eval.py`, `winoground_eval.py` |
| The 13 further benchmarks | `extra_tasks/{blink,cvbench,hrbench,mmstar,mmvp,realworldqa,worldmedqa,tallyqa,vqa}_eval_vllm.py` |
| The four video QA datasets | `extra_tasks/{mvbench,nextqa}_eval_vllm.py`, `extra_tasks/video_qa_eval_vllm.py` (MSVD-QA, TGIF-QA) |
| RF20-VL detection, where the sign flips | `rf20_map_eval.py`, `detpo_map/ordering_eval_vllm.py`, `detpo_map/ordering_eval.py`, `detpo_map/rf20_aerial_eval.py`, `detpo_map/PROMPTS.md` |
| RF20 yes/no presence variant (appendix) | `rf20_eval.py` |
| Perception probe, the question rewrites the image | `logit_lens_overlay.py`, `logit_lens_runner.py`, `make_steering_fig.py` |
| Finding the rewrite without cherry-picking | `fig2_search.py`, `fig2_cands.py`, `fig2_heat.py`, `fig3_scan.py`, `fig3_probe.py`, `fig3_anchors.py`, and the other `fig3_*.py` search stages |
| Per-patch cosine to an image-only pass (appendix) | `scratch_patch_cosine.py`, `scratch_patch_cosine_sit.py` |
| Read-out probe and decision layer | `mechanism_probe.py`, `decision_layer.py`, `make_probe_figs.py` |
| Causal attention knockout | `causal_knockout.py`, `modify_attention.py` |
| Image-echo reversal, `SITIT_rev` | `reverse_image_hooks.py`, `naturalbench_sitit_reverse.py`, `pope_sitit_reverse.py`, `winoground_sitit_reverse.py`, `rf20_sitit_reverse.py` |
| Exact token cost per ordering (appendix) | `token_cost_analysis.py`, `make_tokencount_fig.py` |
| Gap against question-to-answer distance | `naturalbench_tokensweep.py` |
| GEPA prompt-optimization baseline (appendix) | `gepa_baseline.py`, `reflection_server.py` |
| Logit-lens word divergence, `STI` against `IST` (appendix) | `logit_lens_word_diff.py` |
| McNemar tests and paired bootstrap CIs | `make_significance.py` |
| Interactive viewer for all of the above | `logit_lens_app.py` with the `*_browser.py` pages |

---

## Reproducing a run

Each runner takes an ordering and writes one result file per run.

```bash
# NaturalBench, Qwen3-VL-8B, the four orderings of the main ladder
python naturalbench_eval.py --order SIT     # question-last, the baseline
python naturalbench_eval.py --order STI     # question-first
python naturalbench_eval.py --order STIT    # question echoing
python naturalbench_eval.py --order SITIT   # image echoing

# image echo with the second copy reversed
python naturalbench_sitit_reverse.py --model qwen3-vl-8b --order SITIT --reverse

# one of the 13, all four orderings in a single sweep
HF_HOME=hf_cache CUDA_VISIBLE_DEVICES=0,1 \
  python extra_tasks/mmstar_eval_vllm.py --orders STI,SIT,STIT,SITIT --tp 2

# the Echo 1/2 column: second image copy at half resolution
HF_HOME=hf_cache CUDA_VISIBLE_DEVICES=0,1 \
  python extra_tasks/mmstar_eval_vllm.py --orders SITIT \
    --echo-scale 0.5 --echo-which second --tp 2

# RF20-VL detection mAP under each ordering
HF_HOME=hf_cache CUDA_VISIBLE_DEVICES=0,1 \
  python detpo_map/ordering_eval_vllm.py --orders STI,SIT,STIT,SITIT --tp 2

# mechanism: read-out probe, decision layer, causal knockout
CUDA_VISIBLE_DEVICES=0 python mechanism_probe.py --num-pairs 120
CUDA_VISIBLE_DEVICES=0 python decision_layer.py
CUDA_VISIBLE_DEVICES=0 python causal_knockout.py --num-pairs 200

# significance, offline, no GPU, once the ladders exist
python make_significance.py
```

Result files land in `<dataset>/results/` as `<model>__<ordering>__results.json`
for the core benchmarks, and in `extra_tasks/results/` as
`<task>_order-<ordering>_<model>.json` for the rest. Every accuracy in the paper
is read from the `meta` block of one of these files, which records the model, the
ordering, the system prompt, the decoding budget and the sample count.

## Interactive browser

```bash
streamlit run logit_lens_app.py
```

Pages cover the cross-dataset summary, the ordering ladders per benchmark, the
per-patch perturbation analysis, the logit lens on a single image, the mechanism
probe and the decision layer. Every page is read-only and renders whatever result
files exist, so a fresh checkout shows empty pages with the command needed to
fill each one. The pages that animate GIFs need the artifacts regenerated locally
with the `*_gen.py` scripts.

## Reproduction notes

Two details decide whether the mechanism results reproduce.

The attention knockout must restrict the query side to the answer position.
Widened to every position after the question, the question-side half of the
double dissociation does not appear, and the random-span control is itself worth
about 10 points on `STI`, which makes that configuration uninformative rather
than contradictory.

The Gemma-3-27B RF20 run is degenerate and the paper excludes it from every RF20
claim. It answers yes to roughly 96% of items under all orderings, so its
accuracy sits at the positive base rate and the spread across orderings is 0.003.
Re-running it reproduces the degeneracy, not a usable ladder.
