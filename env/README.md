# Environments

Exported 2026-10-07. Three conda environments run this repository; they are not
interchangeable, because the vLLM sweeps and the logit-lens analysis need
different torch and vLLM builds.

| Environment | Runs | Python | torch | transformers | vLLM |
|---|---|---|---|---|---|
| `logitlens` | **default.** Logit-lens analysis, knockouts, probes, every figure script (`fig2_make.py`, `paper/make_paper_figs.py`), the Streamlit browsers, `website/`, `detpo_map/ordering_eval.py` | 3.13.4 | 2.8.0+cu128 | 5.4.0 | 0.11.0 |
| `qwen-vllm-env` | The vLLM benchmark sweeps: `extra_tasks/*_eval_vllm.py`, `detpo_map/*_vllm.py`, `reflection_server.py` | 3.10.19 | 2.10.0+cu126 | — | 0.19.1 |
| `soft-prompt` | `gepa_baseline.py`, `logit_lens_word_diff.py` | 3.13.4 | 2.8.0+cu128 | 5.0.0.dev0 (git pin) | 0.11.0 |

Scripts that hard-code an interpreter path name their environment inline; everything
else runs under `logitlens`.

## Recreate

```bash
conda env create -f env/logitlens/environment.yml       # then qwen-vllm-env, soft-prompt
conda activate logitlens
```

Per environment:

| File | Use |
|---|---|
| `environment.yml` | **start here.** Versions without build strings, so it solves on another machine or architecture. |
| `environment.lock.yml` | exact build strings. Reproduces this machine bit for bit; will not solve on a different platform. |
| `environment.min.yml` | only the packages explicitly asked for, letting the solver choose the rest. Use when `environment.yml` will not solve. |
| `requirements.lock.txt` | raw `pip freeze`, kept as a record. Not installable as-is: conda-installed packages appear as `file:///` build-artifact paths. |
| `requirements.pip.txt` | the pip-installable subset, for a plain venv instead of conda. |

## What was repaired in the export, and why

`conda env export` records what pip reports, and pip reports local installs in
forms that exist only on this machine. Three entries were rewritten so that
`conda env create` succeeds elsewhere:

- `flux==0.0.0.post59+g802fb4713` was dropped from `logitlens` and `soft-prompt`.
  It is an editable install of an unrelated local project, that version exists on
  no index, and nothing in this repository imports it.
- `transformers==5.0.0.dev0` in `soft-prompt` was repinned to its git commit,
  `git+https://github.com/huggingface/transformers@020e713a`. The bare `.dev0`
  version does not resolve.
- `llamafactory` and `segment-anything`, both editable installs from `git+ssh://`
  remotes, are absent from the `environment*.yml` files and are filtered out of
  `requirements.pip.txt`. Neither is imported by this repository, and an `ssh://`
  remote will not clone without the original machine's keys.

The `requirements.lock.txt` files still carry those three entries verbatim,
because their job is to record what was installed rather than to be replayed.

## Machine these were taken on

| | |
|---|---|
| OS | Ubuntu 22.04.5 LTS, kernel 6.8.0-138-generic, x86_64 |
| GPU | 2 x NVIDIA RTX A6000, 48 GB each |
| Driver | 575.57.08 |
| conda | 23.1.0 |

The system CUDA toolkit is 11.7, older than the cu126 and cu128 builds these
environments install. That is fine: the torch wheels carry their own CUDA
runtime, so the NVIDIA driver version is what has to be new enough, not `nvcc`.

Model weights, datasets and the Hugging Face cache are not part of these exports.
See the top-level `README.md` for where each benchmark is expected on disk.
