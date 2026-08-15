"""Emit the supplementary paired-significance table (LaTeX body) for the paper.
Offline, no GPU. For each comparison it aligns the two orderings by example and
reports Delta, a paired bootstrap 95% CI, and a two-sided exact McNemar p.

Correctness unit per benchmark:
  naturalbench -> per-group `group_correct`
  pope         -> per-question `correct`
  winoground   -> per-example all-four-correct
"""
import json
import numpy as np

try:
    from scipy.stats import binomtest
    def mcnemar_p(b, c):
        return binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue if (b + c) else 1.0
except Exception:
    from math import comb
    def mcnemar_p(b, c):
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _load(model, bench, tag):
    d = json.load(open(f"{bench}/results/{model}__{tag}__results.json"))["results"]
    if bench == "winoground":
        by = {}
        for r in d:
            by.setdefault(r["example_id"], []).append(bool(r["correct"]))
        return {e: all(v) for e, v in by.items()}
    if bench == "pope":
        # question_id repeats across the 3 POPE categories; key on both.
        return {(r["category"], r["question_id"]): bool(r["correct"]) for r in d}
    return {g["index"]: bool(g["group_correct"]) for g in d}


def pstr(p):
    if p < 1e-4:
        return "$<$$10^{-4}$"
    return f"${p:.3f}$".replace("$0.", "$.")


def paired(model, bench, tagA, tagB, n_boot=5000, seed=0):
    A, B = _load(model, bench, tagA), _load(model, bench, tagB)
    idx = sorted(set(A) & set(B))
    a = np.array([A[i] for i in idx]); b = np.array([B[i] for i in idx])
    ba, bb = int((a & ~b).sum()), int((~a & b).sum())
    p = mcnemar_p(ba, bb)
    rng = np.random.default_rng(seed); n = len(idx); diff = b.astype(int) - a.astype(int)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return b.mean() - a.mean(), lo, hi, p


MODELS = [("qwen3-vl-8b", "Qwen3-VL-8B"), ("qwen2.5-vl-7b", "Qwen2.5-VL-7B"),
          ("internvl3-8b", "InternVL3-8B"), ("llava-1.5", "LLaVA-1.5-7B"),
          ("gemma-3-27b", "Gemma-3-27B")]
BENCHES = [("naturalbench", "NaturalBench"), ("pope", "POPE"), ("winoground", "Winoground")]


def row(name, model, bench, A, B):
    try:
        d, lo, hi, p = paired(model, bench, A, B)
        return f"    {name} & ${d:+.3f}$ & $[{lo:+.3f},{hi:+.3f}]$ & {pstr(p)}\\\\"
    except (FileNotFoundError, KeyError):
        return f"    {name} & -- & -- & --\\\\"


print("% ---- paradox gap (STI -> SIT), all models x benchmarks ----")
for bkey, bname in BENCHES:
    print(f"    \\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{bname}}}}}\\\\")
    for mkey, mname in MODELS:
        print(row(mname, mkey, bkey, "STI", "SIT"))
print("% ---- ladder steps on the primary models (NaturalBench) ----")
for mkey, mname in [("qwen3-vl-8b", "Qwen3-VL-8B"), ("gemma-3-27b", "Gemma-3-27B")]:
    for A, B in [("SIT", "STIT"), ("STIT", "SITIT")]:
        print(row(f"{mname}: {A}$\\to${B}", mkey, "naturalbench", A, B))
