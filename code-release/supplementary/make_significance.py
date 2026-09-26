#!/usr/bin/env python3
"""
Paired significance for the ordering ladder (offline; no GPU). For each adjacent
rung we align the two orderings by group index and report:
  - group accuracy of each ordering,
  - Delta and a paired bootstrap 95% CI on the difference,
  - McNemar exact test (two-sided) on the discordant pairs.

NaturalBench uses the per-group `group_correct` flag. Extendable to POPE (per
question `correct`) and Winoground (per pair_key) with the same paired machinery.

Usage: python make_significance.py
"""
import json
import numpy as np

try:
    from scipy.stats import binomtest
    def mcnemar_p(b, c):
        return binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue if (b + c) else 1.0
except Exception:  # scipy-free fallback (exact binomial)
    from math import comb
    def mcnemar_p(b, c):
        n = b + c
        if n == 0:
            return 1.0
        k = min(b, c)
        return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def _load(model, bench, tag):
    """Per-example correctness. NaturalBench: per-group `group_correct`.
    Winoground: an example (4 items) is correct iff all four items are correct."""
    d = json.load(open(f"{bench}/results/{model}__{tag}__results.json"))["results"]
    if bench == "winoground":
        by = {}
        for r in d:
            by.setdefault(r["example_id"], []).append(bool(r["correct"]))
        return {e: all(v) for e, v in by.items()}
    return {g["index"]: bool(g["group_correct"]) for g in d}


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "n.s."


def paired(model, bench, tagA, tagB, n_boot=5000, seed=0):
    A, B = _load(model, bench, tagA), _load(model, bench, tagB)
    idx = sorted(set(A) & set(B))
    a = np.array([A[i] for i in idx]); b = np.array([B[i] for i in idx])
    only_a = int((a & ~b).sum()); only_b = int((~a & b).sum())
    p = mcnemar_p(only_a, only_b)
    rng = np.random.default_rng(seed); n = len(idx); diff = b.astype(int) - a.astype(int)
    boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {tagA:>10} {a.mean():.3f}  vs  {tagB:<10} {b.mean():.3f} | "
          f"D={b.mean()-a.mean():+.3f} [{lo:+.3f},{hi:+.3f}] | "
          f"discordant {only_a}/{only_b} | McNemar p={p:.2e} {stars(p)}")


def main():
    rungs = [("STI", "SIT"), ("SIT", "STIT"), ("STIT", "SITIT"), ("SITIT", "SITIT_rev")]
    headline = [("STI", "SITIT_rev"), ("SIT", "SITIT_rev"), ("SIT", "SITIT")]
    for bench in ("naturalbench", "winoground"):
        for model in ("qwen3-vl-8b", "gemma-3-27b"):
            print(f"{model} {bench} group accuracy (paired):")
            for A, B in rungs + (headline if bench == "winoground" else []):
                try:
                    paired(model, bench, A, B)
                except FileNotFoundError:
                    print(f"  {A} vs {B}: missing results")


if __name__ == "__main__":
    main()
