#!/usr/bin/env python3
"""Plot the token-count scaling law from naturalbench/tokensweep.json ->
paper/figs/tokencount.png. Left: STI and SIT group accuracy vs vision tokens.
Right: the STI-SIT gap vs vision tokens (the predictive-theory panel)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os
# prefer the clean large-image sweep (no upscaling confound) when present
src = "naturalbench/tokensweep_big.json" if os.path.exists("naturalbench/tokensweep_big.json") \
      else "naturalbench/tokensweep.json"
d = json.load(open(src))
rows = sorted(d["rows"], key=lambda r: r["tokens"])
tok = [r["tokens"] for r in rows]
sti = [r["STI"] for r in rows]
sit = [r["SIT"] for r in rows]
gap = [r["gap"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
ax1.plot(tok, sit, "o-", color="#1f77b4", lw=2, label="SIT (question-last)")
ax1.plot(tok, sti, "s-", color="#d62728", lw=2, label="STI (question-first)")
ax1.set_xlabel("vision tokens (question$\\to$answer distance)")
ax1.set_ylabel("NaturalBench group acc")
ax1.legend(fontsize=9); ax1.grid(alpha=0.3)

ax2.plot(tok, gap, "D-", color="#2ca02c", lw=2)
ax2.set_xlabel("vision tokens (question$\\to$answer distance)")
ax2.set_ylabel("SIT $-$ STI gap")
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("paper/figs/tokencount.png", dpi=150, bbox_inches="tight")
print("saved paper/figs/tokencount.png")
print("tokens:", tok)
print("gap:   ", [round(g, 3) for g in gap])
