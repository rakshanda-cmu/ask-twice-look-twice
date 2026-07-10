"""Render per-layer mechanism-probe figures from probe_{set}.json."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROBE_DIR = "./naturalbench/probe"
COLORS = {"STI": "#d62728", "STIT": "#2ca02c", "IST": "#1f77b4"}


def _fig(setname):
    p = os.path.join(PROBE_DIR, f"probe_{setname}.json")
    if not os.path.exists(p):
        return None
    s = json.load(open(p))
    orders = s["orders"]
    panels = [("a_q", "Answer → QUESTION attention"),
              ("a_img", "Answer → IMAGE attention"),
              ("p_gt", "Answer emergence: P(correct) by layer")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (key, title) in zip(axes, panels):
        for o in ("STI", "STIT", "IST"):
            y = orders[o][key]
            ax.plot(range(len(y)), y, label=f"{o} (acc {orders[o]['acc']:.2f})",
                    color=COLORS[o], lw=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("layer")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(f"Mechanism probe — {setname} set (n={s['n_pairs']})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = os.path.join(PROBE_DIR, f"probe_{setname}.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", out)
    return out


def main():
    os.makedirs(PROBE_DIR, exist_ok=True)
    for setname in ("neutral", "disagreement"):
        _fig(setname)


if __name__ == "__main__":
    main()
