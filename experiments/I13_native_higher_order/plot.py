"""I13 figure: the trainer's step with autograd and with the kernel, five equations, two networks.

Usage: python experiments/I13_native_higher_order/plot.py [results_dir] [out.png]
"""

import json
import statistics
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "i13_native_higher_order.png"
rows = json.loads((results / "steps.json").read_text())["rows"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    }
)
problems = list(dict.fromkeys(r["problem"] for r in rows))
labels = {r["problem"]: r["label"].split(":")[0] for r in rows}
fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), dpi=200, facecolor=SURFACE)
for ax, layer_norm in zip(axes, (False, True), strict=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    y = np.arange(len(problems))
    for k, (mode, color) in enumerate((("autograd", BLUE), ("native", ORANGE))):
        med = [
            statistics.median(
                r["ms_per_step"]
                for r in rows
                if r["problem"] == p and r["layer_norm"] == layer_norm and r["mode"] == mode
            )
            for p in problems
        ]
        bars = ax.barh(y + (k - 0.5) * 0.38, med, height=0.36, color=color, label=mode)
        for b, v in zip(bars, med, strict=True):
            ax.text(v + 1.5, b.get_y() + b.get_height() / 2, f"{v:.0f}", va="center", fontsize=9)
    ratios = []
    for p in problems:
        a, n = (
            statistics.median(
                r["ms_per_step"]
                for r in rows
                if r["problem"] == p and r["layer_norm"] == layer_norm and r["mode"] == mode
            )
            for mode in ("autograd", "native")
        )
        ratios.append(a / n)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{labels[p]}  (×{r:.1f})" for p, r in zip(problems, ratios, strict=True)])
    ax.invert_yaxis()
    ax.set_xlabel("ms per training step (1 thread, median of 3 seeds, best of 5 rounds)")
    ax.set_title("layer normalization (packaged default)" if layer_norm else "plain tanh network")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
axes[0].legend(frameon=False, loc="lower right")
fig.suptitle(
    "I13: the compiled kernel on third, fourth and mixed derivatives (4 × 64 network, 5,120 points, M4 under load)",
    fontsize=12,
)
fig.tight_layout()
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
