"""K08 figure: error, residual, stability constant, quadrature gap and cost vs dimension.

Usage: python experiments/K08_dimension_scaling/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k08_dimension_scaling.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"heat": "#2a78d6", "convection": "#eb6834"}
STYLES = {"matched total": "-", "matched per axis": "--"}
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
dims = sorted({r["dimension"] for r in runs})
panels = [
    ("error", "relative L2 error of u", True),
    ("residual", "RMS residual on fresh points", True),
    ("ratio", "stability constant C = error / residual", True),
    ("quadrature_gap", "quadrature gap of the residual MSE", True),
    ("time_per_step_ms", "ms per Adam step (2 threads)", True),
]
fig, axes = plt.subplots(
    1, len(panels), figsize=(4.2 * len(panels), 4.4), dpi=220, facecolor=SURFACE
)
for ax, (key, label, log) in zip(axes, panels, strict=True):
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for problem, color in COLORS.items():
        for arm, ls in STYLES.items():
            med = []
            for d in dims:
                sel = [
                    r[key]
                    for r in runs
                    if r["problem"] == problem and r["arm"] == arm and r["dimension"] == d
                ]
                med.append(np.median(sel) if sel else np.nan)
            ax.plot(
                dims, med, ls, marker="o", color=color, linewidth=1.8, label=f"{problem}, {arm}"
            )
    if log:
        ax.set_yscale("log")
    ax.set_xlabel("spatial dimension d")
    ax.set_xticks(dims)
    ax.set_title(label, loc="left", fontsize=10.5, pad=8)
axes[0].legend(frameon=False, fontsize=8, labelcolor=INK)
fig.suptitle(
    "K08: heat and convection in d = 1…5, 4,000 Adam steps, median of the seeds",
    x=0.01,
    ha="left",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
