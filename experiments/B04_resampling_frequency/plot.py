"""B04 figure: error vs resampling interval for uniform and RAD, Burgers and Allen–Cahn.

Usage: python experiments/B04_resampling_frequency/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "b04_resampling_frequency.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("uniform", "uniform", "#2a78d6"), ("rad", "RAD (k=1, c=1)", "#eda100")]
PDES = [("burgers", "Burgers, ν = 0.01/π"), ("allen_cahn", "Allen–Cahn, D = 0.001")]
INTERVALS = [1, 10, 100, 1000, 10000]
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
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=220, facecolor=SURFACE)
for ax, (pde, title) in zip(axes, PDES, strict=False):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for key, label, color in ARMS:
        med, q25, q75 = [], [], []
        for iv in INTERVALS:
            strategies = (
                ("uniform",)
                if (key == "uniform" and iv == 1)
                else (("uniform_periodic",) if key == "uniform" else ("rad",))
            )
            e = [
                r["rel_l2"]
                for r in runs
                if r["pde"] == pde and r["strategy"] in strategies and r["resample_every"] == iv
            ]
            med.append(np.median(e))
            q25.append(np.percentile(e, 25))
            q75.append(np.percentile(e, 75))
        ax.fill_between(INTERVALS, q25, q75, color=color, alpha=0.14, linewidth=0)
        ax.plot(
            INTERVALS,
            med,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            markeredgecolor=SURFACE,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(INTERVALS, ["1", "10", "100", "1,000", "never"])
    ax.set_xlabel("steps between resamplings")
    ax.set_title(title, loc="left", fontsize=12, pad=10)
axes[0].set_ylabel("relative L2 error after 10,000 steps (median, IQR band)")
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(
    "B04: how often should collocation points be resampled? (5 paired seeds, 1,000 points)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
