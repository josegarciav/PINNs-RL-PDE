"""B08 figure: error vs collocation count for uniform and RAD, with the wall-clock cost.

Usage: python experiments/B08_point_count/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "b08_point_count.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [
    ("uniform_periodic", "uniform, every 100 steps", "#2a78d6"),
    ("rad", "RAD (k=1, c=1), every 100 steps", "#eda100"),
]
PDES = [("burgers", "Burgers, ν = 0.01/π"), ("allen_cahn", "Allen–Cahn, D = 0.001")]
COUNTS = [250, 500, 1000, 2000, 5000]
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
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for ax, (pde, title) in zip(axes[:2], PDES, strict=False):
    for key, label, color in ARMS:
        e = [
            [
                r["rel_l2"]
                for r in runs
                if r["pde"] == pde and r["strategy"] == key and r["n_collocation"] == n
            ]
            for n in COUNTS
        ]
        med = [np.median(v) for v in e]
        ax.fill_between(
            COUNTS,
            [np.percentile(v, 25) for v in e],
            [np.percentile(v, 75) for v in e],
            color=color,
            alpha=0.14,
            linewidth=0,
        )
        ax.plot(
            COUNTS,
            med,
            "o-",
            color=color,
            linewidth=2,
            markersize=6,
            markeredgecolor=SURFACE,
            label=label,
        )
        slope = np.polyfit(np.log(COUNTS), np.log(med), 1)[0]
        ax.annotate(
            f"slope {slope:+.2f}",
            (COUNTS[-1], med[-1]),
            textcoords="offset points",
            xytext=(-40, 8 if key == "rad" else -14),
            fontsize=8.5,
            color=color,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("collocation points per step")
    ax.set_title(title, loc="left", fontsize=12, pad=10)
axes[0].set_ylabel("relative L2 error after 5,000 steps (median, IQR band)")
axes[0].legend(frameon=False, fontsize=9)
for key, label, color in ARMS:
    t = [
        np.median(
            [r["train_time_s"] for r in runs if r["strategy"] == key and r["n_collocation"] == n]
        )
        for n in COUNTS
    ]
    axes[2].plot(
        COUNTS,
        t,
        "o-",
        color=color,
        linewidth=2,
        markersize=6,
        markeredgecolor=SURFACE,
        label=label,
    )
axes[2].set_xscale("log")
axes[2].set_yscale("log")
axes[2].set_xlabel("collocation points per step")
axes[2].set_ylabel("wall-clock seconds for 5,000 steps (one thread)")
axes[2].set_title("Cost", loc="left", fontsize=12, pad=10)
fig.suptitle(
    "B08: error and cost against the number of collocation points (3 paired seeds)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
