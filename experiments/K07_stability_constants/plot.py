"""K07 figure: error vs residual across budgets, one curve per equation, and the constant C.

Usage: python experiments/K07_stability_constants/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k07_stability_constants.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PROBLEMS = [
    "heat",
    "convection",
    "wave",
    "klein_gordon",
    "poisson",
    "burgers_travelling_wave",
    "beam",
    "kdv_two_parameter",
    "heat_2d",
    "sine_gordon",
]
LABELS = [
    "heat",
    "convection",
    "wave",
    "Klein–Gordon",
    "Poisson",
    "Burgers wave",
    "beam",
    "KdV (α, β)",
    "heat 2-D",
    "sine-Gordon",
]
cmap = plt.get_cmap("tab10")
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
fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
budgets = sorted({r["steps"] for r in runs})
for i, (p, label) in enumerate(zip(PROBLEMS, LABELS, strict=False)):
    sel = [r for r in runs if r["problem"] == p]
    res = [np.median([r["residual"] for r in sel if r["steps"] == b]) for b in budgets]
    err = [np.median([r["error"] for r in sel if r["steps"] == b]) for b in budgets]
    axes[0].plot(res, err, "o-", color=cmap(i), linewidth=1.8, markersize=5, label=label)
    C = [np.median([r["ratio"] for r in sel if r["steps"] == b]) for b in budgets]
    axes[1].plot(budgets, C, "o-", color=cmap(i), linewidth=1.8, markersize=5, label=label)
lim = [3e-3, 0.6]
axes[0].plot(lim, lim, "--", color=INK2, linewidth=1, label="error = residual (C = 1)")
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlabel("RMS residual on an independent sample")
axes[0].set_ylabel("relative L2 error against the closed form")
axes[0].set_title(
    "Error vs residual as training proceeds (500 → 8,000 steps, left to right)",
    loc="left",
    fontsize=11.5,
    pad=10,
)
axes[0].legend(frameon=False, fontsize=8, ncol=2)
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Adam steps")
axes[1].set_ylabel("C = error / residual")
axes[1].set_title("The empirical stability constant", loc="left", fontsize=11.5, pad=10)
axes[1].axhline(1, color=INK2, linewidth=0.8, linestyle="--")
fig.suptitle(
    "K07: how the training residual translates into solution error (median of 2 seeds, 8,192 fresh points)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
