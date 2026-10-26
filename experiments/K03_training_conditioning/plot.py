"""K03 figure: Hessian extremes, kernel trace ratios and residual-kernel conditioning by problem and architecture.

Usage: python experiments/K03_training_conditioning/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k03_training_conditioning.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARCHS = [("feedforward", "tanh MLP", "#2a78d6"), ("fourier", "Fourier features", "#eb6834")]
PROBLEMS = [
    "convection",
    "heat",
    "wave",
    "klein_gordon",
    "poisson",
    "burgers_travelling_wave",
    "allen_cahn",
    "beam",
]
LABELS = [
    "convection\n(1st)",
    "heat\n(2nd)",
    "wave\n(2nd)",
    "Klein–\nGordon",
    "Poisson\n(steady)",
    "Burgers\nwave",
    "Allen–\nCahn",
    "beam\n(4th)",
]

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    }
)
fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), dpi=200, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
pos = np.arange(len(PROBLEMS))
width = 0.38


def stat(key_fn, arch):
    return [
        np.median([key_fn(r) for r in runs if r["problem"] == p and r["arch"] == arch])
        for p in PROBLEMS
    ]


for k, (arch, label, color) in enumerate(ARCHS):
    x = pos + (k - 0.5) * width
    axes[0].bar(
        x, stat(lambda r: r["hessian"]["largest"], arch), width * 0.9, color=color, label=label
    )
    axes[1].bar(
        x,
        stat(lambda r: r["hessian"]["condition_number"], arch),
        width * 0.9,
        color=color,
        label=label,
    )
    axes[2].bar(
        x,
        stat(lambda r: r["kernels"]["residual"]["trace"] / r["kernels"]["boundary"]["trace"], arch),
        width * 0.9,
        color=color,
        label=label,
    )
    axes[3].bar(
        x,
        stat(lambda r: r["kernels"]["residual"]["condition_number"], arch),
        width * 0.9,
        color=color,
        label=label,
    )
titles = [
    "Largest Hessian eigenvalue λ_max",
    "Hessian condition number (positive part)",
    "tr(K_residual) / tr(K_boundary)",
    "Condition number of K_residual",
]
for ax, title in zip(axes, titles, strict=False):
    ax.set_yscale("log")
    ax.set_xticks(pos, LABELS, fontsize=8.5)
    ax.set_title(title, loc="left", fontsize=11, pad=8)
axes[2].axhline(1, color=INK2, linewidth=0.8, linestyle="--")
axes[0].legend(frameon=False, fontsize=9)
seeds = len({r["seed"] for r in runs})
fig.suptitle(
    f"K03: conditioning of PINN training after {runs[0]['steps']} Adam steps, median over {seeds} seeds",
    x=0.01,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
