"""B01 figure: relative L2 vs residual evaluations (sampler overhead included), median + IQR.

Usage: python experiments/B01_sampling_equal_compute/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "b01_sampling.png"
runs = json.loads((results / "curves.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [  # categorical slots 1-5, fixed order
    ("uniform", "Uniform, every step", "#2a78d6"),
    ("stratified", "Stratified, every step", "#eb6834"),
    ("uniform_periodic", "Uniform, every 100 steps", "#1baf7a"),
    ("rad", "RAD (k=1, c=1), every 100 steps", "#eda100"),
    ("rl_bandit", "RL bandit, every 100 steps", "#e87ba4"),
]
PDES = [("burgers", "Burgers, ν = 0.01/π"), ("allen_cahn", "Allen–Cahn, D = 0.001")]

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
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), dpi=220, facecolor=SURFACE)
for ax, (pde, title) in zip(axes, PDES, strict=False):
    ax.set_facecolor(SURFACE)
    seeds = set()
    for key, label, color in ARMS:
        sel = [r for r in runs if r["pde"] == pde and r["strategy"] == key]
        if not sel:
            continue
        seeds |= {r["seed"] for r in sel}
        evals = np.median([r["curve_evals"] for r in sel], axis=0) / 1e6
        err = np.array([r["curve_rel_l2"] for r in sel])
        med = np.median(err, axis=0)
        q25, q75 = np.percentile(err, [25, 75], axis=0)
        ax.fill_between(evals, q25, q75, color=color, alpha=0.14, linewidth=0)
        ax.plot(evals, med, color=color, linewidth=2, label=label)
        ax.plot(
            evals[-1],
            med[-1],
            "o",
            color=color,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=2,
        )
    ax.set_yscale("log")
    ax.set_title(f"{title}  ({len(seeds)} seeds)", loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("residual evaluations, millions (sampler overhead included)")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes[0].set_ylabel("relative L2 error (median, IQR band)")
axes[1].legend(frameon=False, loc="upper right", fontsize=9, labelcolor=INK)
fig.suptitle(
    "Collocation sampling at equal training steps (15k Adam steps, 1000 points)",
    x=0.01,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
