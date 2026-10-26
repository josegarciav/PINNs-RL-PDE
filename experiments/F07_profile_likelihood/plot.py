"""F07 figure: profile of the data misfit and of the total loss against the fixed alpha.

Usage: python experiments/F07_profile_likelihood/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f07_profile_likelihood.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]
grid = sorted({r["alpha"] for r in runs})
seeds = sorted({r["seed"] for r in runs})
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
fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for s, color in zip(seeds, COLORS, strict=False):
    sel = sorted([r for r in runs if r["seed"] == s], key=lambda r: r["alpha"])
    prof = np.array([r["neg2loglik"] for r in sel])
    exact = np.array([r["neg2loglik_exact_solution"] for r in sel])
    F = sel[0]["fisher_information"]
    a0 = grid[int(np.argmin(exact))]
    axes[0].plot(
        grid,
        exact - exact.min(),
        "--",
        color=color,
        linewidth=1.5,
        label=f"seed {s}: exact solution at each α",
    )
    axes[0].plot(
        grid,
        prof - prof.min(),
        "o-",
        color=color,
        linewidth=2,
        label=f"seed {s}: PINN trained at each α",
    )
    axes[0].plot(grid, F * (np.array(grid) - a0) ** 2, ":", color=color, linewidth=1.2)
    res = np.array([r["residual_loss"] for r in sel])
    axes[1].plot(grid, res, "o-", color=color, linewidth=2, label=f"seed {s}")
axes[0].plot([], [], ":", color=INK2, label="Fisher parabola F (α − α₀)²")
axes[0].axhline(3.84, color=INK2, linewidth=0.8, linestyle="-.")
axes[0].text(grid[0], 3.84 * 1.15, "95 % interval: Δ(−2 log L) = 3.84", fontsize=8.5, color=INK2)
axes[0].axvline(0.1, color=INK2, linewidth=0.8)
axes[0].set_xlabel("fixed α")
axes[0].set_ylabel("−2 log L − minimum")
axes[0].set_title("Profile of the data misfit", loc="left", fontsize=12, pad=10)
axes[0].set_yscale("symlog", linthresh=10)
axes[0].legend(frameon=False, fontsize=8)
axes[1].axvline(0.1, color=INK2, linewidth=0.8)
axes[1].set_yscale("log")
axes[1].set_xlabel("fixed α")
axes[1].set_ylabel("PDE residual loss of the trained network")
axes[1].set_title(
    "What the network gives up: the residual at the wrong α", loc="left", fontsize=12, pad=10
)
axes[1].legend(frameon=False, fontsize=9)
fig.suptitle(
    "F07: profile likelihood of the heat diffusivity (200 observations, noise 0.05, α fixed and the network retrained)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
