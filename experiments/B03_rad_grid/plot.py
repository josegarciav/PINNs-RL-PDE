"""B03 figure: median error over the RAD (k, c) grid, one heat map per equation.

Usage: python experiments/B03_rad_grid/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "b03_rad_grid.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
KS, CS = [0.5, 1.0, 2.0, 4.0], [0.0, 0.5, 1.0, 2.0]
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
fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=220, facecolor=SURFACE)
for ax, (pde, title) in zip(axes, PDES, strict=False):
    grid = np.array(
        [
            [
                np.median(
                    [
                        r["rel_l2"]
                        for r in runs
                        if r["pde"] == pde
                        and r["strategy"] == "rad"
                        and r["rad_k"] == k
                        and r["rad_c"] == c
                    ]
                )
                for c in CS
            ]
            for k in KS
        ]
    )
    control = np.median(
        [r["rel_l2"] for r in runs if r["pde"] == pde and r["strategy"] == "uniform_periodic"]
    )
    ratio = grid / control
    im = ax.imshow(np.log2(ratio), cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
    for i, k in enumerate(KS):
        for j, c in enumerate(CS):
            ax.text(
                j,
                i,
                f"{grid[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=9.5,
                color=INK if abs(np.log2(ratio[i, j])) < 1.2 else SURFACE,
            )
    ax.set_xticks(range(len(CS)), [f"c = {c:g}" for c in CS])
    ax.set_yticks(range(len(KS)), [f"k = {k:g}" for k in KS])
    ax.set_title(f"{title}: uniform control {control:.3f}", loc="left", fontsize=12, pad=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
cbar.set_label("log₂( RAD error / uniform error )  (blue: better than uniform)")
fig.suptitle(
    "B03: the RAD (k, c) grid, median relative L2 of 3 seeds after 10,000 steps, resampled every 100",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.savefig(out_png, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out_png)
