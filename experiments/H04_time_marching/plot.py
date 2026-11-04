"""H04 figure: the PINN error against the number of time windows, per system.

Usage: python experiments/H04_time_marching/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "h04_time_marching.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9"]
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
fig, ax = plt.subplots(figsize=(9, 5.2), dpi=220, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, which="major", color=GRID, linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
systems = list(dict.fromkeys(r["system"] for r in runs))
windows = sorted({r["windows"] for r in runs})
for i, system in enumerate(systems):
    med = [
        np.median([r["rel_l2_mean"] for r in runs if r["system"] == system and r["windows"] == w])
        for w in windows
    ]
    ax.plot(
        windows,
        med,
        "-o",
        color=PALETTE[i % len(PALETTE)],
        linewidth=2,
        label=system.replace("_", " "),
    )
    for w in windows:
        pts = [r["rel_l2_mean"] for r in runs if r["system"] == system and r["windows"] == w]
        ax.scatter([w] * len(pts), pts, color=PALETTE[i % len(PALETTE)], s=14, alpha=0.6, zorder=3)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(windows)
ax.set_xticklabels([str(w) for w in windows])
ax.set_xlabel("time windows (same total Adam budget, L-BFGS finish on the whole window)")
ax.set_ylabel("relative L2 error, mean over fields (median of seeds, dots per seed)")
ax.set_title("H04: time-marching windows against one-shot training", loc="left", fontweight="bold")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
