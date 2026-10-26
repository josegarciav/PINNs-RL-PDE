"""B02 figure: final error per seed, by equation and sampler.

Usage: python experiments/B02_more_pdes/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "b02_more_pdes.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [
    ("uniform", "uniform, every step", "#2a78d6"),
    ("uniform_periodic", "uniform, every 100 steps", "#1baf7a"),
    ("rad", "RAD (k=1, c=1), every 100 steps", "#eda100"),
]
PDES = [
    ("heat", "heat"),
    ("convection", "convection"),
    ("wave", "wave"),
    ("kdv_soliton", "KdV soliton"),
]
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
fig, ax = plt.subplots(figsize=(12, 5), dpi=220, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.grid(True, axis="y", color=GRID, linewidth=0.8)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
pos = np.arange(len(PDES))
for k, (key, label, color) in enumerate(ARMS):
    off = (k - 1) * 0.26
    for i, (pde, _) in enumerate(PDES):
        sel = [r for r in runs if r["pde"] == pde and r["strategy"] == key]
        e = [r["rel_l2"] for r in sel]
        ax.scatter(
            [i + off] * len(e),
            e,
            color=color,
            s=26,
            alpha=0.75,
            edgecolors=SURFACE,
            linewidths=0.6,
            zorder=3,
        )
        ax.plot(
            [i + off - 0.1, i + off + 0.1],
            [np.median(e)] * 2,
            color=color,
            linewidth=2.5,
            label=label if i == 0 else None,
            zorder=4,
        )
ax.set_yscale("log")
ax.set_xticks(pos, [p[1] for p in PDES])
ax.set_ylabel("relative L2 error after 10,000 steps (dots: seeds; bar: median)")
ax.set_title(
    "B02: samplers on four more equations, 5 paired seeds", loc="left", fontsize=12, pad=10
)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
