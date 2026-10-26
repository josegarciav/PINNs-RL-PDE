"""F06 figure: the four 90 % intervals on every data set, and their coverage, width and CRPS.

Usage: python experiments/F06_coverage/plot.py [results_dir] [out.png]
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f06_coverage.png"
rows = list(csv.DictReader(open(results / "intervals.csv")))
summary = json.loads((results / "summary.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
METHODS = [  # categorical slots, fixed order
    ("ensemble", "ensemble of 8 (empirical)", "#2a78d6"),
    ("laplace", "Laplace at member 0", "#eb6834"),
    ("fisher", "Fisher bound at member 0", "#1baf7a"),
    ("swarm", "swarm on the exact misfit", "#eda100"),
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
fig = plt.figure(figsize=(14, 5.4), dpi=220, facecolor=SURFACE)
gs = fig.add_gridspec(1, 3, width_ratios=[2.4, 0.8, 0.8])
ax_iv, ax_cov, ax_w = (fig.add_subplot(gs[0, i]) for i in range(3))
for ax in (ax_iv, ax_cov, ax_w):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
n = len(rows)
order = np.argsort([float(r["alpha_member0"]) for r in rows])
for k, (key, label, color) in enumerate(METHODS):
    off = (k - 1.5) * 0.18
    for j, i in enumerate(order):
        r = rows[i]
        lo, hi = float(r[f"{key}_lo"]), float(r[f"{key}_hi"])
        ax_iv.plot(
            [j + off, j + off],
            [lo, hi],
            color=color,
            linewidth=2.2,
            alpha=0.9,
            label=label if j == 0 else None,
        )
ax_iv.axhline(0.1, color=INK, linewidth=1, linestyle="--", label="true α")
ax_iv.scatter(
    np.arange(n),
    [float(rows[i]["alpha_member0"]) for i in order],
    color=INK,
    s=10,
    zorder=4,
    label="member 0 estimate",
)
ax_iv.set_xlabel("data set (sorted by member 0's estimate)")
ax_iv.set_ylabel("90 % interval for α")
ax_iv.set_title(
    f"Four intervals on each of {n} synthetic data sets (200 points, noise 0.05)",
    loc="left",
    fontsize=12,
    pad=10,
)
ax_iv.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper left")

keys = [m[0] for m in METHODS]
colors = [m[2] for m in METHODS]
names = ["ensemble", "Laplace", "Fisher", "swarm"]
x = np.arange(len(keys))
ax_cov.bar(x, [summary[k]["coverage"] for k in keys], 0.7, color=colors)
ax_cov.axhline(0.9, color=INK, linewidth=1, linestyle="--")
ax_cov.text(-0.4, 0.915, "nominal 90 %", fontsize=8.5, color=INK2)
ax_cov.set_xticks(x, names, fontsize=9)
ax_cov.set_ylim(0, 1.05)
ax_cov.set_ylabel("empirical coverage")
ax_cov.set_title("Coverage", loc="left", fontsize=12, pad=10)
ax_w.bar(x, [summary[k]["mean_width"] / 0.1 for k in keys], 0.7, color=colors)
ax_w.set_xticks(x, names, fontsize=9)
ax_w.set_ylabel("mean width of the interval / α")
ax_w.set_title("Width", loc="left", fontsize=12, pad=10)
fig.suptitle(
    "F06 / K10: are the intervals for the identified heat diffusivity calibrated?",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
