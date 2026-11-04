"""J07b figure: the drying-rate closure under the four second-pass arms.

Usage: python experiments/J07_droplet_drying/plot_closure2.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import plant  # noqa: E402

results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j07b_closure.png"
runs = json.loads((results / "closure2.json").read_text())
first = json.loads((results / "runs.json").read_text()) if (results / "runs.json").exists() else []

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {
    "pointwise_only": "#7a7a7a",
    "pointwise": "#2a78d6",
    "two_stage": "#eb6834",
    "basis": "#1baf7a",
    "first": "#8e5bd9",
    "lsq_cdc": "#eda100",
    "lsq_linear": "#d1427f",
}
LABELS = {
    "pointwise_only": "pointwise estimate, no PINN",
    "pointwise": "pointwise start + PINN",
    "two_stage": "two-stage (record first)",
    "basis": "Legendre basis, one stage",
    "first": "first pass (monotone net)",
    "lsq_cdc": "least squares, right form",
    "lsq_linear": "least squares, linear form",
}
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
fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

dr = plant()
ax = axes[0]
X = np.linspace(dr.X_eq, dr.X0, 400)
ax.plot(X, dr.drying_curve(X), "-", color=INK, linewidth=2.6, label="plant's drying curve")
seen = set()
for arm in ("pointwise", "basis", "two_stage"):
    for r in runs:
        if r["arm"] != arm:
            continue
        ax.plot(
            r["X_grid"],
            r["f_hat"],
            "-",
            color=COLORS[arm],
            linewidth=1.3,
            alpha=0.85,
            label=LABELS[arm] if arm not in seen else None,
        )
        seen.add(arm)
ax.set_xlabel("dry-basis moisture $X$")
ax.set_ylabel("drying-rate closure $f(X)$")
ax.set_xlim(0, dr.X0)
ax.set_ylim(-0.05, 1.1)
ax.set_title("a) the identified closure, 3 seeds per arm", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[1]
order = ["lsq_cdc", "pointwise_only", "pointwise", "basis", "two_stage", "first", "lsq_linear"]
vals, labels, cols = [], [], []
for key in order:
    if key == "first":
        v = [r["closure_rel_l2"] for r in first if r["part"] == "closure"]
    elif key.startswith("lsq"):
        v = [r["closure_rel_l2"] for r in first if r["part"] == key]
    else:
        v = [r["closure_rel_l2"] for r in runs if r["arm"] == key]
    if not v:
        continue
    vals.append(v)
    labels.append(LABELS[key])
    cols.append(COLORS[key])
x = np.arange(len(vals))
ax.bar(x, [np.median(v) for v in vals], color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), x[i]), v, color=INK, s=14, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_yscale("log")
ax.set_ylabel("relative L2 error of $f$ (median, dots per seed)")
ax.set_title("b) closure error by method", loc="left")
fig.suptitle(
    "J07b: making the free drying-rate closure train",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
