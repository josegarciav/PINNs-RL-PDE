"""J07c/J07d figure: the reference-solver misfit of every closure, and the shooting control.

Usage: python experiments/J07_droplet_drying/plot_guard.py [results_dir] [out.png]
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
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j07c_guard.png"
closures = json.loads((results / "closure2.json").read_text())
guard = json.loads((results / "guard.json").read_text())
shooting = json.loads((results / "shooting.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GREY, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#7a7a7a", "#8e5bd9"
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
fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

dr = plant()
X = np.linspace(dr.X_eq, dr.X0, 400)
cold = [r for r in shooting if r["arm"] == "anchored" and r["start"] == "cold"]


def draw_closures(ax):
    ax.plot(X, dr.drying_curve(X), "-", color=INK, linewidth=2.6, label="plant's drying curve")
    for i, r in enumerate(r for r in closures if r["arm"] == "pointwise_only"):
        ax.plot(
            r["X_grid"],
            r["f_hat"],
            "-",
            color=GREY,
            linewidth=1.2,
            label="pointwise start" if i == 0 else None,
        )
    for i, r in enumerate(r for r in closures if r["arm"] == "pointwise"):
        ax.plot(
            r["X_grid"],
            r["f_hat"],
            "-",
            color=BLUE,
            linewidth=1.3,
            alpha=0.9,
            label="pointwise start + PINN" if i == 0 else None,
        )
    for i, r in enumerate(cold):
        ax.plot(
            r["X_grid"],
            r["f_hat"],
            "--",
            color=ORANGE,
            linewidth=1.4,
            label="shooting, 24 knots, cold start" if i == 0 else None,
        )
    ax.set_xlabel("dry-basis moisture $X$")
    ax.set_ylabel("drying-rate closure $f(X)$")


ax = axes[0]
draw_closures(ax)
ax.set_xlim(0, 1.6)
ax.set_ylim(-0.05, 1.1)
ax.set_title("a) the closures, 3 seeds", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")

ax = axes[1]
draw_closures(ax)
ax.set_xlim(0.04, 0.2)
ax.set_ylim(-0.005, 0.08)
ax.axvline(dr.X_eq, color=INK2, linestyle=":", linewidth=1.2)
ax.text(dr.X_eq + 0.003, 0.074, "$X_{eq}$", fontsize=9, color=INK2)
ax.set_title("b) at the equilibrium moisture", loc="left")

ax = axes[2]
rows = [
    ("plant curve", GREY, [g["total"] for g in guard if g["arm"] == "plant curve"]),
    ("shooting\n(cold start)", ORANGE, [r["misfit"]["total"] for r in cold]),
    ("pointwise\n+ PINN", BLUE, [g["total"] for g in guard if g["arm"] == "pointwise"]),
    ("Legendre\nbasis", GREEN, [g["total"] for g in guard if g["arm"] == "basis"]),
    ("pointwise\nstart", GREY, [g["total"] for g in guard if g["arm"] == "pointwise_only"]),
    ("two-stage", PURPLE, [g["total"] for g in guard if g["arm"] == "two_stage"]),
]
cap = 3e3
for i, (_label, color, values) in enumerate(rows):
    shown = [min(v, cap) if np.isfinite(v) else cap for v in values]
    ax.scatter(
        np.full(len(shown), i) + np.linspace(-0.15, 0.15, len(shown)),
        shown,
        color=color,
        s=46,
        zorder=3,
    )
    if any(not np.isfinite(v) for v in values):
        ax.text(i, cap * 1.25, "∞", ha="center", fontsize=12, color=color)
ax.axhline(1.0, color=INK2, linestyle=":", linewidth=1.2)
ax.text(len(rows) - 0.5, 1.12, "noise level", fontsize=9, color=INK2, ha="right")
ax.set_yscale("log")
ax.set_ylim(0.5, cap * 2.2)
ax.set_xticks(range(len(rows)))
ax.set_xticklabels([r[0] for r in rows], fontsize=9)
ax.set_ylabel("record misfit under the reference solver (reduced $\\chi^2$)")
ax.set_title("c) does the closure reproduce the record?", loc="left")

fig.suptitle(
    "J07c/d: a reference-solver misfit as the guard on a learned closure, and the closure by shooting",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
