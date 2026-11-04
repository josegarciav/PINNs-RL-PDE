"""K45 figure: the parametric surrogate against dedicated networks, and the swarm on it.

Usage: python experiments/K45_parametric_surrogate/plot.py [results_dir] [out.png]
"""

import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k45_parametric_surrogate.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SURR, DED = "#2a78d6", "#eb6834"
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

surr = [r for r in runs if r["part"] == "surrogate"]
ded = [r for r in runs if r["part"] == "dedicated"]
swarm = [r for r in runs if r["part"] == "swarm"]

fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

PANELS = [
    ("heat", "heat: α ∈ [0.02, 0.5]", lambda th: th[0], "α"),
    ("kdv", "KdV: (α, β) ∈ [2, 12] × [0.3, 3]", None, "test value (α, β)"),
    ("burgers", "Burgers: ν ∈ [0.05, 0.5]", lambda th: math.exp(th[0]), "ν"),
]
for ax, (problem, title, xval, xlabel) in zip(axes[:3], PANELS, strict=True):
    ss = sorted([r for r in surr if r["problem"] == problem], key=lambda r: r["seed"])
    dd = [r for r in ded if r["problem"] == problem]
    n = len(ss[0]["errors_at_test_values"])
    thetas = [next(r["theta"] for r in dd if r["k"] == k) for k in range(n)]
    xs = np.array([xval(th) for th in thetas]) if xval else np.arange(n)
    s_err = np.array([r["errors_at_test_values"] for r in ss])  # seeds x values
    d_err = np.array([[r["error"] for r in dd if r["k"] == k] for k in range(n)]).T
    for arr, color, label, marker in (
        (s_err, SURR, "one parametric surrogate", "o"),
        (d_err, DED, "a network per value", "s"),
    ):
        ax.plot(xs, arr.mean(0), "-", color=color, linewidth=2, marker=marker, label=label)
        ax.fill_between(xs, arr.min(0), arr.max(0), color=color, alpha=0.15, linewidth=0)
    ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.set_xlabel(xlabel)
    if not xval:
        ax.set_xticks(xs)
        ax.set_xticklabels([f"({a:g}, {b:g})" for a, b in thetas], fontsize=9)
    elif problem == "burgers":
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs])
        ax.minorticks_off()
axes[0].set_ylabel("relative L2 error vs the closed form (2 seeds)")
axes[0].legend(frameon=False, loc="upper center")

ax = axes[3]
SERIES = [
    ("heat", 0, "heat α", "#2a78d6"),
    ("kdv", 0, "KdV α", "#1baf7a"),
    ("kdv", 1, "KdV β", "#eda100"),
]
for problem, j, label, color in SERIES:
    ww = [r for r in swarm if r["problem"] == problem]
    truth = ww[0]["truth"][j]
    c = np.array([r["closed_form"]["width"][j] for r in ww]) / truth
    s = np.array([r["surrogate"]["width"][j] for r in ww]) / truth
    ax.scatter(c, s, color=color, s=34, label=label, zorder=3)
lim = [0.01, 0.1]
ax.plot(lim, lim, "--", color=INK2, linewidth=1)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lim)
ax.set_ylim(lim)
for axis in (ax.xaxis, ax.yaxis):
    axis.set_ticks([0.01, 0.02, 0.05, 0.1])
    axis.set_ticklabels(["1 %", "2 %", "5 %", "10 %"])
ax.minorticks_off()
ax.set_xlabel("90 % interval width on the closed form (relative)")
ax.set_ylabel("width on the surrogate")
ax.set_title("swarm region on the surrogate (10 data sets)", loc="left")
ax.legend(frameon=False, loc="upper left")

fig.suptitle(
    "K45: one network u(x, t; θ) over the prior range, against a network per θ at the same budget",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
