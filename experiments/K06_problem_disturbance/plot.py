"""K06 figure: the PINN's response to a problem perturbation against the exact solution's.

Usage: python experiments/K06_problem_disturbance/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k06_problem_disturbance.png"
rows = json.loads((results / "responses.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CASES = [
    ("heat", "ic", "heat: initial amplitude", "#2a78d6", "o"),
    ("heat", "alpha", "heat: diffusivity α", "#eb6834", "o"),
    ("heat", "bc", "heat: boundary value", "#1baf7a", "o"),
    ("wave", "ic", "wave: initial amplitude", "#2a78d6", "^"),
    ("wave", "c", "wave: speed c", "#eb6834", "^"),
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
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
for p, k, label, color, marker in CASES:
    sel = [r for r in rows if r["problem"] == p and r["kind"] == k]
    ex = np.array([r["exact_response"] for r in sel])
    pi = np.array([r["pinn_response"] for r in sel])
    axes[0].scatter(
        ex,
        pi,
        color=color,
        marker=marker,
        s=34,
        edgecolors=SURFACE,
        linewidths=0.8,
        label=label,
        zorder=3,
    )
    eps = sorted({r["eps"] for r in sel})
    med = [np.median([r["response_error"] for r in sel if r["eps"] == e]) for e in eps]
    axes[1].plot(eps, med, "-", marker=marker, color=color, linewidth=2, label=label)
lim = [3e-3, 0.5]
axes[0].plot(lim, lim, "--", color=INK2, linewidth=1, label="PINN response = exact response")
base_l2 = np.median([r["l2_error_base"] for r in rows])
axes[0].axhline(base_l2, color=INK2, linewidth=0.8, linestyle=":")
axes[0].text(
    lim[0] * 1.1,
    base_l2 * 1.1,
    f"field error of the base run ≈ {base_l2:.3f}",
    fontsize=8.5,
    color=INK2,
)
axes[0].set_xscale("log")
axes[0].set_yscale("log")
axes[0].set_xlim(lim)
axes[0].set_ylim(lim)
axes[0].set_xlabel("‖δu_exact‖ / ‖u‖  (perturbation of 1 %, 5 %, 20 %)")
axes[0].set_ylabel("‖δu_PINN‖ / ‖u‖")
axes[0].set_title("Size of the response", loc="left", fontsize=12, pad=10)
axes[0].legend(frameon=False, fontsize=8.5, loc="upper left")
axes[1].set_xscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("relative perturbation ε")
axes[1].set_ylabel("‖δu_PINN − δu_exact‖ / ‖δu_exact‖ (median of 3 seeds)")
axes[1].set_title("Error of the response", loc="left", fontsize=12, pad=10)
axes[1].axhline(1, color=INK2, linewidth=0.8, linestyle="--")
axes[1].legend(frameon=False, fontsize=8.5)
fig.suptitle(
    "K06: does the trained field respond to a problem perturbation as the equation does? (3,000 Adam steps)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
