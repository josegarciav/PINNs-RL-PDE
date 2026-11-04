"""K48 figure: Fisher spectrum of the cosine modes, bound per mode vs the truth's coefficients,
basis truncation error, and the recovered field with a basis vs a network.

Usage: python experiments/K48_reduced_bases/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k48_reduced_bases.png"
fisher = json.loads((results / "fisher.json").read_text())
runs_path = results / "runs.json"
runs = json.loads(runs_path.read_text()) if runs_path.exists() else []

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"bump": "#2a78d6", "layers": "#eb6834"}
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
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# (a) Fisher eigenvalues
ax = axes[0]
for name, color in COLORS.items():
    ev = np.array(fisher[name]["eigenvalues"])
    ax.plot(np.arange(1, len(ev) + 1), 1 / np.sqrt(ev), "o-", color=color, linewidth=2, label=name)
ax.set_yscale("log")
ax.set_xlabel("Fisher eigen-direction (best determined first)")
ax.set_ylabel("Cramér–Rao std along the direction")
ax.set_title("What 200 observations determine (noise 0.01)", loc="left", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)

# (b) per-mode bound vs the truth's coefficient magnitude
ax = axes[1]
modes = np.arange(len(fisher["bump"]["cr_std_per_mode"]))
width = 0.38
for i, (name, color) in enumerate(COLORS.items()):
    cr = np.array(fisher[name]["cr_std_per_mode"])
    coef = np.maximum(np.abs(np.array(fisher[name]["truth_coefficients"])), 1e-4)  # odd modes are 0
    ax.bar(
        modes + (i - 0.5) * width,
        cr,
        width=width * 0.9,
        color=color,
        alpha=0.9,
        label=f"{name}: CR std of c_k",
    )
    ax.plot(
        modes + (i - 0.5) * width,
        coef,
        "v",
        color=color,
        markersize=6,
        markeredgecolor=INK,
        label=f"{name}: |c_k| of the truth",
    )
ax.set_yscale("log")
ax.set_ylim(5e-5, 0.5)
ax.set_xlabel("cosine mode k")
ax.set_xticks(modes)
ax.set_title("Bound per mode against the truth's coefficients", loc="left", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=8, labelcolor=INK)

# (c) truncation error of the basis
ax = axes[2]
for name, color in COLORS.items():
    pe = fisher[name]["projection_error_vs_modes"]
    ax.plot(np.arange(1, len(pe) + 1), pe, "s-", color=color, linewidth=2, label=name)
ax.set_yscale("log")
ax.set_xlabel("number of cosine modes n")
ax.set_ylabel("relative L2 error of the best n-mode fit")
ax.set_title("How many modes the truth needs", loc="left", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=9, labelcolor=INK)

# (d) recovered fields: basis vs network
ax = axes[3]
if runs:
    grid = np.linspace(0, 1, len(runs[0]["profile"]))
    truths = {
        "bump": 0.05 + 0.1 * np.exp(-(((grid - 0.5) / 0.15) ** 2)),
        "layers": 0.02 + 0.18 / (1 + np.exp(-(grid - 0.5) / 0.05)),
    }
    for name, color in COLORS.items():
        ax.plot(grid, truths[name], "--", color=INK, linewidth=1.2)
        for model, ls in (("basis", "-"), ("network", ":")):
            sel = [r["profile"] for r in runs if r["truth"] == name and r["model"] == model]
            if sel:
                ax.plot(
                    grid,
                    np.median(sel, axis=0),
                    ls,
                    color=color,
                    linewidth=2,
                    label=f"{name}: {model}",
                )
    ax.set_xlabel("x")
    ax.set_ylabel("alpha")
    ax.set_title(
        "Recovered field, basis (6 modes) vs network; truth dashed",
        loc="left",
        fontsize=10.5,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
else:
    ax.text(0.5, 0.5, "training part not run", ha="center", va="center", transform=ax.transAxes)

fig.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
