"""F09 figure: the recovered alpha(t) and alpha(x) against the truth, and the error vs noise.

Usage: python experiments/F09_field_unknowns/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f09_field_unknowns.png"
runs = json.loads((results / "runs.json").read_text())
best = json.loads((results / "best_constant.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRUTH, FIT, CONST = "#0b0b0b", "#2a78d6", "#eb6834"
NOISE_COLORS = {0.0: "#2a78d6", 0.01: "#eb6834", 0.05: "#1baf7a"}
ARMS = [
    ("linear", "alpha(t) = 0.05 + 0.1 t"),
    ("step", "alpha(t) = 0.05 + 0.1 H(t - 0.5)"),
    ("bump", "alpha(x): centred bump"),
    ("layers", "alpha(x): two layers"),
]
SHOW_NOISE = 0.01
grid = np.linspace(0, 1, len(runs[0]["profile"]))
seeds = sorted({r["seed"] for r in runs})
noises = sorted({r["noise"] for r in runs if r["model"] == "function"})

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

fig, axes = plt.subplots(2, 4, figsize=(18, 8.5), dpi=220, facecolor=SURFACE)
for ax in axes.ravel():
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

for col, (arm, title) in enumerate(ARMS):
    ax = axes[0, col]
    sel = [
        r for r in runs if r["arm"] == arm and r["model"] == "function" and r["noise"] == SHOW_NOISE
    ]
    profiles = np.array([r["profile"] for r in sel])
    truth = np.array(sel[0]["truth"])
    q25, q75 = np.percentile(profiles, [25, 75], axis=0)
    ax.fill_between(grid, q25, q75, color=FIT, alpha=0.18, linewidth=0)
    ax.plot(
        grid, np.median(profiles, axis=0), color=FIT, linewidth=2, label="recovered (median, IQR)"
    )
    ax.plot(grid, truth, color=TRUTH, linewidth=1.6, ls="--", label="truth")
    const = [
        r for r in runs if r["arm"] == arm and r["model"] == "constant" and r["noise"] == SHOW_NOISE
    ]
    if const:
        a_hat = np.median([r["alpha_hat"] for r in const])
        ax.axhline(a_hat, color=CONST, linewidth=1.4, ls=":", label=f"scalar fit {a_hat:.3f}")
    ax.axhline(
        best[arm]["alpha"],
        color=INK2,
        linewidth=1,
        ls=(0, (1, 2)),
        label=f"best constant {best[arm]['alpha']:.3f}",
    )
    ax.set_xlabel(sel[0]["variable"])
    ax.set_ylabel("alpha")
    ax.set_title(
        title + f"   (noise {SHOW_NOISE}, {len(sel)} seeds)", loc="left", fontsize=11, pad=8
    )
    if col == 0:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper left")

    ax = axes[1, col]
    width = 0.8 / len(noises)
    for k, noise in enumerate(noises):
        vals = np.array(
            [
                r["alpha_rel_l2"]
                for r in runs
                if r["arm"] == arm and r["model"] == "function" and r["noise"] == noise
            ]
        )
        xpos = k * width
        ax.bar(
            xpos,
            np.median(vals),
            width=width * 0.9,
            color=NOISE_COLORS[noise],
            label=f"noise {noise}",
        )
        ax.errorbar(
            xpos,
            np.median(vals),
            yerr=[
                [np.median(vals) - np.percentile(vals, 25)],
                [np.percentile(vals, 75) - np.median(vals)],
            ],
            color=INK2,
            capsize=3,
            linewidth=1,
        )
    ax.axhline(
        best[arm]["alpha_rel_l2"],
        color=INK2,
        linewidth=1,
        ls=(0, (1, 2)),
        label="best constant",
    )
    ax.set_xticks([])
    ax.set_ylabel("relative L2 error of alpha")
    ax.set_title("Error of the recovered function vs noise", loc="left", fontsize=11, pad=8)
    if col == 0:
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)

fig.suptitle(
    "F09: field-valued unknowns on the heat equation, 200 observations, 10,000 Adam steps",
    x=0.01,
    ha="left",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
