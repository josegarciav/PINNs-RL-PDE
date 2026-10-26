"""K05 figure: observed shift of the estimate vs the linear prediction, per disturbance.

Usage: python experiments/K05_data_disturbance/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k05_data_disturbance.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
KINDS = [  # categorical slots, fixed order
    ("bias", "bias on all points", "#2a78d6", "o"),
    ("bias_one", "bias on one probe", "#2a78d6", "s"),
    ("delay", "timestamp delay", "#eb6834", "o"),
    ("delay_one", "delay on one probe", "#eb6834", "s"),
    ("noise", "extra noise", "#1baf7a", "o"),
    ("outliers", "outliers", "#eda100", "o"),
]
base = {(r["layout"], r["seed"]): r["alpha_hat"] for r in runs if r["kind"] == "none"}
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
ax_lin, ax_drop = axes
for kind, label, color, marker in KINDS:
    sel = [r for r in runs if r["kind"] == kind]
    pred = np.array([r["predicted_shift"] for r in sel]) / 0.1
    obs = np.array([r["alpha_hat"] - base[(r["layout"], r["seed"])] for r in sel]) / 0.1
    ax_lin.scatter(
        pred,
        obs,
        color=color,
        marker=marker,
        s=34,
        edgecolors=SURFACE,
        linewidths=0.8,
        label=label,
        zorder=3,
    )
lim = [-0.25, 0.15]
ax_lin.plot(lim, lim, "--", color=INK2, linewidth=1, label="observed = predicted")
ax_lin.set_xlim(lim)
ax_lin.set_ylim(lim)
ax_lin.set_aspect("equal")
ax_lin.set_xlabel("predicted shift of α, F⁻¹Jᵀδy / α")
ax_lin.set_ylabel("observed shift of α (paired with the undisturbed run) / α")
ax_lin.set_title(
    "Linear response vs the PINN's response (5 seeds per point)", loc="left", fontsize=12, pad=10
)
ax_lin.legend(frameon=False, fontsize=8.5, loc="upper left")

# Dropout and lost probe: shift and spread vs retained fraction, with the bound's growth.
drop = [r for r in runs if r["kind"] in ("dropout", "dropout_one", "none")]
for layout, color in [("scatter", "#2a78d6"), ("probes", "#eb6834")]:
    sel = [r for r in drop if r["layout"] == layout]
    fracs = sorted({r["n_obs"] / 200 for r in sel}, reverse=True)
    med = [
        np.median([abs(r["alpha_hat"] - 0.1) / 0.1 for r in sel if r["n_obs"] / 200 == f])
        for f in fracs
    ]
    q25 = [
        np.percentile([abs(r["alpha_hat"] - 0.1) / 0.1 for r in sel if r["n_obs"] / 200 == f], 25)
        for f in fracs
    ]
    q75 = [
        np.percentile([abs(r["alpha_hat"] - 0.1) / 0.1 for r in sel if r["n_obs"] / 200 == f], 75)
        for f in fracs
    ]
    ax_drop.fill_between(fracs, q25, q75, color=color, alpha=0.14, linewidth=0)
    ax_drop.plot(
        fracs, med, "o-", color=color, linewidth=2, label=f"{layout}: error of α (median, IQR)"
    )
ax_drop.set_xlabel("fraction of the 200 observations retained")
ax_drop.set_ylabel("relative error of α")
ax_drop.set_title(
    "Dropout: random points, or one of four probes lost", loc="left", fontsize=12, pad=10
)
ax_drop.invert_xaxis()
ax_drop.legend(frameon=False, fontsize=9)
fig.suptitle(
    "K05: response of the identified heat diffusivity to data disturbances (200 observations, noise 0.05)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
