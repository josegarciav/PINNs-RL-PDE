"""F02 figure: joint (alpha, beta) estimates per seed against the Cramér–Rao ellipse.

Usage: python experiments/F02_kdv_joint/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f02_kdv_joint.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
TRUE = (6.0, 1.0)
NOISES = sorted({r["noise"] for r in runs})
N_OBS = sorted({r["n_obs"] for r in runs})
seeds = sorted({r["seed"] for r in runs})
steps = runs[0]["steps"]

plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    }
)
fig, axes = plt.subplots(
    len(NOISES), len(N_OBS), figsize=(15, 4.6 * len(NOISES)), dpi=200, facecolor=SURFACE
)
axes = np.atleast_2d(axes)
for i, noise in enumerate(NOISES):
    for j, n in enumerate(N_OBS):
        ax = axes[i, j]
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        sel = [r for r in runs if r["noise"] == noise and r["n_obs"] == n]
        a = np.array([r["alpha_hat"] for r in sel]) / TRUE[0] - 1
        b = np.array([r["beta_hat"] for r in sel]) / TRUE[1] - 1
        # Cramér–Rao 2-sigma ellipse in relative coordinates (median bound over seeds).
        sa = np.median([r["crlb_alpha_rel_std"] for r in sel])
        sb = np.median([r["crlb_beta_rel_std"] for r in sel])
        rho = np.median([r["crlb_correlation"] for r in sel])
        cov = np.array([[sa**2, rho * sa * sb], [rho * sa * sb, sb**2]])
        vals, vecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        ax.add_patch(
            Ellipse(
                (0, 0),
                2 * 2 * np.sqrt(vals[1]),
                2 * 2 * np.sqrt(vals[0]),
                angle=angle,
                facecolor=ORANGE,
                alpha=0.18,
                edgecolor=ORANGE,
                linewidth=1.5,
            )
        )
        ax.scatter(
            a,
            b,
            color=BLUE,
            s=36,
            edgecolors=SURFACE,
            linewidths=1,
            zorder=3,
            label="estimate per seed",
        )
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.axvline(0, color=INK2, linewidth=0.8)
        lim = max(0.05, 1.2 * np.abs(np.concatenate([a, b, [2 * sa, 2 * sb]])).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        emp_rho = np.corrcoef(a, b)[0, 1] if len(a) > 2 else np.nan
        ax.set_title(
            f"noise {noise:g}, {n} observations\nbound: ±{sa:.1%} α, ±{sb:.1%} β, ρ = {rho:+.2f}; seeds: ρ = {emp_rho:+.2f}",
            loc="left",
            fontsize=10,
        )
        if i == len(NOISES) - 1:
            ax.set_xlabel("relative error of α̂")
        if j == 0:
            ax.set_ylabel("relative error of β̂")
axes[0, 0].plot([], [], color=ORANGE, linewidth=1.5, label="Cramér–Rao 2σ ellipse")
axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper left")
fig.suptitle(
    f"F02: joint identification of the KdV coefficients (α, β) from (4, 1.5), {len(seeds)} seeds, {steps} Adam steps",
    x=0.01,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
