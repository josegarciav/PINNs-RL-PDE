"""F04 figure: final error and settling step vs the initial guess, plus trajectories.

Usage: python experiments/F04_initial_guess/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f04_initial_guess.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
factors = sorted({r["guess_factor"] for r in runs})
seeds = sorted({r["seed"] for r in runs})
steps = runs[0]["steps"]
every = steps / len(runs[0]["trajectory"])

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
fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax_err, ax_settle, ax_traj = axes

err = np.array([[r["rel_error"] for r in runs if r["guess_factor"] == f] for f in factors]).T
med = np.median(err, axis=0)
q25, q75 = np.percentile(err, [25, 75], axis=0)
ax_err.fill_between(factors, q25, q75, color=BLUE, alpha=0.14, linewidth=0)
ax_err.plot(factors, med, color=BLUE, linewidth=2)
ax_err.set_xscale("log")
ax_err.set_yscale("log")
ax_err.set_xlabel("initial guess / true α")
ax_err.set_ylabel("relative error of α at the last step (median, IQR)")
ax_err.set_title(f"Final error  ({len(seeds)} seeds)", loc="left", fontsize=12, pad=10)

for f in factors:
    sel = [r for r in runs if r["guess_factor"] == f]
    settled = [r["settle_step"] for r in sel if r["settle_step"] is not None]
    n_fail = len(sel) - len(settled)
    ax_settle.scatter(
        [f] * len(settled), settled, color=BLUE, s=22, alpha=0.7, edgecolors=SURFACE, linewidths=0.8
    )
    if n_fail:
        ax_settle.scatter([f], [steps], marker="x", color=ORANGE, s=60, linewidths=2)
never = sorted({r["seed"] for r in runs if r["settle_step"] is None})
if never:
    ax_settle.annotate(
        f"seeds {never} never stay within 5 %, whatever the guess",
        (factors[0], steps),
        textcoords="offset points",
        xytext=(0, 8),
        ha="left",
        fontsize=9,
        color=ORANGE,
    )
ax_settle.set_xscale("log")
ax_settle.set_ylim(0, steps * 1.12)
ax_settle.set_xlabel("initial guess / true α")
ax_settle.set_ylabel("step after which α stays within 5 % of the truth")
ax_settle.set_title("Settling step per seed", loc="left", fontsize=12, pad=10)

cmap = plt.get_cmap("viridis")
t = np.arange(1, len(runs[0]["trajectory"]) + 1) * every
for i, f in enumerate(factors):
    traj = np.array([r["trajectory"] for r in runs if r["guess_factor"] == f]) / 0.1
    ax_traj.plot(
        t,
        np.median(traj, axis=0),
        color=cmap(i / (len(factors) - 1)),
        linewidth=1.8,
        label=f"{f:.2g}×",
    )
ax_traj.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
ax_traj.set_yscale("log")
ax_traj.set_xlabel("Adam step")
ax_traj.set_ylabel("α estimate / true α (median over seeds)")
ax_traj.set_title("Trajectories by initial guess", loc="left", fontsize=12, pad=10)
ax_traj.legend(frameon=False, fontsize=8, ncol=2, title="guess", title_fontsize=8.5)

fig.suptitle(
    f"F04: sensitivity of the recovered heat diffusivity to the initial guess "
    f"({steps} Adam steps, 200 observations, noise 0.05)",
    x=0.01,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
