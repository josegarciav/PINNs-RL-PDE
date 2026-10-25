"""F01 figure: error of the recovered alpha vs observation count and noise.

Usage: python experiments/F01_heat_alpha_recovery/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f01_heat_alpha.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NOISES = [  # categorical slots 1-4, fixed order
    (0.0, "noise 0", "#2a78d6"),
    (0.01, "noise 0.01", "#eb6834"),
    (0.05, "noise 0.05", "#1baf7a"),
    (0.1, "noise 0.1", "#eda100"),
]
N_OBS = sorted({r["n_obs"] for r in runs})
TRAJ_N_OBS = 50
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


def band(ax, x, values, color, label, ls="-"):
    values = np.asarray(values)
    med = np.median(values, axis=0)
    q25, q75 = np.percentile(values, [25, 75], axis=0)
    ax.fill_between(x, q25, q75, color=color, alpha=0.14, linewidth=0)
    ax.plot(x, med, ls, color=color, linewidth=2, label=label)
    return med


fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

ax_alpha, ax_field, ax_traj = axes
for noise, label, color in NOISES:
    sel = [r for r in runs if r["noise"] == noise]
    if not sel:
        continue
    by_n = {n: [r for r in sel if r["n_obs"] == n] for n in N_OBS}
    alpha_err = np.array([[r["rel_error"] for r in by_n[n]] for n in N_OBS]).T  # seeds x n_obs
    field_err = np.array([[r["l2_error"] for r in by_n[n]] for n in N_OBS]).T
    band(ax_alpha, N_OBS, alpha_err, color, label)
    band(ax_field, N_OBS, field_err, color, label)
    if noise > 0:
        crlb = np.array([np.median([r["crlb_rel_std"] for r in by_n[n]]) for n in N_OBS])
        ax_alpha.plot(N_OBS, crlb, ":", color=color, linewidth=1.6)
    traj = np.array([r["trajectory"] for r in by_n[TRAJ_N_OBS]])
    t = np.arange(1, traj.shape[1] + 1) * every
    band(ax_traj, t, traj / 0.1, color, label)

ax_alpha.plot([], [], ":", color=INK2, linewidth=1.6, label="Cramér–Rao bound (std)")
ax_alpha.set_xscale("log")
ax_alpha.set_yscale("log")
ax_alpha.set_xlabel("observations")
ax_alpha.set_ylabel("relative error of α (median, IQR band)")
ax_alpha.set_title(f"Recovered diffusivity  ({len(seeds)} seeds)", loc="left", fontsize=12, pad=10)
ax_alpha.legend(frameon=False, fontsize=9, labelcolor=INK)

ax_field.set_xscale("log")
ax_field.set_yscale("log")
ax_field.set_xlabel("observations")
ax_field.set_ylabel("relative L2 error of u (median, IQR band)")
ax_field.set_title("Field error of the same runs", loc="left", fontsize=12, pad=10)

ax_traj.axhline(1.0, color=INK2, linewidth=1, linestyle="--")
ax_traj.set_xlabel("Adam step")
ax_traj.set_ylabel("α estimate / true α")
ax_traj.set_title(
    f"Trajectory from 5× the truth, {TRAJ_N_OBS} observations", loc="left", fontsize=12, pad=10
)
ax_traj.set_ylim(0.6, 1.6)

fig.suptitle(
    f"F01: heat diffusivity α identified from noisy observations ({steps} Adam steps, 4×64 MLP)",
    x=0.01,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
