"""K02 figure: the Cramér-Rao bound of a layout vs probe count and position, the KdV conditioning
vs probe position, and the identified alpha against the bound of its layout.

Usage: python experiments/K02_sensor_layout/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k02_sensor_layout.png"
fisher = json.loads((results / "fisher.json").read_text())
runs_path = results / "runs.json"
runs = json.loads(runs_path.read_text()) if runs_path.exists() else []

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
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

fig, axes = plt.subplots(1, 4, figsize=(19, 4.8), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# (a) heat: one probe swept over x0, and the sensor kinds
ax = axes[0]
pos = fisher["heat"]["position"]
ax.plot(
    [r["x0"] for r in pos],
    [r["cr_rel_std"] for r in pos],
    color=BLUE,
    linewidth=2,
    label="one probe at x0",
)
for row, color in zip(fisher["heat"]["kinds"], (BLUE, BLUE, ORANGE, GREEN, YELLOW), strict=True):
    if row["layout"].startswith("probe at"):
        continue
    ax.axhline(row["cr_rel_std"], color=color, linewidth=1.4, ls="--", label=row["layout"])
ax.set_yscale("log")
ax.set_xlabel("probe position x0")
ax.set_ylabel("Cramér–Rao relative std of alpha")
ax.set_title(
    "Heat: where a probe informs alpha (50 readings, noise 0.05)", loc="left", fontsize=11, pad=8
)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)

# (b) KdV: one probe swept, condition number
ax = axes[1]
pos = fisher["kdv"]["position"]
ax.plot([r["x0"] for r in pos], [r["condition_number"] for r in pos], color=ORANGE, linewidth=2)
ax.axvspan(0, 2, color=GREEN, alpha=0.12, linewidth=0, label="soliton path x = t, t in [0, 2]")
ax.set_yscale("log")
ax.set_xlabel("probe position x0")
ax.set_ylabel("condition number of the Fisher matrix")
ax.set_title(
    "KdV (alpha, beta): conditioning vs one probe's position", loc="left", fontsize=11, pad=8
)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)

# (c) count: heat bound and KdV condition number vs number of probes
ax = axes[2]
hc = fisher["heat"]["count"]
ax.plot(
    [r["n_sensors"] for r in hc],
    [r["cr_rel_std"] for r in hc],
    "o-",
    color=BLUE,
    linewidth=2,
    label="heat: CR rel. std of alpha",
)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("evenly spaced probes")
ax.set_ylabel("Cramér–Rao relative std", color=BLUE)
ax2 = ax.twinx()
ax2.set_facecolor(SURFACE)
kc = fisher["kdv"]["count"]
ax2.plot(
    [r["n_sensors"] for r in kc],
    [r["condition_number"] for r in kc],
    "s--",
    color=ORANGE,
    linewidth=2,
    label="KdV: condition number",
)
ax2.set_yscale("log")
ax2.set_ylabel("condition number", color=ORANGE)
for side in ("top",):
    ax2.spines[side].set_visible(False)
ax.set_title("Evenly spaced probes: more is not monotone", loc="left", fontsize=11, pad=8)
lines = ax.get_legend_handles_labels()
lines2 = ax2.get_legend_handles_labels()
ax.legend(
    lines[0] + lines2[0],
    lines[1] + lines2[1],
    frameon=False,
    fontsize=8.5,
    labelcolor=INK,
    loc="upper right",
)

# (d) training: identified alpha vs the bound of its layout
ax = axes[3]
if runs:
    counts = sorted({r["n_sensors"] for r in runs})
    err = np.array([[r["rel_error"] for r in runs if r["n_sensors"] == n] for n in counts])
    spread = np.array(
        [np.std([r["alpha_hat"] for r in runs if r["n_sensors"] == n]) / 0.1 for n in counts]
    )
    cr = np.array(
        [np.median([r["crlb_rel_std"] for r in runs if r["n_sensors"] == n]) for n in counts]
    )
    med = np.median(err, axis=1)
    q25, q75 = np.percentile(err, [25, 75], axis=1)
    ax.errorbar(
        counts,
        med,
        yerr=[med - q25, q75 - med],
        fmt="o-",
        color=BLUE,
        linewidth=2,
        capsize=3,
        label="|error| of alpha (median, IQR)",
    )
    ax.plot(counts, spread, "s-", color=GREEN, linewidth=1.6, label="spread across seeds")
    ax.plot(counts, cr, "--", color=INK2, linewidth=1.6, label="Cramér–Rao bound of the layout")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("evenly spaced probes")
    ax.set_ylabel("relative error of alpha")
    ax.set_title(
        f"Heat: alpha identified from the probes ({err.shape[1]} seeds)",
        loc="left",
        fontsize=11,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
else:
    ax.text(0.5, 0.5, "training part not run", ha="center", va="center", transform=ax.transAxes)

fig.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
