"""J03 figure: the recovered fouling resistance against the truth, function and scalar arms.

Usage: python experiments/J03_fouling_estimation/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j03_fouling_estimation.png"
runs = json.loads((results / "runs.json").read_text())
first = results / "unscaled" / "runs.json"
first_runs = json.loads(first.read_text()) if first.exists() else []

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
FUNC, SCAL = "#2a78d6", "#eb6834"
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
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

ax = axes[0]
t = np.array(runs[0]["t"]) / 60
ax.plot(t, 1e4 * np.array(runs[0]["truth"]), "-", color=INK, linewidth=2.2, label="truth $R_f(t)$")
for r in runs:
    color = FUNC if r["model"] == "function" else SCAL
    ax.plot(t, 1e4 * np.array(r["profile"]), "-", color=color, linewidth=1.4, alpha=0.85)
ax.plot([], [], "-", color=FUNC, label="unknown function (3 seeds)")
ax.plot([], [], "-", color=SCAL, label="scalar control (3 seeds)")
for r in first_runs:
    if r["model"] == "function":
        ax.plot(t, 1e4 * np.array(r["profile"]), ":", color=FUNC, linewidth=1, alpha=0.6)
if first_runs:
    ax.plot([], [], ":", color=FUNC, label="function, sensors unscaled (first pass)")
ax.set_xlabel("time (min)")
ax.set_ylabel("$R_f$ ($10^{-4}$ m²K/W)")
ax.set_title("a) the recovered fouling resistance", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
labels = ["rel. L2 of $R_f(t)$", "error at the end of the window", "rel. L2 of $X - X_{in}$"]
keys = ["rf_rel_l2", "rf_end_error", "rel_l2_X_minus_feed"]
x = np.arange(len(keys))
for j, (model, color) in enumerate((("function", FUNC), ("scalar", SCAL))):
    vals = np.array([[r[k] for k in keys] for r in runs if r["model"] == model])
    ax.bar(x + (j - 0.5) * 0.36, vals.mean(0), width=0.34, color=color, label=model)
    for i in range(len(keys)):
        ax.scatter(
            np.full(vals.shape[0], x[i] + (j - 0.5) * 0.36), vals[:, i], color=INK, s=12, zorder=3
        )
ax.axhline(0.05, color=INK2, linestyle=":", linewidth=1.2)
ax.text(2.45, 0.055, "J04 bound on the end value, 5 %", fontsize=8.5, color=INK2, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_yscale("log")
ax.set_ylabel("relative error (mean of 3 seeds, dots per seed)")
ax.set_title("b) errors", loc="left")
ax.legend(frameon=False)

fig.suptitle(
    "J03: fouling estimation from outlet sensors with the steam and feed records as inputs",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
