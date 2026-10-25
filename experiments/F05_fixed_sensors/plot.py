"""F05 figure: error of alpha by sensor layout and noise, against the Cramér–Rao bound.

Usage: python experiments/F05_fixed_sensors/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "f05_fixed_sensors.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
NOISES = [
    (0.01, "noise 0.01", "#eb6834"),
    (0.05, "noise 0.05", "#1baf7a"),
    (0.1, "noise 0.1", "#eda100"),
]
LAYOUTS = ["random", "1", "2", "4", "8"]
LABELS = ["random\nscatter", "1 sensor", "2 sensors", "4 sensors", "8 sensors"]
seeds = sorted({r["seed"] for r in runs})
steps = runs[0]["steps"]

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
ax_err, ax_spread, ax_layout = axes

pos = np.arange(len(LAYOUTS))
width = 0.26
for k, (noise, label, color) in enumerate(NOISES):
    med, lo, hi, spread, crlb = [], [], [], [], []
    for layout in LAYOUTS:
        sel = [r for r in runs if r["noise"] == noise and r["layout"] == layout]
        e = np.array([r["rel_error"] for r in sel])
        a = np.array([r["alpha_hat"] for r in sel])
        med.append(np.median(e))
        lo.append(np.median(e) - np.percentile(e, 25))
        hi.append(np.percentile(e, 75) - np.median(e))
        spread.append(a.std() / 0.1)
        crlb.append(np.median([r["crlb_rel_std"] for r in sel]))
    x = pos + (k - 1) * width
    ax_err.bar(
        x,
        med,
        width * 0.92,
        color=color,
        label=label,
        yerr=[lo, hi],
        capsize=2,
        error_kw={"linewidth": 1, "color": INK2},
    )
    ax_spread.plot(
        pos,
        spread,
        "o-",
        color=color,
        linewidth=2,
        markersize=7,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        label=label,
    )
    ax_spread.plot(pos, crlb, ":", color=color, linewidth=1.6)
ax_err.set_xticks(pos, LABELS)
ax_err.set_yscale("log")
ax_err.set_ylabel("relative error of α (median, IQR)")
ax_err.set_title(f"Final error by layout  ({len(seeds)} seeds)", loc="left", fontsize=12, pad=10)
ax_err.legend(frameon=False, fontsize=9)

ax_spread.plot([], [], ":", color=INK2, linewidth=1.6, label="Cramér–Rao bound")
ax_spread.set_xticks(pos, LABELS)
ax_spread.set_yscale("log")
ax_spread.set_ylabel("std of α̂ across seeds / α")
ax_spread.set_title("Spread against the bound for each layout", loc="left", fontsize=12, pad=10)
ax_spread.legend(frameon=False, fontsize=9)

# Where the information is: the squared sensitivity of u to alpha over the domain.
xx, tt = np.meshgrid(np.linspace(0, 1, 200), np.linspace(0, 1, 200), indexing="ij")
u = np.exp(-0.1 * np.pi**2 * tt) * np.sin(np.pi * xx)
info = (np.pi**2 * tt * u) ** 2
im = ax_layout.pcolormesh(tt, xx, info, cmap="Blues", shading="auto", rasterized=True)
for n_s, color in zip([1, 2, 4, 8], ["#0b0b0b", "#52514e", "#8a8985", "#b5b4b0"], strict=False):
    for p in np.arange(1, n_s + 1) / (n_s + 1):
        ax_layout.axhline(p, color=color, linewidth=1.2, linestyle="--")
ax_layout.set_xlabel("t")
ax_layout.set_ylabel("x")
ax_layout.set_title(
    "Information density (∂u/∂α)² and the probe positions", loc="left", fontsize=12, pad=10
)
fig.colorbar(im, ax=ax_layout, fraction=0.046, pad=0.03)

fig.suptitle(
    f"F05: fixed probes vs random scatter at 200 observations ({steps} Adam steps, α started at 5× the truth)",
    x=0.01,
    ha="left",
    fontsize=14,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
