"""J08 figure: the stickiness margin over the operating window and four ways to estimate it.

Usage: python experiments/J08_stickiness/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j08_stickiness.png"
d = json.loads((results / "j08.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, AMBER, PURPLE = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9"
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
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), dpi=200, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# a) the map
ax = axes[0]
m = d["map"]
T = np.array(m["T_in"]) - 273.15
F = 1e3 * np.array(m["solids_flow"])
Z = np.array(m["margin_out"])
cf = ax.contourf(F, T, Z, levels=np.arange(0, 91, 5), cmap="Blues")
ax.contour(F, T, Z, levels=[d["limit"]], colors=[ORANGE], linewidths=2.5)
cb = fig.colorbar(cf, ax=ax)
cb.set_label("outlet margin $T_p - T_g(X)$ (K)")
for name, (Tin, Fs) in d["points"].items():
    ax.plot(1e3 * Fs, Tin - 273.15, "o", color=INK, markersize=7)
    ax.annotate(
        name, (1e3 * Fs, Tin - 273.15), xytext=(6, 6), textcoords="offset points", fontsize=9
    )
ax.plot([], [], color=ORANGE, linewidth=2.5, label=f"sticky limit, {d['limit']:.0f} K")
ax.legend(frameon=False, fontsize=9, loc="lower left")
ax.set_xlabel("solids flow (g/s)")
ax.set_ylabel("inlet air temperature (°C)")
ax.set_title("a) the margin over the operating window", loc="left")

# b) the sensors
ax = axes[1]
ax.grid(True, axis="y", which="both", color=GRID, linewidth=0.8)
rows = d["sensors"]
x = np.arange(len(rows))
for j, (key, color, label) in enumerate(
    (
        ("balance", AMBER, "water balance of the air"),
        ("lab", INK2, "lab assay of the powder (hours late)"),
        ("twin", BLUE, "twin, from the campaign alone"),
        ("twin_online", GREEN, "twin + the point's online records"),
    )
):
    vals = [r[key] for r in rows]
    ax.bar(x + (j - 1.5) * 0.2, vals, width=0.18, color=color, label=label)
    for xi, v in zip(x + (j - 1.5) * 0.2, vals, strict=True):
        ax.text(
            xi, v * 1.08, f"{v:.0f}" if v >= 10 else f"{v:.1f}", ha="center", fontsize=8, color=INK2
        )
ax.set_yscale("log")
ax.set_ylim(0.5, 150)
ax.set_xticks(x)
ax.set_xticklabels([f"{r['point']}\n(margin {r['margin_out']:.0f} K)" for r in rows], fontsize=9)
ax.set_ylabel("standard deviation of the estimated margin (K)")
ax.set_title("b) four estimates of the outlet margin", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper right")

# c) validation
ax = axes[2]
ax.grid(True, color=GRID, linewidth=0.8)
bins = np.linspace(-5, 5, 21)
xs = np.linspace(-5, 5, 300)
for v, color in zip(d["validation"], (GREEN, BLUE, ORANGE, PURPLE), strict=True):
    s = v["predicted_out"]
    ax.hist(v["errors_out"], bins=bins, density=True, histtype="step", color=color, linewidth=1.6)
    ax.plot(
        xs,
        np.exp(-0.5 * (xs / s) ** 2) / (s * np.sqrt(2 * np.pi)),
        color=color,
        linewidth=1.2,
        linestyle="--",
        label=f"{v['point']}: predicted {s:.1f} K, found {v['rmse_out']:.1f} K",
    )
ax.set_xlabel("error of the twin's outlet margin (K), 40 fitted campaigns")
ax.set_ylabel("density")
ax.set_title("c) the twin's prediction checked", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_ylim(0, 0.75)

fig.suptitle(
    "J08: a soft sensor for the stickiness margin of the powder leaving the dryer",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
