"""K09 figure: ODE spectra against the PINN error, and the evaporator's method-of-lines spectrum.

Usage: python experiments/K09_linearized_stability/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "k09_linearized_stability.png"
r = json.loads((results / "summary.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9", "#d1427f", "#3aa8c7", "#7a7a7a"]
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
fig, axes = plt.subplots(1, 3, figsize=(19, 5.4), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# a) PINN error vs the perturbation growth over the window
ax = axes[0]
pinn = r["pinn"]
for i, o in enumerate(r["ode"]):
    errs = [p["rel_l2_mean"] for p in pinn if p["system"] == o["system"]]
    if not errs:
        continue
    x = o["fast_units"]
    ax.scatter([x] * len(errs), errs, color=PALETTE[i % len(PALETTE)], s=36, zorder=3)
    ax.annotate(
        o["system"].replace("_", " "),
        (x, float(np.median(errs))),
        xytext=(6, 4),
        textcoords="offset points",
        fontsize=9,
    )
ax.set_yscale("log")
ax.set_xlabel("fast units in the window, $T \\max |\\lambda|$")
ax.set_ylabel("PINN relative L2 error (2 seeds, mean over fields)")
ax.set_title("a) catalogue ODE systems: PINN error vs the window's fast units", loc="left")

# b) evaporator eigenvalues in the complex plane for two grids
ax = axes[1]
for j, nz in enumerate((51, 201)):
    e = next(x for x in r["evaporator"] if x["nz"] == nz and x["fouling"] == 0.0)
    ax.scatter(
        e["eigenvalues_real"],
        e["eigenvalues_imag"],
        s=8,
        color=PALETTE[j],
        alpha=0.7,
        label=f"$n_z$ = {nz}",
        zorder=3,
    )
    ax.axvline(-e["transport_rate"], color=PALETTE[j], linestyle=":", linewidth=1.2)
e = next(x for x in r["evaporator"] if x["nz"] == 201 and x["fouling"] == 0.0)
ax.axvline(
    -e["source_rate_max"],
    color=INK2,
    linestyle="--",
    linewidth=1,
    label="source rate (no transport)",
)
ax.axvline(0, color=INK, linewidth=0.8)
ax.set_xlabel("Re λ (1/s); dotted: $-v/\\Delta z$")
ax.set_ylabel("Im λ (1/s)")
ax.set_title("b) evaporator, method of lines: the spectrum", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

# c) rates vs grid
ax = axes[2]
ev = [x for x in r["evaporator"] if x["fouling"] == 0.0]
nzs = [x["nz"] for x in ev]
ax.plot(
    nzs, [x["transport_rate"] for x in ev], "-o", color=PALETTE[0], label="transport $v/\\Delta z$"
)
ax.plot(
    nzs,
    [x["fastest_rate"] for x in ev],
    "--s",
    color=PALETTE[1],
    label="fastest computed eigenvalue",
)
ax.plot(
    nzs,
    [-x["max_real_part"] for x in ev],
    "--^",
    color=PALETTE[2],
    label="slowest computed eigenvalue",
)
ax.plot(
    nzs, [x["source_rate_max"] for x in ev], "-d", color=PALETTE[3], label="source rate (local)"
)
ax.axhline(1 / ev[0]["residence_time_s"], color=INK2, linestyle=":", linewidth=1.2)
ax.text(nzs[0], 1.15 / ev[0]["residence_time_s"], "1 / residence time", fontsize=9, color=INK2)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("axial grid points $n_z$")
ax.set_ylabel("rate (1/s)")
ax.set_title("c) grid-set and physical rates", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle(
    "K09: linearized stability of the catalogue's ODE systems and of the evaporator model",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
