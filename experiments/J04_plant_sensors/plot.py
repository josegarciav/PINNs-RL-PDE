"""J04 figure: Cramér–Rao bounds on the fouling modes per layout and nuisance, and the profiles.

Usage: python experiments/J04_plant_sensors/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j04_plant_sensors.png"
r = json.loads((results / "summary.json").read_text())
R_END = 2e-4

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9", "#d1427f"]
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

fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, axis="y")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def grouped(ax, cases, title):
    n_modes = r["modes"]
    width = 0.8 / len(cases)
    for i, (label, std) in enumerate(cases):
        std = np.asarray(std[:n_modes], dtype=float)
        x = np.arange(n_modes) + (i - (len(cases) - 1) / 2) * width
        ax.bar(
            x, np.minimum(std, 10), width=width * 0.95, color=PALETTE[i % len(PALETTE)], label=label
        )
    ax.set_yscale("log")
    ax.set_xticks(np.arange(n_modes))
    ax.set_xticklabels([f"$c_{k}$" for k in range(n_modes)])
    ax.set_xlabel("Legendre mode of $R_f(t)$")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=9)


grouped(
    axes[0],
    [(name, f["std"]) for name, f in r["layouts"].items()],
    "a) sensor layout (inputs known)",
)
axes[0].set_ylabel("Cramér–Rao std of the mode, in units of the mean fouling")
nuis = r["nuisances"]
LABELS = {
    "T_s_bias": "+ steam bias",
    "W_in_bias": "+ feed-flow bias",
    "X_in_bias": "+ feed-solids bias",
    "U": "+ U unknown",
}
cases = [("inputs known", r["layouts"]["outlet X + W"]["std"])]
for key, f in nuis.items():
    if key in LABELS:
        cases.append((LABELS[key], f["std"]))
    elif "drift only" in key:
        cases.append(("+ all four, drift only", [np.nan] + f["std"]))
    else:
        cases.append(("+ the three biases", f["std"]))
grouped(axes[1], cases, "b) unknown nuisances (outlet X + W)")
axes[1].set_ylim(top=20)

ax = axes[2]
ax.grid(True, which="major", color=GRID, linewidth=0.8)
p = r["profile"]
g = np.array(p["grid_rel"])
fine = np.linspace(g[0], g[-1], 200)
for d in p["datasets"]:
    fx, fr = np.array(d["fixed"]), np.array(d["free"])
    ax.plot(g, fx - fx.min(), "-", color=PALETTE[0], linewidth=1.2, alpha=0.7)
    ax.plot(g, fr - fr.min(), "-", color=PALETTE[1], linewidth=1.2, alpha=0.7)
ax.plot(
    fine,
    p["F_rr"] * ((fine - 1) * R_END) ** 2,
    "--",
    color=PALETTE[0],
    linewidth=2,
    label="steam known: profiles and Fisher parabola",
)
ax.plot(
    fine,
    p["F_profile_rr"] * ((fine - 1) * R_END) ** 2,
    "--",
    color=PALETTE[1],
    linewidth=2,
    label="steam bias re-fitted: profiles and parabola",
)
ax.axhline(3.84, color=INK2, linewidth=1, linestyle=":")
ax.text(g[0], 4.6, "Δ = 3.84 (95 %)", fontsize=9, color=INK2)
ax.set_yscale("symlog", linthresh=10)
ax.set_ylim(0, 1200)
ax.set_xlabel("drift end value $r_{end}$ / truth")
ax.set_ylabel("misfit above its minimum (5 data sets)")
ax.set_title("c) profile likelihood of the drift", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper center")

fig.suptitle(
    "J04: what outlet sensors say about the fouling resistance of the evaporator (reference solver, finite differences)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
