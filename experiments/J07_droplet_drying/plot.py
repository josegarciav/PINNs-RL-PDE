"""J07 figure: the droplet's drying record and the identified drying-rate closure.

Usage: python experiments/J07_droplet_drying/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import NOISE_M_REL, NOISE_T, T_MAX, plant, records  # noqa: E402

results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j07_droplet_drying.png"
runs = json.loads((results / "runs.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = {"pinn": "#2a78d6", "cdc": "#1baf7a", "linear": "#eb6834", "T": "#eda100"}
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
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# a) the record
dr = plant()
times, m_rec, T_rec, ref = records(0)
ax = axes[0]
ax.plot(ref.t, ref.m / dr.m0, "-", color=PALETTE["pinn"], linewidth=2, label="mass / initial mass")
ax.scatter(times, m_rec / dr.m0, s=6, color=PALETTE["pinn"], alpha=0.5)
ax.set_xlabel("time (s)")
ax.set_ylabel("droplet mass / initial mass", color=PALETTE["pinn"])
ax2 = ax.twinx()
ax2.plot(ref.t, ref.T - 273.15, "-", color=PALETTE["T"], linewidth=2, label="temperature")
ax2.scatter(times, T_rec - 273.15, s=6, color=PALETTE["T"], alpha=0.5)
ax2.set_ylabel("droplet temperature (°C)", color=PALETTE["T"])
for side in ("top",):
    ax2.spines[side].set_visible(False)
ax.set_title(
    f"a) the record: balance (±{100 * NOISE_M_REL:g} %) and thermocouple (±{NOISE_T:g} K), every second",
    loc="left",
    fontsize=10.5,
)

# b) the closure
ax = axes[1]
X = np.linspace(dr.X_eq, dr.X0, 400)
ax.plot(X, dr.drying_curve(X), "-", color=INK, linewidth=2.4, label="plant's drying curve $f(X)$")
labels = {
    "closure": ("pinn", "PINN, unknown function (3 seeds)"),
    "lsq_cdc": ("cdc", "least squares, right form ($n$, $X_{cr}$)"),
    "lsq_linear": ("linear", "least squares, linear form"),
}
seen = set()
for r in runs:
    if r["part"] not in labels:
        continue
    key, label = labels[r["part"]]
    ax.plot(
        r["X_grid"],
        r["f_hat"],
        "--" if key != "pinn" else "-",
        color=PALETTE[key],
        linewidth=1.4,
        alpha=0.85,
        label=label if key not in seen else None,
    )
    seen.add(key)
ax.set_xlabel("dry-basis moisture $X$")
ax.set_ylabel("drying-rate closure $f(X)$")
ax.set_xlim(0, dr.X0)
ax.set_title("b) the identified closure", loc="left")
ax.legend(frameon=False, fontsize=9, loc="lower right")

# c) errors
ax = axes[2]
groups = [("closure", "PINN"), ("lsq_cdc", "LSQ, right form"), ("lsq_linear", "LSQ, linear form")]
vals = [[r["closure_rel_l2"] for r in runs if r["part"] == p] for p, _ in groups]
x = np.arange(len(groups))
ax.bar(
    x,
    [np.mean(v) if v else np.nan for v in vals],
    color=[PALETTE[labels[p][0]] for p, _ in groups],
    width=0.6,
)
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), x[i]), v, color=INK, s=14, zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([g[1] for g in groups])
ax.set_yscale("log")
ax.set_ylabel("relative L2 error of $f(X)$ on $[X_{eq}, X_0]$")
fw = [r for r in runs if r["part"] == "forward"]
if fw:
    ax.set_title(
        f"c) closure error (forward PINN: X to {np.mean([r['rel_l2_X'] for r in fw]):.1%})",
        loc="left",
    )
else:
    ax.set_title("c) closure error", loc="left")

fig.suptitle(
    f"J07: a drying droplet over {T_MAX:g} s and its drying-rate closure identified from mass and temperature records",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
