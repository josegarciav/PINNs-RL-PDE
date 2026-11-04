"""J01 / J02 figure: the steady profiles, the grid convergence, the three scenarios' outlet
records, and the PINN's outlet against the reference for the steam step.

Usage: python experiments/J01_evaporator_reference/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j01_evaporator.png"
ref = json.loads((results / "reference.json").read_text())
runs_path = results / "runs.json"
runs = json.loads(runs_path.read_text()) if runs_path.exists() else []

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
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
fig, axes = plt.subplots(1, 4, figsize=(19, 4.6), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# (a) steady profiles, from the model itself
from pinnrl.process import FallingFilmEvaporator  # noqa: E402

ax = axes[0]
for k, (label, ev) in enumerate(
    {
        "clean, steam 355 K": FallingFilmEvaporator(),
        "fouled R_f = 2e-4": FallingFilmEvaporator(R_f=2e-4),
        "steam +5 K": FallingFilmEvaporator(T_s=360.15),
    }.items()
):
    ss = ev.steady_state()
    ax.plot(ss["z"], ss["X"], color=COLORS[k], linewidth=2, label=label)
ax.set_xlabel("z along the tube (m)")
ax.set_ylabel("solids mass fraction X")
ax.set_title("Steady profiles (feed 0.1 kg/s at X = 0.1)", loc="left", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)

# (b) grid convergence
ax = axes[1]
errs = ref["grid_convergence"]["errors"]
nz = sorted(int(k) for k in errs)
ax.loglog(
    nz,
    [errs[str(n)] for n in nz],
    "o-",
    color=COLORS[0],
    linewidth=2,
    label="|outlet X(600 s) − extrapolated|",
)
ax.loglog(
    nz,
    [errs[str(nz[0])] * nz[0] / n for n in nz],
    "--",
    color=INK2,
    linewidth=1.2,
    label="first order",
)
ax.set_xlabel("axial grid points nz")
ax.set_ylabel("error of the outlet X")
ax.set_title("Grid convergence of the dynamic solver", loc="left", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)

# (c) scenarios
ax = axes[2]
for k, (name, s) in enumerate(ref["scenarios"].items()):
    ax.plot(np.array(s["t"]) / 60, s["outlet_X"], color=COLORS[k], linewidth=2, label=name)
ax.set_xlabel("time (min)")
ax.set_ylabel("outlet X(L, t)")
ax.set_title("Outlet records of three scenarios", loc="left", fontsize=10.5, pad=8)
ax.legend(frameon=False, fontsize=8, labelcolor=INK)

# (d) PINN vs reference
ax = axes[3]
if runs:
    t = np.array(runs[0]["t"]) / 60
    ax.plot(t, runs[0]["outlet_X_ref"], color=INK, linewidth=1.6, ls="--", label="reference")
    runs = [r for r in runs if np.isfinite(r["rel_l2_X"])]  # a diverged seed is left out
    for r in runs:
        ax.plot(
            t,
            r["outlet_X_pred"],
            color=COLORS[0],
            linewidth=1.4,
            alpha=0.8,
            label=f"PINN seed {r['seed']}" if r["seed"] == 0 else None,
        )
    ax.set_xlabel("time (min)")
    ax.set_ylabel("outlet X(L, t)")
    errs = [r["rel_l2_X_minus_feed"] for r in runs]
    ax.set_title(
        f"J02: PINN outlet, steam step (rel. L2 of X − feed: {np.nanmedian(errs):.3f})",
        loc="left",
        fontsize=10.5,
        pad=8,
    )
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK)
else:
    ax.text(0.5, 0.5, "J02 not run", ha="center", va="center", transform=ax.transAxes)

fig.tight_layout()
out_png.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
