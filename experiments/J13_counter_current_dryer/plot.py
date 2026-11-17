"""J13 figure: the counter-current tower, its solves, the PINN's forward solve and the identification.

Usage: python experiments/J13_counter_current_dryer/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run as J  # noqa: E402

results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j13_counter_current_dryer.png"
d = json.loads((results / "j13.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, AMBER, PURPLE, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#8e5bd9",
    "#d6453d",
)
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
fig, axes = plt.subplots(2, 2, figsize=(17, 11.5), dpi=200, facecolor=SURFACE)
for ax in axes.ravel():
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

# (a) steady profiles: counter-current against co-current
ax = axes[0, 0]
cc, co = d["steady_profile"], d["co_current_profile"]
ax.plot(cc["z"], np.array(cc["T_a"]) - 273.15, color=ORANGE, lw=2.2, label="air, counter-current")
ax.plot(cc["z"], np.array(cc["T_p"]) - 273.15, color=RED, lw=2.2, label="droplets, counter-current")
ax.plot(
    co["z"],
    np.array(co["T_a"]) - 273.15,
    color=ORANGE,
    lw=1.4,
    ls="--",
    label="air, co-current (J11)",
)
ax.plot(
    co["z"], np.array(co["T_p"]) - 273.15, color=RED, lw=1.4, ls="--", label="droplets, co-current"
)
ax.set_xlabel("height below the atomiser, m")
ax.set_ylabel("temperature, °C")
ax2 = ax.twinx()
ax2.plot(cc["z"], cc["X"], color=BLUE, lw=2.2, label="moisture, counter-current")
ax2.plot(co["z"], co["X"], color=BLUE, lw=1.4, ls="--", label="moisture, co-current")
ax2.set_ylabel("droplet moisture, dry basis", color=BLUE)
ax2.tick_params(axis="y", colors=BLUE)
for side in ("top",):
    ax2.spines[side].set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9, frameon=False)
ax.set_title(
    "(a) Steady profiles: the air enters at the bottom, the dry powder meets the hottest air",
    loc="left",
    fontsize=11,
)

# (b) the step response and the PINN's forward solve
ax = axes[0, 1]
if d.get("forward"):
    best = min(d["forward"], key=lambda r: r["rel_l2"]["X"])
    t = np.array(best["t"])
    for k, z in enumerate(J.PROBES):
        ax.plot(t, np.array(best["T_air_ref"][k]) - 273.15, color=ORANGE, lw=1.6, alpha=0.9)
        ax.plot(t, np.array(best["T_air_pinn"][k]) - 273.15, color=INK, lw=1.0, ls=":")
        ax.annotate(
            f"{z} m",
            (t[-1], best["T_air_ref"][k][-1] - 273.15),
            xytext=(4, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            va="center",
        )
    ax.plot([], [], color=ORANGE, lw=1.6, label="reference (401 cells)")
    ax.plot(
        [],
        [],
        color=INK,
        lw=1.0,
        ls=":",
        label=f"PINN forward solve, best of {len(d['forward'])} seeds",
    )
    ax.set_xlabel("time, s")
    ax.set_ylabel("air temperature at the thermocouples, °C")
    axb = ax.twinx()
    axb.plot(t, best["X_out_ref"], color=BLUE, lw=1.8, label="powder moisture, reference")
    axb.plot(t, best["X_out_pinn"], color=BLUE, lw=1.0, ls=":", label="powder moisture, PINN")
    axb.set_ylabel("powder moisture leaving, dry basis", color=BLUE)
    axb.tick_params(axis="y", colors=BLUE)
    axb.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = axb.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9, frameon=False)
    errs = {k: np.median([r["rel_l2"][k] for r in d["forward"]]) for k in ("X", "T", "Y", "Ta")}
    ax.set_title(
        "(b) Step response: the PINN forward solve, median rel. L2 "
        + ", ".join(f"{k} {100 * v:.0f} %" for k, v in errs.items()),
        loc="left",
        fontsize=11,
    )

# (c) the cost of the solves and of the estimators
ax = axes[1, 0]
c = d["cost"]
items = [
    ("co-current steady (J11)", c["co_current_steady_ms"] / 1e3, GRID),
    ("counter-current steady (shooting)", c["counter_steady_ms"] / 1e3, ORANGE),
]
for nz in ("51", "201", "401"):
    g = c["grids"][nz]
    items.append((f"transient, {nz} cells, step, rtol 1e-6", g["step_rtol1e-06_s"], BLUE))
    items.append((f"transient, {nz} cells, record, rtol 1e-8", g["record_rtol1e-08_s"], PURPLE))
if d.get("lsq"):
    items.append(
        (
            "least squares, 3 parameters (per fit)",
            np.median([r["seconds"] for r in d["lsq"]]),
            GREEN,
        )
    )
if d.get("forward"):
    items.append(
        ("PINN forward solve (per run)", np.median([r["train_time_s"] for r in d["forward"]]), RED)
    )
if d.get("pinn_identify"):
    items.append(
        (
            "PINN twin, 3 parameters (per run)",
            np.median([r["train_time_s"] for r in d["pinn_identify"]]),
            RED,
        )
    )
y = np.arange(len(items))
ax.barh(y, [v for _, v, _ in items], color=[col for _, _, col in items], height=0.7)
ax.set_yticks(y)
ax.set_yticklabels([n for n, _, _ in items], fontsize=9)
ax.invert_yaxis()
ax.set_xscale("log")
ax.set_xlabel("seconds (log)")
for yi, (_, v, _) in zip(y, items, strict=True):
    ax.annotate(
        f"{v:.3g} s" if v >= 1 else f"{1e3 * v:.0f} ms",
        (v, yi),
        xytext=(4, 0),
        textcoords="offset points",
        va="center",
        fontsize=8,
        color=INK2,
    )
ax.set_title(
    "(c) What a solve costs on this unit, and what the estimators cost", loc="left", fontsize=11
)

# (d) the identification
ax = axes[1, 1]
if d.get("lsq") and d.get("pinn_identify"):
    lin = d["linear_prediction"]["std"]
    for j, name in enumerate(J.NAMES):
        truth = J.TRUTH[name]
        lsq = np.array([r["estimate"][name] for r in d["lsq"]]) - truth
        pinn = np.array([r["estimate"][name] for r in d["pinn_identify"]]) - truth
        ax.axhline(0, color=INK2, lw=0.8)
        ax.errorbar(
            [j - 0.2],
            [0.0],
            yerr=[[2 * lin[name]]],
            fmt="none",
            ecolor=INK2,
            elinewidth=6,
            alpha=0.25,
            capsize=0,
        )
        ax.scatter(
            np.full(lsq.size, j - 0.2),
            lsq,
            color=GREEN,
            s=40,
            zorder=3,
            label="least squares through the reference" if j == 0 else None,
        )
        ax.scatter(
            np.full(pinn.size, j + 0.2),
            pinn,
            color=RED,
            s=40,
            marker="s",
            zorder=3,
            label="PINN twin" if j == 0 else None,
        )
        ax.annotate(
            f"start {J.GUESS[name] - truth:+.1f}",
            (j + 0.2, J.GUESS[name] - truth),
            xytext=(10, 0),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
            va="center",
        )
        ax.scatter([j + 0.2], [J.GUESS[name] - truth], marker="_", color=INK2, s=120)
    ax.set_xticks(range(len(J.NAMES)))
    ax.set_xticklabels(
        ["critical moisture $X_{cr}$ (1.0)", "exponent $n$ (1.5)", "wall loss $U_w$ (1.0)"]
    )
    ax.set_ylabel("estimate − truth")
    ax.plot([], [], color=INK2, lw=6, alpha=0.25, label="linear prediction, ±2σ")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    acc = sum(r["accepted"] for r in d["pinn_identify"])
    ax.set_title(
        f"(d) Three parameters from one transient: the misfit test accepts {len(d['lsq'])} of "
        f"{len(d['lsq'])} fits and {acc} of {len(d['pinn_identify'])} PINN runs (±2σ bars 0.01–0.03)",
        loc="left",
        fontsize=11,
    )

fig.suptitle(
    "J13: the counter-current spray dryer, the first unit whose steady state is a boundary-value problem",
    fontsize=14,
    x=0.01,
    ha="left",
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
