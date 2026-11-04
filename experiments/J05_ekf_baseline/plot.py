"""J05 figure: the Kalman filter and smoother on the lumped evaporator against the PINN twin.

Usage: python experiments/J05_ekf_baseline/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j05_ekf_baseline.png"
runs = json.loads((results / "runs.json").read_text())
j03 = HERE.parent / "J03_fouling_estimation" / "results" / "runs.json"
pinn = [r for r in json.loads(j03.read_text()) if r["model"] == "function"] if j03.exists() else []
R_END, T_MAX = 2e-4, 1200.0

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8e5bd9"]
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

ax = axes[0]
sel = [r for r in runs if r["nz"] == 11 and r["sigma_rw"] == 1e-6 and not r["mismatch"]]
t = np.array(sel[0]["t"]) / 60
ax.plot(
    t, 1e4 * R_END * np.array(sel[0]["t"]) / T_MAX, "-", color=INK, linewidth=2.2, label="truth"
)
for r in sel:
    f = 1e4 * np.array(r["profile_filter"])
    s = 1e4 * np.array(r["std_filter"])
    ax.fill_between(t, f - 2 * s, f + 2 * s, color=PALETTE[0], alpha=0.08, linewidth=0)
    ax.plot(t, f, "-", color=PALETTE[0], linewidth=1.3, alpha=0.9)
    ax.plot(
        t, 1e4 * np.array(r["profile_smoother"]), "--", color=PALETTE[2], linewidth=1.2, alpha=0.9
    )
for r in pinn:
    ax.plot(
        np.array(r["t"]) / 60,
        1e4 * np.array(r["profile"]),
        ":",
        color=PALETTE[1],
        linewidth=1.2,
        alpha=0.9,
    )
ax.plot([], [], "-", color=PALETTE[0], label="EKF, 11 cells, $\\sigma_{rw} = 10^{-6}$ (±2σ band)")
ax.plot([], [], "--", color=PALETTE[2], label="RTS smoother")
if pinn:
    ax.plot([], [], ":", color=PALETTE[1], label="PINN twin (J03)")
ax.set_xlabel("time (min)")
ax.set_ylabel("$R_f$ ($10^{-4}$ m²K/W)")
ax.set_ylim(-0.2, 3.2)
ax.set_title("a) the estimates, 3 noise seeds", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

ax = axes[1]
sigmas = sorted({r["sigma_rw"] for r in runs})
for i, nz in enumerate(sorted({r["nz"] for r in runs})):
    f = [
        np.mean(
            [
                r["filter"]["rel_l2"]
                for r in runs
                if r["nz"] == nz and r["sigma_rw"] == s and not r["mismatch"]
            ]
        )
        for s in sigmas
    ]
    sm = [
        np.mean(
            [
                r["smoother"]["rel_l2"]
                for r in runs
                if r["nz"] == nz and r["sigma_rw"] == s and not r["mismatch"]
            ]
        )
        for s in sigmas
    ]
    ax.plot(sigmas, f, "-o", color=PALETTE[i], label=f"filter, {nz} cells")
    ax.plot(sigmas, sm, "--s", color=PALETTE[i], alpha=0.7, markersize=4)
if pinn:
    lo, hi = min(r["rf_rel_l2"] for r in pinn), max(r["rf_rel_l2"] for r in pinn)
    ax.axhspan(lo, hi, color=PALETTE[1], alpha=0.15, linewidth=0)
    ax.text(sigmas[-1], lo * 0.9, "PINN twin (J03), 3 seeds", fontsize=9, color=PALETTE[1], ha="right", va="top")
ax.plot([], [], "--s", color=INK2, markersize=4, label="smoother (dashed)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("random-walk intensity of $R_f$, $\\sigma_{rw}$ (m²K/W per √s)")
ax.set_ylabel("rel. L2 of $R_f(t)$ after a 2-min warm-up")
ax.set_title("b) tuning: model size and process noise", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper center")

ax = axes[2]
labels, vals, cols = [], [], []
base = [r for r in runs if r["nz"] == 11 and r["sigma_rw"] == 1e-6 and not r["mismatch"]]
labels.append("EKF, inputs exact")
vals.append([r["filter"]["end_error"] for r in base])
cols.append(PALETTE[0])
for label in ("steam +0.3 K", "feed +1 %"):
    labels.append(f"EKF, {label}")
    vals.append([r["filter"]["end_error"] for r in runs if r.get("label") == label])
    cols.append(PALETTE[3])
if pinn:
    labels.append("PINN twin")
    vals.append([r["rf_end_error"] for r in pinn])
    cols.append(PALETTE[1])
x = np.arange(len(labels))
ax.bar(x, [np.mean(v) for v in vals], color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.scatter(np.full(len(v), x[i]), v, color=INK, s=14, zorder=3)
ax.axhline(0.05, color=INK2, linestyle=":", linewidth=1.2)
ax.text(x[-1] + 0.3, 0.056, "J04 bound, 5 %", fontsize=8.5, color=INK2, ha="right")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, rotation=12)
ax.set_yscale("log")
ax.set_ylabel("error of $R_f$ at the end of the window")
ax.set_title("c) end-of-window error (11 cells)", loc="left")

fig.suptitle(
    "J05: an extended Kalman filter on a lumped evaporator model against the PINN twin",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
