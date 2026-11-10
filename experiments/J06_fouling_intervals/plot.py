"""J06 figure: are the fouling intervals calibrated?

Usage: python experiments/J06_fouling_intervals/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j06_fouling_intervals.png"
records = json.loads((results / "classical.json").read_text())
summary = json.loads((results / "summary.json").read_text())
pinn_path = results / "pinn.json"
pinn = json.loads(pinn_path.read_text()) if pinn_path.exists() else []
classical = summary["classical"]
R_END, T_MAX, Z = 2e-4, 1200.0, 1.6448536269514722

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
fig, axes = plt.subplots(2, 2, figsize=(17, 11.5), dpi=200, facecolor=SURFACE)
for ax in axes.ravel():
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def first(scenario, arm):
    return next(
        r for r in records if r["scenario"] == scenario and r["arm"] == arm and r["dataset"] == 0
    )


# ---------------------------------------------------------------- a) one data set
ax = axes[0, 0]
t = np.linspace(0, T_MAX, 121)
ax.plot(t / 60, 1e4 * R_END * t / T_MAX, "-", color=INK, linewidth=2.4, label="truth")
r = first("known", "batch-6")
mean, std = 1e4 * np.array(r["profile"]), 1e4 * np.array(r["profile_std"])
tt = np.array(r["t"]) / 60
ax.fill_between(tt, mean - Z * std, mean + Z * std, color=GREEN, alpha=0.16, linewidth=0)
ax.plot(tt, mean, "-", color=GREEN, linewidth=1.5, label="batch fit, 6 modes (90 % band)")
for arm, color, label in (
    ("ekf-rw 1e-06", ORANGE, "filter, random walk $10^{-6}$, 11 cells"),
    ("ekf-llt 1e-09 nz51", BLUE, "filter, trend $10^{-9}$, 51 cells"),
):
    r = first("known", arm)
    mean, std = 1e4 * np.array(r["profile_filter"]), 1e4 * np.array(r["profile_filter_std"])
    tt = np.array(r["t"]) / 60
    ax.fill_between(tt, mean - Z * std, mean + Z * std, color=color, alpha=0.14, linewidth=0)
    ax.plot(tt, mean, "-", color=color, linewidth=1.4, label=label)
limit = summary.get("pinn", {}).get("chi2_limit", np.inf)
shown = set()
for r in (r for r in pinn if r["dataset"] == 0):
    ok = r["chi2"] <= limit
    label = "PINN twin, accepted" if ok else "PINN twin, rejected by the misfit guard"
    ax.plot(
        np.array(r["t"]) / 60,
        1e4 * np.array(r["profile"]),
        "-" if ok else ":",
        color=PURPLE,
        linewidth=1.3,
        label=None if label in shown else label,
    )
    shown.add(label)
ax.set_xlabel("time (min)")
ax.set_ylabel("$R_f$ ($10^{-4}$ m²K/W)")
ax.set_ylim(-0.2, 3.3)
ax.set_title("a) one data set: the estimates and their 90 % bands", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

# ---------------------------------------------------------------- b) coverage, inputs known
ax = axes[0, 1]
rows = [
    ("batch, 2 modes", "known | batch-2", GREEN),
    ("batch, 6 modes", "known | batch-6", GREEN),
    ("random walk $10^{-7}$, 11 cells", "known | ekf-rw 1e-07", ORANGE),
    ("random walk $10^{-6}$, 11 cells", "known | ekf-rw 1e-06", ORANGE),
    ("random walk $10^{-6}$, 51 cells", "known | ekf-rw 1e-06 nz51", ORANGE),
    ("random walk $10^{-5}$, 51 cells", "known | ekf-rw 1e-05 nz51", ORANGE),
    ("trend $10^{-10}$, 11 cells", "known | ekf-llt 1e-10", BLUE),
    ("trend $10^{-10}$, 51 cells", "known | ekf-llt 1e-10 nz51", BLUE),
    ("trend $10^{-9}$, 11 cells", "known | ekf-llt 1e-09", BLUE),
    ("trend $10^{-9}$, 51 cells", "known | ekf-llt 1e-09 nz51", BLUE),
    ("trend $10^{-8}$, 51 cells", "known | ekf-llt 1e-08 nz51", BLUE),
]
bars = [
    (label, classical[key]["end"]["coverage"], classical[key]["end"]["width_rel"], color)
    for label, key, color in rows
    if key in classical
]
if "pinn" in summary:
    for name, label in (
        ("fisher at one member", "PINN, Fisher width"),
        ("ensemble", "PINN, ensemble of accepted"),
        ("ensemble + fisher", "PINN, ensemble + Fisher"),
    ):
        iv = summary["pinn"]["end"]["accepted"]["intervals"].get(name)
        if iv:
            bars.append((f"{label} (n = {iv['n']})", iv["coverage"], iv["width_rel"], PURPLE))
y = np.arange(len(bars))[::-1]
ax.barh(y, [b[1] for b in bars], color=[b[3] for b in bars], height=0.62)
for yi, b in zip(y, bars, strict=True):
    ax.text(max(b[1], 0.0) + 0.012, yi, f"±{50 * b[2]:.1f} %", va="center", fontsize=9, color=INK2)
ax.axvline(0.9, color=INK, linestyle=":", linewidth=1.4)
ax.axvspan(0.84, 0.96, color=INK2, alpha=0.08, linewidth=0)
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars], fontsize=9)
ax.set_xlim(0, 1.16)
ax.set_xlabel("coverage of $R_f(T)$ by the 90 % interval (half-width at the bar's end)")
ax.set_title("b) inputs known: coverage over 100 data sets", loc="left")
ax.grid(axis="y", visible=False)

# ---------------------------------------------------------------- c) the two harder scenarios
ax = axes[1, 0]
groups = [
    ("batch\n2 modes", "bias | batch-2", "bias | batch-2 + bias"),
    ("batch\n6 modes", "bias | batch-6", "bias | batch-6 + bias"),
    (
        "random walk\n51 cells",
        "bias | ekf-rw 1e-06 nz51, no clip",
        "bias | ekf-rw 1e-06 nz51 + bias, no clip",
    ),
    (
        "trend\n51 cells",
        "bias | ekf-llt 1e-09 nz51, no clip",
        "bias | ekf-llt 1e-09 nz51 + bias, no clip",
    ),
]
x = np.arange(len(groups))
for offset, idx, color, label in (
    (-0.2, 1, AMBER, "steam offset ignored"),
    (0.2, 2, GREEN, "steam offset estimated"),
):
    values = [classical[g[idx]]["end"] for g in groups]
    ax.bar(x + offset, [v["coverage"] for v in values], width=0.38, color=color, label=label)
    for xi, v in zip(x + offset, values, strict=True):
        ax.text(
            xi,
            v["coverage"] + 0.02,
            f"±{50 * v['width_rel']:.0f} %",
            ha="center",
            fontsize=9,
            color=INK2,
        )
sat = [
    ("batch\n2 modes", "saturating | batch-2"),
    ("batch\n6 modes", "saturating | batch-6"),
    ("random walk\n$10^{-6}$", "saturating | ekf-rw 1e-06 nz51"),
    ("trend\n$10^{-9}$", "saturating | ekf-llt 1e-09 nz51"),
    ("trend\n$10^{-8}$", "saturating | ekf-llt 1e-08 nz51"),
]
xs = len(groups) + 0.6 + np.arange(len(sat))
values = [classical[key]["end"] for _, key in sat]
ax.bar(
    xs,
    [v["coverage"] for v in values],
    width=0.5,
    color=BLUE,
    label="saturating fouling, inputs known",
)
for xi, v in zip(xs, values, strict=True):
    ax.text(
        xi,
        v["coverage"] + 0.02,
        f"±{50 * v['width_rel']:.0f} %",
        ha="center",
        fontsize=9,
        color=INK2,
    )
ax.axhline(0.9, color=INK, linestyle=":", linewidth=1.4)
ax.axvline(len(groups) - 0.1, color=GRID, linewidth=1.2)
ax.set_xticks(np.concatenate([x, xs]))
ax.set_xticklabels([g[0] for g in groups] + [g[0] for g in sat], fontsize=9)
ax.set_ylim(0, 1.18)
ax.set_ylabel("coverage of $R_f(T)$ by the 90 % interval")
ax.set_title("c) a steam record 0.3 K uncertain; a fouling curve that saturates", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper center", ncol=3)
ax.grid(axis="x", visible=False)

# ---------------------------------------------------------------- d) time to clean
ax = axes[1, 1]
shown_n = 40
for arm, color, offset, label in (
    ("batch-2", GREEN, -0.18, "batch, 2 modes"),
    ("ekf-llt 1e-09 nz51", BLUE, 0.18, "filter, trend $10^{-9}$, 51 cells"),
):
    rows_ = sorted(
        (
            r
            for r in records
            if r["scenario"] == "known" and r["arm"] == arm and r["dataset"] < shown_n
        ),
        key=lambda r: r["dataset"],
    )
    s = classical[f"known | {arm}"]["ttc"]
    for i, r in enumerate(rows_):
        tc = r["targets"]["ttc"]
        hi = min(tc["hi"], 3600.0) if np.isfinite(tc["hi"]) else 3600.0
        ax.plot(
            [i + offset, i + offset],
            [tc["lo"] / 60, hi / 60],
            "-",
            color=color,
            linewidth=1.6,
            alpha=0.9,
        )
        ax.plot(i + offset, tc["est"] / 60, "o", color=color, markersize=3)
    ax.plot(
        [],
        [],
        "-o",
        color=color,
        markersize=3,
        label=f"{label}: coverage {s['coverage']:.2f}, width {100 * s['width_rel']:.0f} % of the truth",
    )
ax.axhline(10.0, color=INK, linewidth=1.6, label="truth, 10 min")
ax.set_xlabel(f"data set (the first {shown_n} of 100)")
ax.set_ylabel("time until $R_f = 3 \\times 10^{-4}$, from the end of the record (min)")
ax.set_ylim(6, 17)
ax.set_title("d) the time to the next cleaning: median and 90 % interval", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle(
    "J06: calibration of the fouling intervals, by estimator, model error and input uncertainty",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
