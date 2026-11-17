"""J10 figure: the line over a shift, the fouling monitors, the propagation budget and what each unit's records determine.

Usage: python experiments/J10_line_twin/plot.py [results_dir] [out.png]
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
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j10_line_twin.png"
d = json.loads((results / "j10.json").read_text())

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, AMBER, PURPLE, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#8e5bd9",
    "#d6453d",
)
PASS_COLORS = (BLUE, PURPLE, ORANGE)
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
hours = np.array(d["setup"]["t"]) / 3600.0
rec_h = J.TIMES / 3600.0

# (a) the shift: concentrate after each pass and the powder
ax = axes[0, 0]
noisy = d["setup"]["records_noisy"]
for k in range(3):
    ax.plot(rec_h, noisy["X_pass"][k], ".", color=PASS_COLORS[k], ms=2, alpha=0.5)
    ax.plot(
        hours,
        d["setup"]["X_pass"][k],
        color=PASS_COLORS[k],
        lw=1.8,
        label=f"solids after pass {k + 1}",
    )
ax.set_xlabel("time, h")
ax.set_ylabel("solids mass fraction")
axb = ax.twinx()
axb.plot(hours, d["setup"]["powder"], color=RED, lw=2.2, label="powder moisture (dry basis)")
axb.plot(
    J.LAB_TIMES / 3600.0, noisy["X_powder"], "o", color=RED, ms=6, mfc="white", label="lab assays"
)
axb.set_ylabel("powder moisture", color=RED)
axb.tick_params(axis="y", colors=RED)
axb.spines["top"].set_visible(False)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = axb.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=9, frameon=False)
s0, s1 = d["setup"]["steady_start"], d["setup"]["steady_end"]
ax.set_title(
    f"(a) Four hours of fouling: concentrate {s0['X'][-1]:.3f} → {s1['X'][-1]:.3f}, "
    f"powder {s0['powder']:.3f} → {s1['powder']:.3f}, margin {s0['margin']:.0f} → {s1['margin']:.0f} K",
    loc="left",
    fontsize=11,
)

# (b) the fouling monitors per pass
ax = axes[0, 1]
for m in d["monitors"]:
    k = m["pass"]
    truth = J.FOULING_END[k] * rec_h / 4.0
    ax.fill_between(
        rec_h,
        np.array(m["lower"]) * 1e5,
        np.array(m["upper"]) * 1e5,
        color=PASS_COLORS[k],
        alpha=0.15,
    )
    ax.plot(
        rec_h,
        np.array(m["track"]) * 1e5,
        color=PASS_COLORS[k],
        lw=1.8,
        label=f"pass {k + 1}: monitor, 90 % band",
    )
    ax.plot(rec_h, truth * 1e5, color=PASS_COLORS[k], lw=1.0, ls="--")
ax.plot([], [], color=INK2, ls="--", lw=1.0, label="truth")
ax.set_xlabel("time, h")
ax.set_ylabel("fouling resistance, 1e-5 m²K/W")
ax.legend(loc="upper left", fontsize=9, frameon=False)
mis = ", ".join(f"{m['misfit'][0]:.1f}/{m['misfit'][1]:.1f}" for m in d["monitors"])
ax.set_title(
    f"(b) The fouling monitor on each pass, fed the previous pass's record (misfits {mis})",
    loc="left",
    fontsize=11,
)

# (c) the variance budget of the powder at the end of the shift
ax = axes[1, 0]
prop = d["propagation"]
cases = [
    "unmonitored fouling + dryer + feed",
    "monitored fouling + dryer + feed",
    "monitored fouling only",
    "dryer + feed only",
]
sources = ["R_f_0", "R_f_1", "R_f_2", "X_cr", "n", "U_wall", "d0", "W_feed", "X_feed"]
colors = {
    "R_f_0": BLUE,
    "R_f_1": PURPLE,
    "R_f_2": ORANGE,
    "X_cr": GREEN,
    "n": "#7fd0a8",
    "U_wall": AMBER,
    "d0": "#f3c65a",
    "W_feed": INK2,
    "X_feed": "#a7a5a0",
}
labels = {
    "R_f_0": "fouling, pass 1",
    "R_f_1": "fouling, pass 2",
    "R_f_2": "fouling, pass 3",
    "X_cr": "drying curve $X_{cr}$",
    "n": "drying curve $n$",
    "U_wall": "wall loss",
    "d0": "droplet size",
    "W_feed": "feed flow",
    "X_feed": "feed solids",
}
y = np.arange(len(cases))
left = np.zeros(len(cases))
for src in sources:
    var = np.array(
        [
            prop[c]["linear"]["powder"]["std"] ** 2
            * prop[c]["linear"]["powder"]["shares"].get(src, 0.0)
            for c in cases
        ]
    )
    ax.barh(y, var, left=left, color=colors[src], height=0.6, label=labels[src])
    left += var
for yi, c in zip(y, cases, strict=True):
    lin = prop[c]["linear"]["powder"]["std"]
    mc = prop[c].get("monte_carlo", {}).get("powder", {}).get("std")
    txt = f"σ = {lin:.3f}" + (f" (Monte Carlo {mc:.3f})" if mc else "")
    ax.annotate(
        txt,
        (left[yi], yi),
        xytext=(5, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK2,
    )
ax.set_yticks(y)
ax.set_yticklabels(cases, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(
    "variance of the powder moisture at the end of the shift (propagated, no dryer records)"
)
ax.set_xlim(0, left.max() * 1.45)
ax.legend(loc="lower right", fontsize=8, frameon=False, ncol=3)
ax.set_title(
    "(c) Whose uncertainty the powder inherits, by propagation through the line",
    loc="left",
    fontsize=11,
)

# (d) what each unit's records determine
ax = axes[1, 1]
jt = d["joint"]
combos = ["evaporator", "dryer online", "dryer online + lab", "evaporator + dryer online", "all"]
x = np.arange(len(combos))
powder = [jt[c]["powder_end_std"] for c in combos]
fouling = [
    np.sqrt(np.mean(np.square(jt[c]["fouling_end_std"][1:]))) for c in combos
]  # passes 2 and 3 (pass 1 is confounded with the feed)
ax.bar(x - 0.2, powder, width=0.4, color=RED, label="powder moisture at the end (std)")
ax.set_ylabel("std of the powder moisture", color=RED)
ax.tick_params(axis="y", colors=RED)
ax.set_yscale("log")
for xi, v in zip(x - 0.2, powder, strict=True):
    ax.annotate(
        f"{v:.4f}",
        (xi, v),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color=RED,
    )
axb = ax.twinx()
shown = [min(v * 1e5, 0.6) for v in fouling]  # a flat posterior (no evaporator records) is clipped
axb.bar(x + 0.2, shown, width=0.4, color=PURPLE, label="fouling at the end, passes 2–3 (std)")
axb.set_yscale("log")
axb.set_ylim(0.05, 3.0)
axb.set_ylabel("std of the fouling, 1e-5 m²K/W", color=PURPLE)
axb.tick_params(axis="y", colors=PURPLE)
axb.spines["top"].set_visible(False)
for xi, v, sh in zip(x + 0.2, fouling, shown, strict=True):
    if v * 1e5 < 0.6:
        axb.annotate(
            f"{v * 1e5:.2f}",
            (xi, sh),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=PURPLE,
        )
    else:
        axb.annotate(
            "undetermined",
            (xi, 0.06),
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=8,
            color="white",
        )
ax.set_xticks(x)
ax.set_xticklabels([c.replace(" + ", "\n+ ") for c in combos], fontsize=9)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = axb.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9, frameon=False)
two = d["propagation"]["monitored fouling + dryer + feed"]["linear"]["powder"]["std"]
ax.axhline(two, color=INK2, ls=":", lw=1.2)
ax.annotate(
    f"two-stage (monitors, then propagation): {two:.3f}",
    (len(combos) - 0.6, two),
    xytext=(0, -11),
    textcoords="offset points",
    fontsize=8,
    color=INK2,
    ha="right",
)
ax.set_ylim(8e-4, 0.2)
ax.set_title(
    "(d) Linear-Gaussian posteriors of twelve unknowns from each unit's records",
    loc="left",
    fontsize=11,
)

fig.suptitle(
    "J10: the line-level twin, a three-pass evaporator feeding the spray dryer",
    fontsize=14,
    x=0.01,
    ha="left",
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.97))
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
