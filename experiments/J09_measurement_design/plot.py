"""J09 figure: what each instrument error costs and what each remedy buys.

Usage: python experiments/J09_measurement_design/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "j09_measurement_design.png"
d = json.loads((results / "j09.json").read_text())

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
fig, axes = plt.subplots(2, 2, figsize=(17, 12), dpi=200, facecolor=SURFACE)
for ax in axes.ravel():
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def hbars(ax, rows, colors):
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [100 * r["end_rel"] for r in rows], color=colors, height=0.62)
    ax.plot([100 * r["end_rel_ramp"] for r in rows], y, "|", color=INK, markersize=13, mew=2)
    for yi, r in zip(y, rows, strict=True):
        ax.text(100 * r["end_rel"] + 0.15, yi, f"{100 * r['end_rel']:.1f}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels([r["name"] for r in rows], fontsize=9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("std of $R_f(T)$, % of the truth (bar: 6 free modes; tick: a ramp assumed)")


# a) the cost of each instrument error
ax = axes[0, 0]
rows = d["nuisances"]
colors = [INK2] + [AMBER] * (len(rows) - 2) + [ORANGE]
hbars(ax, rows, colors)
ax.set_xlim(0, 13)
ax.set_title("a) what each instrument error costs (online outlet solids and flow)", loc="left")

# b) remedies
ax = axes[0, 1]
rem = d["remedies"]
single = sorted((r for r in rem if len(r["specs"]) == 1), key=lambda r: r["end_rel"])
combos = sorted((r for r in rem if len(r["specs"]) > 1), key=lambda r: r["end_rel"])
short = {
    "condensate meter (gain unknown to 2 %)": "condensate meter",
    "steam swing 4 K (period 400 s)": "4 K swing",
    "steam swing 2 K, period 100 s": "2 K swing at 100 s",
    "condensate meter calibrated to 0.5 %": "condensate meter to 0.5 %",
    "steam calibrated to 0.1 K": "steam to 0.1 K",
    "steam calibrated to 0.03 K": "steam to 0.03 K",
    "3 lab samples of the outlet solids": "3 lab samples",
    "feed flow calibrated to 0.1 %": "feed flow to 0.1 %",
    "outlet flow calibrated to 0.1 %": "outlet flow to 0.1 %",
}
rows = (
    [dict(r, name=" + ".join(short.get(s, s) for s in r["specs"])) for r in combos]
    + single
    + [r for r in rem if not r["specs"]]
)


def colour(r):
    if not r["specs"]:
        return ORANGE
    if len(r["specs"]) > 1:
        return PURPLE
    name = r["specs"][0]
    if "swing" in name:
        return BLUE
    return GREEN if ("condensate" in name or "lab" in name) else AMBER


hbars(ax, rows, [colour(r) for r in rows])
handles = []
for label, c in (
    ("a new measurement", GREEN),
    ("a calibration", AMBER),
    ("the steam signal", BLUE),
    ("combinations", PURPLE),
):
    handles.append(Patch(color=c, label=label))
ax.legend(handles=handles, frameon=False, fontsize=9, loc="upper right")
ax.set_xlim(0, 16)
ax.tick_params(axis="y", labelsize=8)
ax.set_title("b) what each remedy buys, from the baseline", loc="left")

# c) lab samples: how many and when
ax = axes[1, 0]
lab = d["lab"]
k = [0] + [r["k"] for r in lab["rows"]]
for key, color, label in (
    ("greedy", GREEN, "chosen one at a time (greedy)"),
    ("with_ends", BLUE, "equally spaced, ends included"),
    ("at_end", AMBER, "the last k sampling slots"),
    ("interior", INK2, "equally spaced, interior"),
):
    ax.plot(
        k,
        [100 * lab["none"]] + [100 * r[key] for r in lab["rows"]],
        "-o",
        color=color,
        label=label,
        markersize=4,
    )
ax.plot(
    [3],
    [100 * lab["exhaustive3"]],
    "s",
    mfc="none",
    color=INK,
    markersize=11,
    label="best of all 1140 triples",
)
times = lab["rows"][2]["greedy_times"]
ax.annotate(
    "3 samples at " + ", ".join(f"{t / 60:.0f}" for t in times) + " min",
    (3, 100 * lab["rows"][2]["greedy"]),
    xytext=(3.3, 100 * lab["rows"][2]["greedy"] + 0.02),
    fontsize=9,
    color=INK2,
)
ax.set_xlabel("lab samples of the outlet solids in the 20 min window (noise 0.0005, no offset)")
ax.set_ylabel("posterior std of $R_f(T)$, % of the truth")
ax.set_title("c) lab samples: when, and how little they buy here", loc="left")
ax.legend(frameon=False, fontsize=9)

# d) validation
ax = axes[1, 1]
bins = np.linspace(-35, 35, 36)
x = np.linspace(-35, 35, 400)
for v, color in zip(d["validation"], (ORANGE, AMBER, PURPLE), strict=True):
    err = 100 * np.array(v["errors"])
    s = 100 * v["predicted_std"]
    ax.hist(err, bins=bins, density=True, color=color, alpha=0.35)
    ax.plot(
        x,
        np.exp(-0.5 * (x / s) ** 2) / (s * np.sqrt(2 * np.pi)),
        color=color,
        linewidth=1.8,
        label=f"{v['design']}:\npredicted {s:.1f} %, found {100 * v['error_std']:.1f} %, coverage {v['coverage']:.2f}",
    )
ax.set_xlabel("error of the fitted $R_f(T)$, % of the truth (200 simulated data sets each)")
ax.set_ylabel("density")
ax.set_ylim(0, 0.2)
ax.set_title("d) the prediction checked by fitting simulated records", loc="left")
ax.legend(frameon=False, fontsize=9, loc="upper left")

fig.suptitle(
    "J09: measurement design for the fouling estimate, every instrument error admitted",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
