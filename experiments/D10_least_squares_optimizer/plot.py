"""D10 figure and table: the error each optimizer has reached after T seconds on one core.

Usage: python experiments/D10_least_squares_optimizer/plot.py [runs.json] [out.png]

Prints the table of the README (median over seeds, with the range) and saves the figure.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "runs_3x32.json"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else path.with_suffix(".png")
data = json.loads(path.read_text())
runs, budget = data["runs"], data["meta"]["budget_s"]

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
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
SERIES = {  # method: (label, color, line style)
    "lm": ("Levenberg–Marquardt", "#eb6834", "-"),
    "flow": ("M1 damping set by the linearization error", "#b0441a", "--"),
    "varpro60": ("M5 last layer solved, after 60 plain steps", "#7a3fb5", "-"),
    "varpro": ("M5 last layer solved, from the start", "#7a3fb5", ":"),
    "block128": ("M4 blocks of 128 rows", "#1f9d6b", "-"),
    "block32": ("M4 blocks of 32 rows", "#1f9d6b", ":"),
    "continue4": ("M6 viscosity in 4 stages", "#c2255c", ":"),
    "continue8": ("M6 viscosity in 8 stages", "#c2255c", "-"),
    "continue16": ("M6 viscosity in 16 stages", "#c2255c", "--"),
    "filter64": ("M3 Kalman filter, batches of 64", "#0b7285", "-"),
    "filter256": ("M3 Kalman filter, batches of 256", "#0b7285", "--"),
    "lmfilter20": ("Levenberg–Marquardt for 20 s, then the filter", "#0b7285", "-."),
    "lmfilter20w": ("The same, covariance started from the last linearization", "#0b7285", ":"),
    "dfilter64layer": ("M3 filter, covariance per layer", "#5f3dc4", "-"),
    "sfilter64r256": ("M3 filter, 256 directions of information kept", "#c2255c", "-"),
    "sfilter64r64": ("M3 filter, 64 directions kept", "#c2255c", ":"),
    "dfilter64node": ("M3 filter, covariance per neuron", "#5f3dc4", ":"),
    "al": ("M8 multipliers, penalty 1", "#946200", "-"),
    "lm_w1": ("Levenberg–Marquardt, loss weights 1", "#946200", ":"),
    "adam_lbfgs": ("Adam, then L-BFGS", "#2a78d6", "-"),
    "lbfgs": ("L-BFGS", "#2a78d6", "--"),
    "adam": ("Adam", "#8a8985", "-"),
}
problems = list(dict.fromkeys(r["problem"] for r in runs))
methods = [m for m in SERIES if any(r["method"] == m for r in runs)]
grid = np.linspace(0.0, budget, 121)


def at(curve, t):
    """The error held at time ``t``: the last reading at or before it."""
    times = np.array([c[0] for c in curve])
    errors = np.array([c[1] for c in curve])
    return errors[np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(errors) - 1)]


def band(problem, method):
    curves = [r["curve"] for r in runs if (r["problem"], r["method"]) == (problem, method)]
    held = np.array([at(c, grid) for c in curves])
    return np.median(held, 0), held.min(0), held.max(0)


steps = (10.0, 30.0, 60.0, 120.0) if budget <= 180 else (30.0, 60.0, 180.0, 300.0)
marks = [t for t in steps if t <= budget]


def settled(curve):
    """Geometric mean of the readings in the last sixth of the budget: the error wanders as
    the points are redrawn, so one reading is a noisy summary."""
    late = [e for t, e in curve if t >= budget * 5.0 / 6.0]
    return float(np.exp(np.mean(np.log(late))))


head = " | ".join(f"{t:.0f} s" for t in marks)
print(f"| Equation | Method | {head} | Last {budget / 6:.0f} s, mean | Steps |")
print("|---|---|" + "---|" * (len(marks) + 2))
for problem in problems:
    for method in methods:
        mine = [r for r in runs if (r["problem"], r["method"]) == (problem, method)]
        if not mine:
            continue
        cells = []
        for t in marks:
            values = [float(at(r["curve"], t)) for r in mine]
            cells.append(f"{np.median(values):.1e} ({min(values):.0e}–{max(values):.0e})")
        cells.append(f"{np.median([settled(r['curve']) for r in mine]):.1e}")
        steps = int(np.median([r["iterations"] for r in mine]))
        print(f"| {problem} | {SERIES[method][0]} | " + " | ".join(cells) + f" | {steps} |")

fig, axes = plt.subplots(1, len(problems), figsize=(5.2 * len(problems), 5.2), facecolor=SURFACE)
for ax, problem in zip(np.atleast_1d(axes), problems, strict=True):
    ax.set_facecolor(SURFACE)
    for method in methods:
        if not any((r["problem"], r["method"]) == (problem, method) for r in runs):
            continue
        label, color, style = SERIES[method]
        median, low, high = band(problem, method)
        ax.fill_between(grid, low, high, color=color, alpha=0.10, linewidth=0)
        ax.plot(grid, median, style, color=color, linewidth=1.8, label=label)
    ax.set_yscale("log")
    ax.set_title(problem, loc="left")
    ax.set_xlabel("seconds of optimization, one CPU core")
    ax.grid(True, which="major", color="#e6e5e1", linewidth=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("relative $L_2$ error")
handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, fontsize=9, ncol=3, loc="lower center")
fig.tight_layout(rect=(0, 0.14, 1, 1))
fig.savefig(out_png, dpi=160, facecolor=SURFACE)
print("saved", out_png)
