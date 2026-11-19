"""I17 figure: one training step per framework on nine equations, one thread.

Usage: python experiments/I17_step_head_to_head/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from run import PROBLEMS  # noqa: E402

results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "i17_step_head_to_head.png"
rows = json.loads((results / "steps.json").read_text())["rows"]
ms = {(r["arm"], r["problem"]): r["ms_per_step"] for r in rows if "ms_per_step" in r}

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
JAX = ("jax-forward", "jax-jet", "jax-reverse")
SERIES = [  # label, color, arms (the best of them per problem)
    ("pinnrl (step compiled per equation)", "#eb6834", ("pinnrl",)),
    ("JAX, jit (its best mode)", "#2a78d6", JAX),
    ("pinnrl, general kernel", "#f2b79d", ("pinnrl-general",)),
    ("PyTorch, nested autograd.grad", "#8a8985", ("torch-eager",)),
    ("DeepXDE (PyTorch backend)", "#c9c8c4", ("deepxde",)),
]
problems = [p for p in PROBLEMS if ("pinnrl", p) in ms]
fig, ax = plt.subplots(figsize=(13, 8.4), dpi=200, facecolor=SURFACE)
ax.set_facecolor(SURFACE)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
y = np.arange(len(problems))[::-1]
height = 0.16
for k, (label, color, arms) in enumerate(SERIES):
    values = [min(ms[(a, p)] for a in arms if (a, p) in ms) for p in problems]
    bars = ax.barh(y + (2 - k) * height, values, height=height * 0.92, color=color, label=label)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            value * 1.03,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}",
            va="center",
            fontsize=8,
            color=INK if k == 0 else INK2,
        )
ax.set_xscale("log")
ax.set_xlim(2.5, 90)
ax.set_xticks([3, 5, 10, 20, 50], ["3", "5", "10", "20", "50"])
ax.set_yticks(y, [PROBLEMS[p]["label"] for p in problems])
ax.set_xlabel("ms per training step, one thread (log scale)")
ax.legend(frameon=False, loc="lower right", fontsize=10)
ax.set_title(
    "One PINN training step (4 × 64 tanh, 5,120 points, Adam) on an Apple M4",
    loc="left",
    fontsize=13,
)
fig.tight_layout()
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
