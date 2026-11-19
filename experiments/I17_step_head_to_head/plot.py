"""I17 figure: one training step in every framework, three equations, one thread.

Usage: python experiments/I17_step_head_to_head/plot.py [results_dir] [out.png]
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "i17_step_head_to_head.png"
rows = json.loads((results / "steps.json").read_text())["rows"]

SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#a9a8a4"
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
ARMS = {  # arm -> (label, color)
    "jax-forward": ("JAX, jit, forward mode (nested jvp)", BLUE),
    "jax-jet": ("JAX, jit, Taylor mode (jet)", BLUE),
    "pinnrl": ("pinnrl (compiled kernel, the default)", ORANGE),
    "jax-reverse": ("JAX, jit, reverse over reverse", BLUE),
    "torch-eager": ("PyTorch, nested autograd.grad", GREY),
    "pinnrl-autograd": ('pinnrl, derivatives="autograd"', GREY),
    "deepxde": ("DeepXDE (PyTorch backend)", GREY),
    "torch-func": ("torch.func, nested jvp", GREY),
}
SHORT = {
    "jax-forward": "JAX forward",
    "jax-jet": "JAX jet",
    "pinnrl": "pinnrl",
    "jax-reverse": "JAX reverse",
    "torch-eager": "PyTorch",
    "pinnrl-autograd": "pinnrl autograd",
    "deepxde": "DeepXDE",
    "torch-func": "torch.func",
}
TITLES = {
    "heat": "heat: $u_t, u_{xx}$",
    "kdv": "KdV: $u_t, u_x, u_{xxx}$",
    "cahn_hilliard": "Cahn–Hilliard: $u_t, u_x, u_{xx}, u_{xxxx}$",
}
fig, axes = plt.subplots(
    1, 3, figsize=(17, 5.2), dpi=200, facecolor=SURFACE, gridspec_kw={"wspace": 0.38}
)
for ax, problem in zip(axes, TITLES, strict=True):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ms = {
        r["arm"]: r["ms_per_step"]
        for r in rows
        if r["problem"] == problem and r["threads"] == 1 and "ms_per_step" in r
    }
    order = sorted((a for a in ARMS if a in ms), key=ms.get, reverse=True)
    y = np.arange(len(order))
    bars = ax.barh(y, [ms[a] for a in order], height=0.62, color=[ARMS[a][1] for a in order])
    for b, a in zip(bars, order, strict=True):
        ax.text(
            ms[a] + 0.012 * max(ms.values()),
            b.get_y() + b.get_height() / 2,
            f"{ms[a]:.1f}",
            va="center",
            fontsize=9.5,
        )
    ax.set_yticks(y, [ARMS[a][0] for a in order] if ax is axes[0] else [""] * len(order))
    if ax is not axes[0]:
        for b, a in zip(bars, order, strict=True):
            ax.text(
                -0.01 * max(ms.values()),
                b.get_y() + b.get_height() / 2,
                SHORT[a],
                va="center",
                ha="right",
                fontsize=8.5,
                color=INK2,
            )
    ax.set_xlim(0, 1.12 * max(ms.values()))
    ax.set_xlabel("ms per training step, one thread")
    ax.set_title(TITLES[problem], loc="left", fontsize=12)
fig.suptitle(
    "One PINN training step (4 × 64 tanh, 5,120 points, Adam) on an Apple M4: "
    "pinnrl, JAX, PyTorch and DeepXDE",
    x=0.01,
    ha="left",
    fontsize=13,
)
fig.subplots_adjust(left=0.2, right=0.985, top=0.83, bottom=0.12)
fig.savefig(out_png, facecolor=SURFACE)
print("saved", out_png)
