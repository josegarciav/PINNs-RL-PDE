"""I11 figure: time per training step by implementation, and where a step's time goes.

Usage: python experiments/I11_native_pinn/plot.py [results_dir] [out.png]
"""

import collections
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
results = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
out_png = Path(sys.argv[2]) if len(sys.argv) > 2 else results / "i11_native_pinn.png"
rows = []
for name in ("python.json", "native.json", "trainer.json"):
    rows += json.loads((results / name).read_text())

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
fig, axes = plt.subplots(
    1, 2, figsize=(17, 6.8), dpi=200, facecolor=SURFACE, width_ratios=[1.25, 1]
)
for ax in axes:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

times = collections.defaultdict(list)
for r in rows:
    if "ms_per_step" in r and "long" not in r["impl"] and "breakdown" not in r["impl"]:
        times[(r["impl"], r.get("threads"))].append(r["ms_per_step"])


def ms(impl, threads=None):
    return float(np.mean(times[(impl, threads)]))


# a) time per step
ax = axes[0]
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
bars = [
    ("C++, Accelerate kernels (standalone program)", ms("cpp-blas"), None, PURPLE),
    ("Rust, Accelerate kernels (standalone program)", ms("rust-blas"), None, PURPLE),
    ("pinnrl trainer, derivatives='native'", ms("pinnrl-native", 1), ms("pinnrl-native", 4), GREEN),
    (
        "PyTorch, hand-written gradients + Accelerate tanh",
        ms("python-manual + accelerate-tanh", 1),
        ms("python-manual + accelerate-tanh", 4),
        BLUE,
    ),
    (
        "PyTorch autograd + Accelerate tanh",
        ms("python-autograd + accelerate-tanh", 1),
        ms("python-autograd + accelerate-tanh", 4),
        BLUE,
    ),
    (
        "PyTorch autograd (the textbook step)",
        ms("python-autograd", 1),
        ms("python-autograd", 4),
        BLUE,
    ),
    ("PyTorch, Taylor-mode forward", ms("python-taylor", 1), ms("python-taylor", 4), BLUE),
    (
        "pinnrl trainer, autograd (before)",
        ms("pinnrl-autograd", 1),
        ms("pinnrl-autograd", 4),
        ORANGE,
    ),
    ("C++, plain loops", ms("cpp-naive"), None, INK2),
    ("Rust, plain loops", ms("rust-naive"), None, INK2),
]
y = np.arange(len(bars))[::-1]
ax.barh(y, [min(b[1], 16.0) for b in bars], color=[b[3] for b in bars], height=0.62)
for yi, (_, one, four, _) in zip(y, bars, strict=True):
    if four is not None:
        ax.plot([four], [yi], "|", color=INK, markersize=15, mew=2.5)
    label = f"{one:.1f} ms" + (f"  (4 threads: {four:.1f})" if four is not None else "")
    ax.text(min(one, 16.0) + 0.2, yi, label, va="center", fontsize=9)
ax.plot([], [], "|", color=INK, markersize=12, mew=2.5, label="with 4 PyTorch threads")
ax.set_yticks(y)
ax.set_yticklabels([b[0] for b in bars], fontsize=9)
ax.set_xlim(0, 21)
ax.set_xlabel("time per training step, ms (bars: one thread; plain loops are off the scale)")
ax.set_title("a) one step of the same heat PINN, ten ways", loc="left")
ax.legend(frameon=False, fontsize=9, loc="center right")

# b) where the time goes
ax = axes[1]
ax.grid(True, axis="x", color=GRID, linewidth=0.8)
stages = {r["impl"]: r["stages_ms"] for r in rows if r["impl"].startswith("stages")}
cpp = next(r for r in rows if r["impl"] == "cpp-blas")
before, after = stages["stages pinnrl-autograd"], stages["stages pinnrl-native"]
groups = [
    (
        "pinnrl, autograd",
        [
            ("tanh (forward)", before["tanh"], AMBER),
            ("autograd engine (three backward passes)", before["autograd engine"], ORANGE),
            ("matrix products (forward)", before["linear"], BLUE),
            (
                "everything else",
                before["everything else"] + before["sampling"] + before["optimizer"],
                INK2,
            ),
        ],
    ),
    (
        "pinnrl, native",
        [
            ("kernel, forward", after["kernel forward"], GREEN),
            ("kernel, backward", after["kernel backward"], PURPLE),
            ("autograd engine", after["autograd engine"], ORANGE),
            (
                "everything else",
                after["everything else"] + after["sampling"] + after["optimizer"],
                INK2,
            ),
        ],
    ),
    (
        "C++ standalone",
        [
            ("matrix products", cpp["kernels_ms"]["matrix products"], BLUE),
            ("tanh", cpp["kernels_ms"]["tanh"], AMBER),
            (
                "elementwise and the rest",
                cpp["ms_per_step"]
                - cpp["kernels_ms"]["matrix products"]
                - cpp["kernels_ms"]["tanh"],
                INK2,
            ),
        ],
    ),
]
seen = set()
for yi, (_name, parts) in zip((2, 1, 0), groups, strict=True):
    left = 0.0
    for label, value, color in parts:
        ax.barh(
            yi, value, left=left, color=color, height=0.55, label=None if label in seen else label
        )
        seen.add(label)
        if value > 0.45:
            ax.text(
                left + value / 2,
                yi,
                f"{value:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color=SURFACE,
            )
        left += value
    ax.text(left + 0.15, yi, f"{left:.1f} ms", va="center", fontsize=9)
ax.set_yticks([2, 1, 0])
ax.set_yticklabels([g[0] for g in groups], fontsize=10)
ax.set_xlim(0, 12.5)
ax.set_xlabel("ms per step, one thread (the two trainer bars are measured under the profiler)")
ax.set_title("b) where a step's time goes", loc="left")
ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

fig.suptitle(
    "I11: a PINN training step in PyTorch, C++ and Rust, and the kernel that went back into the trainer",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
