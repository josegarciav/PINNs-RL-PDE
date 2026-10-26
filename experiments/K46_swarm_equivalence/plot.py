"""K46 figure: the swarm's accepted samples against the 90 % Fisher ellipse, one data set per problem.

Usage: python experiments/K46_swarm_equivalence/plot.py [out.png]
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Ellipse

from pinnrl.analysis import closed_form_bound, fisher_information
from pinnrl.pdes.catalog import from_catalog
from pinnrl.uq import sample_equivalence_region

HERE = Path(__file__).parent
out_png = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results" / "k46_swarm_equivalence.png"
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b5b4b0"
plt.rcParams.update(
    {
        "font.size": 10.5,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
    }
)
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), dpi=220, facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
seed = 0


def ellipse(ax, centre, cov, color, label):
    vals, vecs = np.linalg.eigh(cov)
    angle = math.degrees(math.atan2(vecs[1, 1], vecs[0, 1]))
    r = math.sqrt(4.605)
    ax.add_patch(
        Ellipse(
            centre,
            2 * r * math.sqrt(vals[1]),
            2 * r * math.sqrt(vals[0]),
            angle=angle,
            facecolor=color,
            alpha=0.18,
            edgecolor=color,
            linewidth=1.6,
            label=label,
        )
    )


# KdV two-parameter
pde = from_catalog("kdv_two_parameter")
obs = pde.generate_synthetic_observations(n_points=200, noise_std=0.01, seed=seed)
x, t, y = obs["x"].numpy().T, obs["t"].numpy().T, obs["u"].numpy().T
bound = closed_form_bound(pde, ["alpha", "beta"], obs["x"], obs["t"], 0.01)


def misfit_kdv(theta):
    a, b = theta[:, 0:1], theta[:, 1:2]
    return ((3 / a / np.cosh(0.5 * np.sqrt(1 / b) * (x - t)) ** 2 - y) ** 2).sum(axis=1)


out = sample_equivalence_region(
    misfit_kdv, [(2, 12), (0.3, 3)], None, sigma=0.01, n_particles=40, n_iterations=150, seed=seed
)
ax = axes[0]
ax.scatter(
    out.positions[~out.accepted, 0],
    out.positions[~out.accepted, 1],
    s=6,
    color=GREY,
    alpha=0.5,
    label="evaluated, outside the region",
)
ax.scatter(
    out.samples[:, 0],
    out.samples[:, 1],
    s=12,
    color=BLUE,
    label="accepted: misfit within σ²χ²₂(0.9) of the best",
)
ellipse(ax, out.best, bound.covariance.numpy(), ORANGE, "90 % Fisher ellipse at the best fit")
ax.plot(6, 1, "k+", markersize=12, markeredgewidth=2, label="truth")
s = out.samples
pad = 4 * bound.std.numpy()
ax.set_xlim(out.best[0] - pad[0], out.best[0] + pad[0])
ax.set_ylim(out.best[1] - pad[1], out.best[1] + pad[1])
ax.set_xlabel("α")
ax.set_ylabel("β")
ax.set_title(
    f"KdV soliton (α, β), 200 points at noise 0.01: {int(out.accepted.sum())} accepted of {out.positions.shape[0]}",
    loc="left",
    fontsize=11,
    pad=10,
)
ax.legend(frameon=False, fontsize=8, loc="upper left")

# Saturation model
xs = np.linspace(1, 30, 40)
true = np.array([20.0, 0.1])
rng = np.random.default_rng(seed)
sigma = 0.3
ys = true[0] * (1 - np.exp(-true[1] * xs)) + sigma * rng.normal(size=xs.size)
J = np.stack([1 - np.exp(-true[1] * xs), true[0] * xs * np.exp(-true[1] * xs)], axis=1)
bound2 = fisher_information(torch.tensor(J), sigma, ["a", "b"])


def misfit_sat(theta):
    return ((theta[:, :1] * (1 - np.exp(-theta[:, 1:2] * xs[None, :])) - ys[None, :]) ** 2).sum(
        axis=1
    )


out2 = sample_equivalence_region(
    misfit_sat,
    [(5, 40), (0.02, 0.3)],
    None,
    sigma=sigma,
    n_particles=40,
    n_iterations=150,
    seed=seed,
)
ax = axes[1]
ax.scatter(
    out2.positions[~out2.accepted, 0], out2.positions[~out2.accepted, 1], s=6, color=GREY, alpha=0.5
)
ax.scatter(out2.samples[:, 0], out2.samples[:, 1], s=12, color=BLUE)
ellipse(ax, out2.best, bound2.covariance.numpy(), ORANGE, None)
ax.plot(true[0], true[1], "k+", markersize=12, markeredgewidth=2)
pad = 4 * bound2.std.numpy()
ax.set_xlim(out2.best[0] - pad[0], out2.best[0] + pad[0])
ax.set_ylim(out2.best[1] - pad[1], out2.best[1] + pad[1])
ax.set_xlabel("a")
ax.set_ylabel("b")
ax.set_title(
    f"y = a(1 − e^(−bx)), 40 points at noise 0.3: {int(out2.accepted.sum())} accepted of {out2.positions.shape[0]}",
    loc="left",
    fontsize=11,
    pad=10,
)
fig.suptitle(
    "K46: the region of equivalence sampled by the cloud swarm vs the Fisher ellipse (data set 0)",
    x=0.01,
    ha="left",
    fontsize=13,
    fontweight="bold",
    color=INK,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out_png, facecolor=SURFACE)
print("wrote", out_png)
