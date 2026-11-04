"""K47: is the swarm sample a posterior sample?

A cloud particle swarm evaluates the misfit at thousands of positions while it
optimizes. Weighting every evaluated position by exp(-(misfit - min)/2 sigma^2)
turns the run into an importance sample of the Gaussian posterior, provided the
positions cover the posterior's support. This is tested against the exact
posterior, computed on a grid with the same forward model:

* heat alpha (closed form), 200 observations, noise 0.05, 30 data sets;
* damped pendulum theta'' + b theta' + k sin(theta) = 0 with k = g/L
  (g and L alone are not separately identifiable), parameters (k, b),
  theta observed at 50 times on [0, 10] with noise 0.02, 10 data sets; the
  forward model is an ODE solve.

For each data set: the exact posterior mean, std and central 90 % interval;
the same from the weighted evaluated positions (importance sampling) and from
the accepted (unweighted) positions as K46 used them; the coverage of the
truth by each 90 % interval; the effective sample size of the weights; and the
Kolmogorov distance between the weighted sample's marginal CDFs and the exact
ones.

Usage: python experiments/K47_swarm_posterior/run.py [--heat 30] [--pendulum 10]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent


def weighted_quantiles(values, weights, qs):
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cdf = np.cumsum(w) / w.sum()
    return np.interp(qs, cdf, v)


def kolmogorov(values, weights, grid, exact_cdf):
    """Max distance between the weighted sample's CDF and the exact CDF on ``grid``."""
    order = np.argsort(values)
    v, w = values[order], np.cumsum(weights[order]) / weights.sum()
    sample_cdf = np.interp(grid, v, w, left=0.0, right=1.0)
    return float(np.max(np.abs(sample_cdf - exact_cdf)))


def summarize(problem, seed, out, sigma, truth, grid_axes, log_post, names):
    """Exact posterior on a grid vs the swarm's weighted and accepted samples."""
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    w = np.exp(-(out.misfits - out.misfits.min()) / (2 * sigma**2))
    ess = float(w.sum() ** 2 / (w**2).sum())
    rec = {
        "problem": problem,
        "seed": seed,
        "n_evaluated": int(out.positions.shape[0]),
        "n_accepted": int(out.accepted.sum()),
        "ess": ess,
        "per_parameter": [],
    }
    for j, name in enumerate(names):
        axes = [k for k in range(len(grid_axes)) if k != j]
        marginal = post.sum(axis=tuple(axes)) if axes else post
        g = grid_axes[j]
        cdf = np.cumsum(marginal)
        mean = float((g * marginal).sum())
        std = float(np.sqrt(((g - mean) ** 2 * marginal).sum()))
        lo, hi = np.interp([0.05, 0.95], cdf, g)
        vals = out.positions[:, j]
        wl, wh = weighted_quantiles(vals, w, [0.05, 0.95])
        wmean = float((w * vals).sum() / w.sum())
        wstd = float(np.sqrt((w * (vals - wmean) ** 2).sum() / w.sum()))
        acc = vals[out.accepted]
        al, ah = np.percentile(acc, [5, 95]) if acc.size else (np.nan, np.nan)
        rec["per_parameter"].append(
            {
                "name": name,
                "truth": float(truth[j]),
                "exact_mean": mean,
                "exact_std": std,
                "exact_90": [float(lo), float(hi)],
                "weighted_mean": wmean,
                "weighted_std": wstd,
                "weighted_90": [float(wl), float(wh)],
                "accepted_std": float(acc.std()) if acc.size else float("nan"),
                "accepted_90": [float(al), float(ah)],
                "truth_in_exact_90": bool(lo <= truth[j] <= hi),
                "truth_in_weighted_90": bool(wl <= truth[j] <= wh),
                "truth_in_accepted_90": bool(al <= truth[j] <= ah) if acc.size else False,
                "kolmogorov_weighted": kolmogorov(vals, w, g, cdf),
                "kolmogorov_accepted": kolmogorov(acc, np.ones_like(acc), g, cdf)
                if acc.size
                else float("nan"),
            }
        )
    return rec


def heat_case(seed):
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.uq import sample_equivalence_region

    sigma = 0.05
    pde = from_catalog("heat")
    obs = pde.generate_synthetic_observations(n_points=200, noise_std=sigma, seed=seed)
    x, t, y = obs["x"].numpy().T, obs["t"].numpy().T, obs["u"].numpy().T

    def misfit(theta):
        return ((np.exp(-theta[:, 0:1] * math.pi**2 * t) * np.sin(math.pi * x) - y) ** 2).sum(
            axis=1
        )

    out = sample_equivalence_region(
        misfit, [(0.02, 0.5)], None, sigma=sigma, n_particles=30, n_iterations=100, seed=seed
    )
    grid = np.linspace(0.02, 0.5, 4001)
    log_post = -misfit(grid[:, None]) / (2 * sigma**2)
    return summarize("heat_alpha", seed, out, sigma, np.array([0.1]), [grid], log_post, ["alpha"])


def pendulum_solve(k, b, times):
    from scipy.integrate import solve_ivp

    def rhs(_t, s):
        return [s[1], -b * s[1] - k * math.sin(s[0])]

    sol = solve_ivp(rhs, (0.0, float(times[-1])), [0.5, 0.0], t_eval=times, rtol=1e-8, atol=1e-10)
    return sol.y[0]


def pendulum_case(seed):
    from pinnrl.uq import sample_equivalence_region

    sigma = 0.02
    truth = np.array([9.81, 0.1])
    times = np.linspace(0.2, 10.0, 50)
    rng = np.random.default_rng(seed)
    y = pendulum_solve(*truth, times) + sigma * rng.standard_normal(times.size)

    def misfit(theta):
        return np.array(
            [((pendulum_solve(k, b, times) - y) ** 2).sum() for k, b in np.atleast_2d(theta)]
        )

    bounds = [(6.0, 14.0), (0.0, 0.4)]
    out = sample_equivalence_region(
        misfit, bounds, None, sigma=sigma, n_particles=30, n_iterations=80, seed=seed
    )
    # The posterior is far narrower than the prior box: locate its peak from the
    # swarm's best, estimate the curvature by finite differences, and grid +-8 std.
    best = out.best.copy()
    h = np.array([0.01, 0.002])
    stds = []
    for j in range(2):
        e = np.zeros(2)
        e[j] = h[j]
        f0, fp, fm = misfit(np.stack([best, best + e, best - e]))
        curv = max((fp - 2 * f0 + fm) / h[j] ** 2, 1e-12)  # d2 misfit / d theta^2
        stds.append(math.sqrt(2 * sigma**2 / curv))  # std of exp(-misfit / 2 sigma^2)
    gk = np.linspace(best[0] - 8 * stds[0], best[0] + 8 * stds[0], 161)
    gb = np.linspace(best[1] - 8 * stds[1], best[1] + 8 * stds[1], 121)
    K, B = np.meshgrid(gk, gb, indexing="ij")
    log_post = -misfit(np.stack([K.ravel(), B.ravel()], axis=1)).reshape(K.shape) / (2 * sigma**2)
    return summarize("pendulum_k_b", seed, out, sigma, truth, [gk, gb], log_post, ["k", "b"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--heat", type=int, default=30)
    parser.add_argument("--pendulum", type=int, default=10)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for seed in range(max(args.heat, args.pendulum)):
        cases = []
        if seed < args.heat:
            cases.append(heat_case)
        if seed < args.pendulum:
            cases.append(pendulum_case)
        for case in cases:
            rec = case(seed)
            records.append(rec)
            p = rec["per_parameter"][0]
            print(
                f"{rec['problem']:<14} seed {seed}: ess {rec['ess']:.0f}/{rec['n_evaluated']} accepted {rec['n_accepted']} | "
                f"{p['name']}: exact std {p['exact_std']:.4f} weighted {p['weighted_std']:.4f} accepted {p['accepted_std']:.4f} "
                f"KS {p['kolmogorov_weighted']:.3f}/{p['kolmogorov_accepted']:.3f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
