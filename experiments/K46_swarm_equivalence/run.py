"""K46: does an exploratory swarm recover the nonlinear region of equivalence?

The forward map is the closed form (heat alpha; the two-parameter KdV soliton;
the a(1 - exp(-b x)) example of Pallero et al., 2018), so the region sampled is
the one the physics implies for a given noisy data set, independent of any
surrogate. For each problem and 10 data sets: the swarm's 90 % interval and
ranges from the accepted samples, the correlation of the accepted samples for
two-parameter problems, and the Fisher ellipse at the truth, with the fraction
of accepted samples inside the 90 % Fisher ellipse (1 for a linear problem).

Usage: python experiments/K46_swarm_equivalence/run.py [--datasets 10]
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
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent


def heat_case(seed):
    from pinnrl.analysis import closed_form_bound
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.uq import misfit_tolerance, sample_equivalence_region

    pde = from_catalog("heat")
    obs = pde.generate_synthetic_observations(n_points=200, noise_std=0.05, seed=seed)
    x, t, y = obs["x"].numpy().T, obs["t"].numpy().T, obs["u"].numpy().T
    bound = closed_form_bound(pde, ["alpha"], obs["x"], obs["t"], 0.05)

    def misfit(theta):
        return ((np.exp(-theta[:, 0:1] * math.pi**2 * t) * np.sin(math.pi * x) - y) ** 2).sum(
            axis=1
        )

    out = sample_equivalence_region(
        misfit,
        [(0.02, 0.5)],
        misfit_tolerance(200, 0.05, 0.1),
        n_particles=30,
        n_iterations=100,
        seed=seed,
    )
    return summarize("heat", seed, out, np.array([0.1]), bound)


def kdv_case(seed):
    from pinnrl.analysis import closed_form_bound
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.uq import misfit_tolerance, sample_equivalence_region

    pde = from_catalog("kdv_two_parameter")
    obs = pde.generate_synthetic_observations(n_points=200, noise_std=0.01, seed=seed)
    x, t, y = obs["x"].numpy().T, obs["t"].numpy().T, obs["u"].numpy().T
    bound = closed_form_bound(pde, ["alpha", "beta"], obs["x"], obs["t"], 0.01)

    def misfit(theta):
        a, b = theta[:, 0:1], theta[:, 1:2]
        pred = 3 / a / np.cosh(0.5 * np.sqrt(1 / b) * (x - t)) ** 2
        return ((pred - y) ** 2).sum(axis=1)

    out = sample_equivalence_region(
        misfit,
        [(2, 12), (0.3, 3)],
        misfit_tolerance(200, 0.01, 0.1),
        n_particles=40,
        n_iterations=150,
        seed=seed,
    )
    return summarize("kdv_two_parameter", seed, out, np.array([6.0, 1.0]), bound)


def saturation_case(seed):
    from pinnrl.analysis import fisher_information
    from pinnrl.uq import misfit_tolerance, sample_equivalence_region

    xs = np.linspace(1, 30, 40)
    true = np.array([20.0, 0.1])
    rng = np.random.default_rng(seed)
    sigma = 0.3
    y = true[0] * (1 - np.exp(-true[1] * xs)) + sigma * rng.normal(size=xs.size)
    J = np.stack([1 - np.exp(-true[1] * xs), true[0] * xs * np.exp(-true[1] * xs)], axis=1)
    bound = fisher_information(torch.tensor(J), sigma, ["a", "b"])

    def misfit(theta):
        pred = theta[:, :1] * (1 - np.exp(-theta[:, 1:2] * xs[None, :]))
        return ((pred - y[None, :]) ** 2).sum(axis=1)

    out = sample_equivalence_region(
        misfit,
        [(5, 40), (0.02, 0.3)],
        misfit_tolerance(xs.size, sigma, 0.1),
        n_particles=40,
        n_iterations=150,
        seed=seed,
    )
    return summarize("saturation_a_b", seed, out, true, bound)


def summarize(problem, seed, out, truth, bound):
    s = out.samples
    cov = bound.covariance.numpy()
    d = s - truth[None, :]
    mahal = np.einsum("ij,jk,ik->i", d, np.linalg.inv(cov), d)
    chi2_90 = {1: 2.706, 2: 4.605}[truth.size]
    iv = out.interval(0.9)
    rec = {
        "problem": problem,
        "seed": seed,
        "n_evaluated": int(out.positions.shape[0]),
        "n_accepted": int(out.accepted.sum()),
        "final_dispersion": float(out.dispersion[-1]),
        "best": out.best.tolist(),
        "best_misfit": out.best_misfit,
        "tolerance": out.tolerance,
        "interval_lo": iv[0].tolist(),
        "interval_hi": iv[1].tolist(),
        "range_lo": out.ranges()[0].tolist(),
        "range_hi": out.ranges()[1].tolist(),
        "fisher_std": bound.std.tolist(),
        "truth_in_interval": bool(np.all((iv[0] <= truth) & (truth <= iv[1]))),
        "fraction_inside_fisher_90": float(np.mean(mahal <= chi2_90)),
        "sample_std": s.std(axis=0).tolist(),
    }
    if truth.size == 2:
        rec["sample_correlation"] = float(np.corrcoef(s[:, 0], s[:, 1])[0, 1])
        rec["fisher_correlation"] = float(bound.correlation[0, 1])
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", type=int, default=10)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    for seed in range(args.datasets):
        for case in (heat_case, kdv_case, saturation_case):
            rec = case(seed)
            records.append(rec)
            print(
                f"{rec['problem']:<18} seed {seed}: accepted {rec['n_accepted']}/{rec['n_evaluated']} "
                f"interval {np.round(rec['interval_lo'], 4)}–{np.round(rec['interval_hi'], 4)} "
                f"fisher std {np.round(rec['fisher_std'], 4)} inside-ellipse {rec['fraction_inside_fisher_90']:.2f} "
                f"corr {rec.get('sample_correlation', float('nan')):+.2f}/{rec.get('fisher_correlation', float('nan')):+.2f} "
                f"dispersion {rec['final_dispersion']:.2f} ({time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
