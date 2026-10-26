"""F06 / K10: are the intervals for the heat diffusivity calibrated?

For each of ``--datasets`` synthetic data sets (200 observations, noise 0.05,
F01 cell) an ensemble of ``--members`` networks is trained from different
initializations. Four 90 % intervals per data set are compared by their
coverage of the truth across data sets, their width and their CRPS:

  ensemble   : empirical central interval of the member estimates;
  laplace    : Gaussian at member 0's estimate with the surrogate Fisher covariance
               (pinnrl.uq.laplace_posterior);
  fisher     : Gaussian at member 0's estimate with the closed-form bound (what an
               efficient unbiased estimator would report);
  swarm      : the region of equivalence of the exact misfit sampled by the swarm
               (pinnrl.uq.sample_equivalence_region), i.e. the nonlinear interval
               the physics alone implies.

Usage: python experiments/F06_coverage/run.py [--datasets 30] [--members 8] [--steps 5000] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
ALPHA_TRUE, ALPHA_GUESS, N_OBS, NOISE, LEVEL = 0.1, 0.5, 200, 0.05, 0.9


def run_member(job):
    dataset, member, steps = job
    import torch

    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.uq import laplace_posterior

    torch.set_num_threads(2)
    pde = from_catalog(
        "heat",
        parameters={"alpha": ALPHA_TRUE},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_GUESS},
    )
    pde.generate_synthetic_observations(n_points=N_OBS, noise_std=NOISE, seed=dataset)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        num_collocation_points=1000,
        batch_size=1000,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
    )
    torch.manual_seed(1000 * dataset + member)
    trainer.model.apply(_reset)
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    alpha_hat = pde.get_trainable_parameter_values()["alpha"]
    rec = {
        "dataset": dataset,
        "member": member,
        "alpha_hat": alpha_hat,
        "train_time_s": time.time() - t0,
    }
    if member == 0:
        post = laplace_posterior(pde, trainer.model, ["alpha"], NOISE)
        rec["laplace_std"] = float(post.std[0])
    return rec


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def physics_intervals(dataset):
    """Closed-form Fisher std and the swarm region of the exact misfit for one data set."""
    import torch

    from pinnrl.analysis import closed_form_bound
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.uq import misfit_tolerance, sample_equivalence_region

    pde = from_catalog("heat")
    obs = pde.generate_synthetic_observations(n_points=N_OBS, noise_std=NOISE, seed=dataset)
    std = closed_form_bound(pde, ["alpha"], obs["x"], obs["t"], NOISE).std[0].item()
    x, t, y = obs["x"].numpy(), obs["t"].numpy(), obs["u"].numpy()

    def misfit(theta):
        pred = np.exp(-theta[:, 0:1] * math.pi**2 * t.T) * np.sin(math.pi * x.T)
        return ((pred - y.T) ** 2).sum(axis=1)

    out = sample_equivalence_region(
        misfit,
        [(0.02, 0.5)],
        misfit_tolerance(N_OBS, NOISE, 0.1),
        n_particles=30,
        n_iterations=80,
        seed=dataset,
    )
    lo, hi = out.interval(LEVEL)[:, 0]
    return {
        "fisher_std": std,
        "swarm_lo": float(lo),
        "swarm_hi": float(hi),
        "swarm_best": float(out.best[0]),
        "swarm_n": int(out.accepted.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", type=int, default=30)
    parser.add_argument("--members", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(d, m, args.steps) for d in range(args.datasets) for m in range(args.members)]
    members = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_member, jobs), 1):
            members.append(rec)
            if i % args.members == 0:
                print(
                    f"[{i}/{len(jobs)}] dataset {rec['dataset']} done (elapsed {time.time() - t0:.0f}s)",
                    flush=True,
                )
    (args.out / "members.json").write_text(json.dumps(members, indent=1))

    from pinnrl.uq import (
        coverage,
        crps_gaussian,
        crps_samples,
        ensemble_summary,
        gaussian_interval,
        mean_width,
    )

    rows = []
    for d in range(args.datasets):
        est = [
            m["alpha_hat"]
            for m in sorted((m for m in members if m["dataset"] == d), key=lambda m: m["member"])
        ]
        summ = ensemble_summary([[e] for e in est])
        lap_std = next(m for m in members if m["dataset"] == d and m["member"] == 0)["laplace_std"]
        phys = physics_intervals(d)
        e_lo, e_hi = summ.interval(LEVEL)[:, 0]
        g_lo, g_hi = summ.interval(LEVEL, gaussian=True)[:, 0]
        l_lo, l_hi = gaussian_interval(est[0], lap_std, LEVEL)
        f_lo, f_hi = gaussian_interval(est[0], phys["fisher_std"], LEVEL)
        rows.append(
            {
                "dataset": d,
                "alpha_member0": est[0],
                "ensemble_mean": float(summ.mean[0]),
                "ensemble_std": float(summ.std[0]),
                "ensemble_dispersion": summ.dispersion,
                "laplace_std": lap_std,
                "fisher_std": phys["fisher_std"],
                "ensemble_lo": float(e_lo),
                "ensemble_hi": float(e_hi),
                "ensemble_gauss_lo": float(g_lo),
                "ensemble_gauss_hi": float(g_hi),
                "laplace_lo": l_lo,
                "laplace_hi": l_hi,
                "fisher_lo": f_lo,
                "fisher_hi": f_hi,
                "swarm_lo": phys["swarm_lo"],
                "swarm_hi": phys["swarm_hi"],
                "swarm_best": phys["swarm_best"],
                "swarm_n": phys["swarm_n"],
                "crps_ensemble": crps_samples(est, ALPHA_TRUE),
                "crps_laplace": crps_gaussian(est[0], lap_std, ALPHA_TRUE),
                "crps_fisher": crps_gaussian(est[0], phys["fisher_std"], ALPHA_TRUE),
            }
        )
    with open(args.out / "intervals.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    truth = [ALPHA_TRUE] * len(rows)
    summary = {}
    for name in ["ensemble", "ensemble_gauss", "laplace", "fisher", "swarm"]:
        lo = [r[f"{name}_lo"] for r in rows]
        hi = [r[f"{name}_hi"] for r in rows]
        summary[name] = {"coverage": coverage(lo, hi, truth), "mean_width": mean_width(lo, hi)}
    for name in ["ensemble", "laplace", "fisher"]:
        summary[name]["crps"] = float(np.mean([r[f"crps_{name}"] for r in rows]))
    summary["bias_member0"] = float(np.mean([r["alpha_member0"] for r in rows]) - ALPHA_TRUE)
    summary["spread_member0_across_datasets"] = float(np.std([r["alpha_member0"] for r in rows]))
    summary["mean_ensemble_std"] = float(np.mean([r["ensemble_std"] for r in rows]))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
