"""F07: practical identifiability of the heat diffusivity by profile likelihood.

Fix alpha on a grid around the truth, retrain the network with alpha held
fixed (physics + data, no trainable parameter), and record the data misfit.
With Gaussian noise the profile of -2 log L = sum (y - u)^2 / sigma^2 against
alpha should be a parabola whose curvature is the Fisher information; a flat
profile means the parameter is not practically identifiable.

Data: the F01 cell with 200 observations and noise 0.05, one data set per seed.

Usage: python experiments/F07_profile_likelihood/run.py [--seeds 3] [--steps 5000] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
ALPHA_TRUE = 0.1
GRID = np.round(ALPHA_TRUE * np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]), 5)
N_OBS = 200
NOISE = 0.05


def run_cell(job):
    alpha, seed, steps = job
    import torch

    from pinnrl.analysis import closed_form_bound
    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    # Observations come from the TRUE alpha; the network is trained at the fixed grid alpha.
    truth = from_catalog("heat", parameters={"alpha": ALPHA_TRUE})
    obs = truth.generate_synthetic_observations(n_points=N_OBS, noise_std=NOISE, seed=seed)
    bound = closed_form_bound(truth, ["alpha"], obs["x"], obs["t"], NOISE)
    pde = from_catalog(
        "heat",
        parameters={"alpha": float(alpha)},
        observation_data={"x": obs["x"], "t": obs["t"], "u": obs["u"]},
    )
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        mode="data_augmented",
        num_collocation_points=1000,
        batch_size=1000,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
    )
    torch.manual_seed(seed)
    trainer.model.apply(_reset)
    t0 = time.time()
    out = trainer.train_steps(num_steps=steps, resample_every=50)
    trainer.model.eval()
    with torch.no_grad():
        pred = trainer.model(torch.cat([obs["x"], obs["t"]], 1))
        misfit = float(((pred - obs["u"]) ** 2).sum())
        # The same misfit for the exact solution at this alpha: the profile the physics would give.
        u_alpha = pde.exact_solution(obs["x"], obs["t"])
        misfit_exact = float(((u_alpha - obs["u"]) ** 2).sum())
    return {
        "alpha": float(alpha),
        "seed": seed,
        "steps": steps,
        "neg2loglik": misfit / NOISE**2,
        "neg2loglik_exact_solution": misfit_exact / NOISE**2,
        "fisher_information": float(bound.information[0, 0]),
        "residual_loss": float(
            trainer.pde.compute_loss(trainer.model, *pde.generate_collocation_points(2000))[
                "residual"
            ].detach()
        ),
        "final_loss": float(out["loss"][-1]),
        "train_time_s": time.time() - t0,
    }


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(a, seed, args.steps) for seed in range(args.seeds) for a in GRID]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] alpha={rec['alpha']:.3f} seed={rec['seed']} -2logL={rec['neg2loglik']:.1f} "
                f"(exact solution {rec['neg2loglik_exact_solution']:.1f}) F={rec['fisher_information']:.0f} "
                f"res {rec['residual_loss']:.2e} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print("saved", args.out / "runs.csv")


if __name__ == "__main__":
    main()
