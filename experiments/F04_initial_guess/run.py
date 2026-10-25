"""F04: does the recovered heat diffusivity depend on the initial guess?

Ten initial guesses from 0.1x to 10x the true alpha, one data set per seed
(200 observations, noise 0.05, as in the F01 cell), 10 seeds. Records the
final estimate, its trajectory and the first step after which it stays
within 5 % of the truth.

Usage: python experiments/F04_initial_guess/run.py [--seeds 10] [--steps 5000] [--workers 4]
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
GUESS_FACTORS = np.logspace(-1, 1, 10)
N_OBS = 200
NOISE = 0.05
TRAJECTORY_EVERY = 50


def run_cell(job: tuple[float, int, int]) -> dict:
    """Train one inverse problem from one initial guess and return its record."""
    factor, seed, steps = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    guess = ALPHA_TRUE * factor
    pde = from_catalog(
        "heat",
        parameters={"alpha": ALPHA_TRUE},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": guess},
    )
    pde.generate_synthetic_observations(n_points=N_OBS, noise_std=NOISE, seed=seed)
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
    torch.manual_seed(seed)
    trainer.model.apply(_reset)
    t0 = time.time()
    trajectory = []
    done = 0
    while done < steps:
        chunk = min(TRAJECTORY_EVERY, steps - done)
        out = trainer.train_steps(num_steps=chunk, resample_every=TRAJECTORY_EVERY)
        done += chunk
        trajectory.append(pde.get_trainable_parameter_values()["alpha"])
    alpha_hat = trajectory[-1]
    return {
        "guess_factor": float(factor),
        "guess": guess,
        "seed": seed,
        "steps": steps,
        "alpha_hat": alpha_hat,
        "rel_error": abs(alpha_hat - ALPHA_TRUE) / ALPHA_TRUE,
        "settle_step": settle_step(trajectory),
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "final_loss": float(out["loss"][-1]),
        "train_time_s": time.time() - t0,
        "trajectory": trajectory,
    }


def settle_step(trajectory, tol=0.05):
    """First step after which the estimate stays within ``tol`` of the truth, else None."""
    ok = np.abs(np.asarray(trajectory) - ALPHA_TRUE) / ALPHA_TRUE < tol
    for i in range(len(ok)):
        if ok[i:].all():
            return (i + 1) * TRAJECTORY_EVERY
    return None


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [(f, seed, args.steps) for seed in range(args.seeds) for f in GUESS_FACTORS]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] guess={rec['guess']:.4f} seed={rec['seed']:<2} "
                f"alpha={rec['alpha_hat']:.4f} rel_err={rec['rel_error']:.3f} "
                f"settle={rec['settle_step']} {rec['train_time_s']:.0f}s  (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    fields = [k for k in records[0] if k != "trajectory"]
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fields})
    print("saved", args.out / "runs.csv")


if __name__ == "__main__":
    main()
