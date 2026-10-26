"""F02: are the two KdV coefficients jointly identifiable?

u_t + alpha u u_x + beta u_xxx = 0 with the soliton u = 3/alpha sech^2(sqrt(1/beta)/2 (x - t)),
true (alpha, beta) = (6, 1), started at (4, 1.5). Observations at uniform random
points of the domain with Gaussian noise; the 2x2 Fisher information from the
closed form gives the Cramér–Rao bound and the predicted correlation of the two
estimates, which are compared with the spread across seeds.

Usage: python experiments/F02_kdv_joint/run.py [--seeds 10] [--steps 8000] [--workers 4]
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

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
TRUE = {"alpha": 6.0, "beta": 1.0}
GUESS = {"alpha": 4.0, "beta": 1.5}
NOISES = [0.01, 0.05]
N_OBS = [50, 200, 1000]
TRAJECTORY_EVERY = 100


def run_cell(job: tuple[float, int, int, int]) -> dict:
    """Train one inverse problem and return its record."""
    noise, n_obs, seed, steps = job
    import torch

    from pinnrl.analysis import closed_form_bound
    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    pde = from_catalog(
        "kdv_two_parameter",
        parameters=dict(TRUE),
        trainable_parameters=list(TRUE),
        parameter_initial_guesses=dict(GUESS),
    )
    obs = pde.generate_synthetic_observations(n_points=n_obs, noise_std=noise, seed=seed)
    bound = closed_form_bound(pde, list(TRUE), obs["x"].cpu(), obs["t"].cpu(), noise)

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
        vals = pde.get_trainable_parameter_values()
        trajectory.append([vals["alpha"], vals["beta"]])
    alpha_hat, beta_hat = trajectory[-1]
    rel_std = bound.relative_std(TRUE)
    return {
        "noise": noise,
        "n_obs": n_obs,
        "seed": seed,
        "steps": steps,
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "alpha_rel_error": abs(alpha_hat - TRUE["alpha"]) / TRUE["alpha"],
        "beta_rel_error": abs(beta_hat - TRUE["beta"]) / TRUE["beta"],
        "crlb_alpha_rel_std": rel_std["alpha"],
        "crlb_beta_rel_std": rel_std["beta"],
        "crlb_correlation": float(bound.correlation[0, 1]),
        "fisher_condition_number": bound.condition_number,
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "final_loss": float(out["loss"][-1]),
        "train_time_s": time.time() - t0,
        "trajectory": trajectory,
    }


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (noise, n_obs, seed, args.steps)
        for seed in range(args.seeds)
        for noise in NOISES
        for n_obs in N_OBS
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] noise={rec['noise']:<5} n_obs={rec['n_obs']:<5} seed={rec['seed']:<2} "
                f"alpha={rec['alpha_hat']:.3f} beta={rec['beta_hat']:.3f} "
                f"err=({rec['alpha_rel_error']:.3f}, {rec['beta_rel_error']:.3f}) "
                f"crlb=({rec['crlb_alpha_rel_std']:.3f}, {rec['crlb_beta_rel_std']:.3f}) "
                f"corr={rec['crlb_correlation']:+.2f} l2={rec['l2_error']:.3f} "
                f"{rec['train_time_s']:.0f}s  (elapsed {time.time() - t0:.0f}s)",
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
