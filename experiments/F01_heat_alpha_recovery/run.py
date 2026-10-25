"""F01: recovery of the heat diffusivity alpha vs observation noise and count.

For every (noise, n_obs, seed) cell: draw n_obs observations of the exact solution
at uniform random points, add Gaussian noise, start alpha at 5x the truth and train
the PINN in inverse mode. Records the final estimate, its trajectory, the field
error and the Cramér–Rao bound computed from the exact sensitivity at the observed
points.

Usage: python experiments/F01_heat_alpha_recovery/run.py [--seeds 10] [--steps 2000]
       [--workers 4] [--out experiments/F01_heat_alpha_recovery/results]
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

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
NOISES = [0.0, 0.01, 0.05, 0.1]
N_OBS = [20, 50, 200, 1000]
ALPHA_TRUE = 0.1
ALPHA_GUESS = 0.5
TRAJECTORY_EVERY = 50


def run_cell(job: tuple[float, int, int, int]) -> dict:
    """Train one inverse problem and return its record."""
    noise, n_obs, seed, steps = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    pde = from_catalog(
        "heat",
        parameters={"alpha": ALPHA_TRUE},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_GUESS},
    )
    obs = pde.generate_synthetic_observations(n_points=n_obs, noise_std=noise, seed=seed)

    # Cramér–Rao bound from the exact sensitivity du/dalpha = -pi^2 t u at the
    # observed points. With noise 0 the bound is 0 and the error is the PINN's own.
    x, t = obs["x"].cpu(), obs["t"].cpu()
    u_true = math.e ** (-ALPHA_TRUE * math.pi**2 * t) * torch.sin(math.pi * x)
    sens = (-(math.pi**2) * t * u_true).pow(2).sum().item()
    crlb_std = noise / math.sqrt(sens) if noise > 0 else 0.0

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
    # Re-seed the network init after the sampler drew its points so the init
    # depends on the seed only (same init for every cell with this seed).
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
        "noise": noise,
        "n_obs": n_obs,
        "seed": seed,
        "steps": steps,
        "alpha_hat": alpha_hat,
        "rel_error": abs(alpha_hat - ALPHA_TRUE) / ALPHA_TRUE,
        "crlb_rel_std": crlb_std / ALPHA_TRUE,
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
    parser.add_argument("--steps", type=int, default=2000)
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
                f"alpha={rec['alpha_hat']:.4f} rel_err={rec['rel_error']:.3f} "
                f"l2={rec['l2_error']:.3f} {rec['train_time_s']:.0f}s  (elapsed {time.time() - t0:.0f}s)",
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
