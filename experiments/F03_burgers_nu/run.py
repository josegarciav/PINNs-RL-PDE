"""F03: recovery of the Burgers viscosity nu = 0.01/pi near the shock.

Same noise x observation grid as F01. Observations are nodes of a fine Cole–Hopf
reference (pinnrl.benchmarks.reference.burgers_reference), so no interpolation
error enters. The viscosity is identified as log nu, written into the residual as
exp(log_nu): Adam's step is then relative, and nu stays positive. The Cramér–Rao
bound uses du/dnu by central differences of the reference at nu(1 +- 1 %).

Usage: python experiments/F03_burgers_nu/run.py [--seeds 10] [--steps 10000] [--workers 4]
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
NU_TRUE = 0.01 / math.pi
GUESS_FACTOR = 5.0
NOISES = [0.0, 0.01, 0.05, 0.1]
N_OBS = [20, 50, 200, 1000]
NX, NT = 1601, 161
TRAJECTORY_EVERY = 100
REFERENCE = HERE / "results" / "reference.npz"


def reference():
    """The fine reference and its nu-sensitivity, cached on disk."""
    if REFERENCE.exists():
        data = np.load(REFERENCE)
        return data["x"], data["t"], data["u"], data["du_dnu"]
    from pinnrl.benchmarks.reference import burgers_reference

    ref = burgers_reference(nu=NU_TRUE, nx=NX, nt=NT)
    h = 0.01 * NU_TRUE
    plus = burgers_reference(nu=NU_TRUE + h, nx=NX, nt=NT).u
    minus = burgers_reference(nu=NU_TRUE - h, nx=NX, nt=NT).u
    du_dnu = (plus - minus) / (2 * h)
    REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(REFERENCE, x=ref.x, t=ref.t, u=ref.u, du_dnu=du_dnu)
    return ref.x, ref.t, ref.u, du_dnu


def run_cell(job: tuple[float, int, int, int]) -> dict:
    """Train one inverse problem and return its record."""
    noise, n_obs, seed, steps = job
    import torch

    from pinnrl.pdes.custom import CustomPDE

    torch.set_num_threads(2)
    x_grid, t_grid, u_grid, s_grid = reference()
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(x_grid), n_obs)
    jj = rng.integers(1, len(t_grid), n_obs)  # t > 0: t = 0 is the initial condition
    x = x_grid[ii]
    t = t_grid[jj]
    u = u_grid[ii, jj] + noise * rng.standard_normal(n_obs)
    sens = float((s_grid[ii, jj] ** 2).sum())
    crlb_rel_std = noise / math.sqrt(sens) / NU_TRUE if noise > 0 else 0.0

    pde = CustomPDE(
        "u_t + u*u_x - exp(log_nu)*u_xx",
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"log_nu": math.log(NU_TRUE)},
        initial_condition="-sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        trainable_parameters=["log_nu"],
        parameter_initial_guesses={"log_nu": math.log(GUESS_FACTOR * NU_TRUE)},
        observation_data={
            "x": torch.tensor(x, dtype=torch.float32)[:, None],
            "t": torch.tensor(t, dtype=torch.float32)[:, None],
            "u": torch.tensor(u, dtype=torch.float32)[:, None],
        },
        name="Burgers (log nu)",
    )
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
        trajectory.append(math.exp(pde.get_trainable_parameter_values()["log_nu"]))
    nu_hat = trajectory[-1]
    # Field error against the reference grid (every 8th node to keep it cheap).
    xx, tt = np.meshgrid(x_grid[::8], t_grid[::2], indexing="ij")
    with torch.no_grad():
        pred = trainer.model(
            torch.tensor(np.stack([xx.ravel(), tt.ravel()], 1), dtype=torch.float32)
        ).numpy()[:, 0]
    ref = u_grid[::8, ::2].ravel()
    return {
        "noise": noise,
        "n_obs": n_obs,
        "seed": seed,
        "steps": steps,
        "nu_hat": nu_hat,
        "rel_error": abs(nu_hat - NU_TRUE) / NU_TRUE,
        "crlb_rel_std": crlb_rel_std,
        "l2_error": float(np.linalg.norm(pred - ref) / np.linalg.norm(ref)),
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
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    reference()
    print(f"reference ready in {time.time() - t0:.0f}s", flush=True)

    jobs = [
        (noise, n_obs, seed, args.steps)
        for seed in range(args.seeds)
        for noise in NOISES
        for n_obs in N_OBS
    ]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] noise={rec['noise']:<5} n_obs={rec['n_obs']:<5} seed={rec['seed']:<2} "
                f"nu={rec['nu_hat']:.5f} rel_err={rec['rel_error']:.3f} crlb={rec['crlb_rel_std']:.3f} "
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
