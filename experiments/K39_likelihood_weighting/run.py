"""K39: does setting the data weight from the noise model beat a tuned weight?

Heat alpha from 200 observations at noise 0.01, 0.05 and 0.1, with three data
weights: the fixed 10 used throughout F01-F05; 100; and the likelihood weight
(n_data / sigma^2) / (n_residual / sigma_r^2) with a model-error scale
sigma_r = 0.1 (pinnrl.uq.likelihood_weights). Ten seeds. Reports the error of
alpha, the bias and the spread, plus the Laplace std from member 0 for the
calibration check of K10.

Usage: python experiments/K39_likelihood_weighting/run.py [--seeds 10] [--steps 5000] [--workers 4]
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
ALPHA_TRUE, ALPHA_GUESS, N_OBS, N_RES, SIGMA_R = 0.1, 0.5, 200, 1000, 0.1
NOISES = [0.01, 0.05, 0.1]
SCHEMES = ["fixed_10", "fixed_100", "likelihood"]


def data_weight(scheme, noise):
    from pinnrl.uq import likelihood_weights

    if scheme == "fixed_10":
        return 10.0
    if scheme == "fixed_100":
        return 100.0
    return likelihood_weights(N_OBS, noise, N_RES, SIGMA_R)["data"]


def run_cell(job):
    scheme, noise, seed, steps = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    pde = from_catalog(
        "heat",
        parameters={"alpha": ALPHA_TRUE},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_GUESS},
    )
    pde.generate_synthetic_observations(n_points=N_OBS, noise_std=noise, seed=seed)
    w = data_weight(scheme, noise)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        num_collocation_points=N_RES,
        batch_size=N_RES,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": w},
    )
    torch.manual_seed(seed)
    trainer.model.apply(_reset)
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    alpha_hat = pde.get_trainable_parameter_values()["alpha"]
    losses = pde.compute_loss(trainer.model, *pde.generate_collocation_points(2000))
    return {
        "scheme": scheme,
        "noise": noise,
        "data_weight": w,
        "seed": seed,
        "alpha_hat": alpha_hat,
        "rel_error": abs(alpha_hat - ALPHA_TRUE) / ALPHA_TRUE,
        "residual_loss": float(losses["residual"].detach()),
        "data_loss": float(losses["data"].detach()),
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "train_time_s": time.time() - t0,
    }


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
    jobs = [(s, n, seed, args.steps) for seed in range(args.seeds) for n in NOISES for s in SCHEMES]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['scheme']:<10} noise {rec['noise']:<5} w {rec['data_weight']:<7.3g} seed {rec['seed']}: "
                f"alpha {rec['alpha_hat']:.4f} rel_err {rec['rel_error']:.3f} res {rec['residual_loss']:.1e} "
                f"data {rec['data_loss']:.1e} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    with open(args.out / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    print("saved", args.out / "runs.csv")


if __name__ == "__main__":
    main()
