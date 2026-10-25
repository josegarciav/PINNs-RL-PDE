"""F02 follow-up: is the wrong minimum a basin of the optimizer or the surrogate's own optimum?

Three variants on the (noise 0.01, 200 observations) cell, 3 seeds each:
  - truth_start: (alpha, beta) started at the true (6, 1), 8,000 steps;
  - long:        started at (4, 1.5) as in F02, 24,000 steps;
  - data_weight: started at (4, 1.5), data weight 100 instead of 10, 8,000 steps.

Usage: python experiments/F02_kdv_joint/run_followup.py [--seeds 3] [--workers 2]
"""

from __future__ import annotations

import argparse
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
VARIANTS = {
    "truth_start": dict(guess={"alpha": 6.0, "beta": 1.0}, steps=8000, data_weight=10.0),
    "long": dict(guess={"alpha": 4.0, "beta": 1.5}, steps=24000, data_weight=10.0),
    "data_weight": dict(guess={"alpha": 4.0, "beta": 1.5}, steps=8000, data_weight=100.0),
}
NOISE, N_OBS = 0.01, 200


def run_cell(job):
    variant, seed = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    cfg = VARIANTS[variant]
    torch.set_num_threads(2)
    pde = from_catalog(
        "kdv_two_parameter",
        parameters=dict(TRUE),
        trainable_parameters=list(TRUE),
        parameter_initial_guesses=dict(cfg["guess"]),
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
        loss_weights={
            "residual": 1.0,
            "boundary": 10.0,
            "initial": 10.0,
            "data": cfg["data_weight"],
        },
    )
    torch.manual_seed(seed)
    trainer.model.apply(_reset)
    t0 = time.time()
    trajectory = []
    done = 0
    while done < cfg["steps"]:
        out = trainer.train_steps(num_steps=200, resample_every=100)
        done += 200
        v = pde.get_trainable_parameter_values()
        trajectory.append([v["alpha"], v["beta"]])
    losses = pde.compute_loss(trainer.model, *pde.generate_collocation_points(2000))
    return {
        "variant": variant,
        "seed": seed,
        "steps": cfg["steps"],
        "alpha_hat": trajectory[-1][0],
        "beta_hat": trajectory[-1][1],
        "alpha_rel_error": abs(trajectory[-1][0] - 6) / 6,
        "beta_rel_error": abs(trajectory[-1][1] - 1),
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "residual_loss": float(losses["residual"].detach()),
        "data_loss": float(losses["data"].detach()),
        "final_loss": float(out["loss"][-1]),
        "train_time_s": time.time() - t0,
        "trajectory": trajectory,
    }


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    jobs = [(v, s) for s in range(args.seeds) for v in VARIANTS]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['variant']:<12} seed {rec['seed']}: alpha {rec['alpha_hat']:.3f} beta {rec['beta_hat']:.3f} "
                f"err ({rec['alpha_rel_error']:.3f}, {rec['beta_rel_error']:.3f}) l2 {rec['l2_error']:.3f} "
                f"res {rec['residual_loss']:.2e} data {rec['data_loss']:.2e} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
            (args.out / "followup.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "followup.json")


if __name__ == "__main__":
    main()
