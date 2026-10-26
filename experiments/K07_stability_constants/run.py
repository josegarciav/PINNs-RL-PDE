"""K07: how does the training residual translate into solution error?

Catalogue problems with a closed form are trained for increasing budgets; at
each budget the residual and the error are evaluated on a dense independent
sample (pinnrl.analysis.stability_ratio), giving the empirical stability
constant C = error / residual of the estimate error <= C (residual + quadrature).

Usage: python experiments/K07_stability_constants/run.py [--seeds 2] [--workers 4]
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
PROBLEMS = [
    "heat",
    "convection",
    "wave",
    "klein_gordon",
    "poisson",
    "burgers_travelling_wave",
    "beam",
    "kdv_two_parameter",
    "heat_2d",
    "sine_gordon",
]
BUDGETS = [500, 1000, 2000, 4000, 8000]


def run_case(job):
    key, seed = job
    import torch

    from pinnrl.analysis import stability_ratio
    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    pde = from_catalog(key)
    extra = {}
    if key == "kdv_two_parameter":  # collapses to zero without data
        pde.generate_synthetic_observations(n_points=200, noise_std=0.0, seed=seed)
        extra = {
            "mode": "data_augmented",
            "loss_weights": {"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
        }
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
        **({"loss_weights": {"residual": 1.0, "boundary": 10.0, "initial": 10.0}} | extra),
    )
    rows = []
    done = 0
    t0 = time.time()
    for budget in BUDGETS:
        trainer.train_steps(num_steps=budget - done, resample_every=100)
        done = budget
        sr = stability_ratio(pde, trainer.model, n_points=8192, seed=seed)
        rows.append(
            {
                "problem": key,
                "seed": seed,
                "steps": budget,
                "error": sr.error,
                "residual": sr.residual,
                "boundary_error": sr.boundary_error,
                "initial_error": sr.initial_error,
                "ratio": sr.ratio,
                "train_time_s": time.time() - t0,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, s) for s in range(args.seeds) for k in PROBLEMS]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(run_case, jobs):
            records.extend(rows)
            r = rows[-1]
            print(
                f"{r['problem']:<24} seed {r['seed']}: "
                + "  ".join(
                    f"{x['steps']}: e {x['error']:.3f} r {x['residual']:.2e} C {x['ratio']:.1f}"
                    for x in rows
                )
                + f"  ({r['train_time_s']:.0f}s, elapsed {time.time() - t0:.0f}s)",
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
