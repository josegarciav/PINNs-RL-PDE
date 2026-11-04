"""K08: how do stability constants and quadrature error grow with dimension?

Heat and convection in d = 1..5 spatial dimensions, built with CustomPDE so the
solution has the same shape and time scale in every d:

* heat: u_t = (alpha/d) sum_i u_{x_i x_i}, u = exp(-alpha pi^2 t) prod_i sin(pi x_i),
  Dirichlet 0 (the decay rate is the same for every d);
* convection: u_t + (1/d) sum_i u_{x_i} = 0, u = prod_i sin(2 pi (x_i - t/d)), periodic.

Two sampling arms: a fixed pool of 4,096 collocation points whatever d
("matched total"), and a pool of 6^(d+1) points ("matched per axis", 36 to
46,656). Batches are capped at 4,096. At the end of training the residual and
the error are measured on 8,192 fresh points (pinnrl.analysis.stability_ratio):
the empirical stability constant C = error / residual; and the quadrature gap,
the relative difference between the mean-squared residual on the training
pool and on the fresh sample, with the spread of that estimate over 8
independent samples of the pool's size.

Usage: python experiments/K08_dimension_scaling/run.py [--seeds 2] [--steps 4000] [--workers 4]
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
DIMS = [1, 2, 3, 4, 5]
ARMS = {"matched total": lambda d: 4096, "matched per axis": lambda d: 6 ** (d + 1)}
BATCH_CAP = 4096
ALPHA = 0.1


def build(problem: str, d: int):
    """The d-dimensional problem as a CustomPDE with a closed form."""
    from pinnrl import CustomPDE

    xs = [f"x{i}" for i in range(1, d + 1)]
    domain = {x: (0.0, 1.0) for x in xs}
    domain["t"] = (0.0, 1.0)
    if problem == "heat":
        lap = " + ".join(f"u_{x}{x}" for x in xs)
        prod = "*".join(f"sin(pi*{x})" for x in xs)
        return CustomPDE(
            f"u_t - {ALPHA / d}*({lap})",
            domain=domain,
            initial_condition=prod,
            boundary_conditions={"dirichlet": {"value": 0.0}},
            exact_solution=f"exp(-{ALPHA}*pi^2*t)*{prod}",
            name=f"heat {d}-D",
        )
    grad = " + ".join(f"u_{x}" for x in xs)
    prod = "*".join(f"sin(2*pi*({x} - t/{d}))" for x in xs)
    prod0 = "*".join(f"sin(2*pi*{x})" for x in xs)
    return CustomPDE(
        f"u_t + {1.0 / d}*({grad})",
        domain=domain,
        initial_condition=prod0,
        boundary_conditions={"periodic": {}},
        exact_solution=prod,
        name=f"convection {d}-D",
    )


def run_case(job: tuple[str, int, str, int, int]) -> dict:
    problem, d, arm, seed, steps = job
    import torch

    from pinnrl.analysis import stability_ratio

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    pde = build(problem, d)
    pool = ARMS[arm](d)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        num_collocation_points=pool,
        batch_size=min(pool, BATCH_CAP),
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    train_time = time.time() - t0
    sr = stability_ratio(pde, trainer.model, n_points=8192, seed=seed)

    # Quadrature gap: the mean-squared residual on pools of the training size vs the dense sample.
    gen = torch.Generator().manual_seed(1000 + seed)
    lo = torch.zeros(d)
    hi = torch.ones(d)
    x_dense = lo + (hi - lo) * torch.rand(8192, d, generator=gen)
    t_dense = torch.rand(8192, 1, generator=gen)
    dense = (pde.compute_residual(trainer.model, x_dense, t_dense).detach() ** 2).mean().item()
    pools = []
    for _ in range(8):
        xp = torch.rand(min(pool, BATCH_CAP), d, generator=gen)
        tp = torch.rand(min(pool, BATCH_CAP), 1, generator=gen)
        pools.append((pde.compute_residual(trainer.model, xp, tp).detach() ** 2).mean().item())
    pools_t = torch.tensor(pools)
    return {
        "problem": problem,
        "dimension": d,
        "arm": arm,
        "pool": pool,
        "batch": min(pool, BATCH_CAP),
        "seed": seed,
        "steps": steps,
        "error": sr.error,
        "residual": sr.residual,
        "boundary_error": sr.boundary_error,
        "initial_error": sr.initial_error,
        "ratio": sr.ratio,
        "mse_residual_dense": dense,
        "quadrature_gap": float((pools_t.mean() - dense).abs() / dense),
        "quadrature_spread": float(pools_t.std() / dense),
        "train_time_s": train_time,
        "time_per_step_ms": 1000 * train_time / steps,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (problem, d, arm, seed, args.steps)
        for seed in range(args.seeds)
        for problem in ("heat", "convection")
        for arm in ARMS
        for d in DIMS
    ]
    # Largest first, so the slow 5-D runs do not trail at the end.
    jobs.sort(key=lambda j: -j[1])
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_case, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['problem']:<10} d={rec['dimension']} {rec['arm']:<16} seed={rec['seed']} "
                f"e={rec['error']:.3f} r={rec['residual']:.2e} C={rec['ratio']:.1f} gap={rec['quadrature_gap']:.3f} "
                f"{rec['time_per_step_ms']:.0f} ms/step (elapsed {time.time() - t0:.0f}s)",
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
