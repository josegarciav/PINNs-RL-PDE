"""F05: plant-like data. A few fixed probe positions, densely sampled in time.

Layouts at an equal data count (200 points): random scatter over the domain,
or 1, 2, 4 or 8 fixed x positions (equispaced in the interior) each read at
200 / n_sensors regular times. Noise 0.01, 0.05 and 0.1; 10 seeds. Each cell
is compared with the Cramér–Rao bound for its own sensor layout.

Usage: python experiments/F05_fixed_sensors/run.py [--seeds 10] [--steps 5000] [--workers 4]
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
ALPHA_TRUE = 0.1
ALPHA_GUESS = 0.5
N_TOTAL = 200
LAYOUTS = ["random", 1, 2, 4, 8]
NOISES = [0.01, 0.05, 0.1]
TRAJECTORY_EVERY = 50


def observations(layout, seed):
    """Observation points (x, t) for a layout; the noise is drawn by the caller."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    if layout == "random":
        x = torch.rand(N_TOTAL, 1, generator=gen)
        t = torch.rand(N_TOTAL, 1, generator=gen)
        return x, t
    n_s = int(layout)
    m = N_TOTAL // n_s
    positions = torch.arange(1, n_s + 1, dtype=torch.float32) / (n_s + 1)
    times = (torch.arange(m, dtype=torch.float32) + 0.5) / m
    x = positions.repeat_interleave(m)[:, None]
    t = times.repeat(n_s)[:, None]
    return x, t


def run_cell(job: tuple[object, float, int, int]) -> dict:
    """Train one inverse problem and return its record."""
    layout, noise, seed, steps = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    x, t = observations(layout, seed)
    u_true = torch.exp(-ALPHA_TRUE * math.pi**2 * t) * torch.sin(math.pi * x)
    gen = torch.Generator().manual_seed(10_000 + seed)
    u = u_true + noise * torch.randn(u_true.shape, generator=gen)
    sens = (-(math.pi**2) * t * u_true).pow(2).sum().item()
    crlb_std = noise / math.sqrt(sens)

    pde = from_catalog(
        "heat",
        parameters={"alpha": ALPHA_TRUE},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_GUESS},
        observation_data={"x": x, "t": t, "u": u},
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
        trajectory.append(pde.get_trainable_parameter_values()["alpha"])
    alpha_hat = trajectory[-1]
    return {
        "layout": str(layout),
        "noise": noise,
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
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = [
        (layout, noise, seed, args.steps)
        for seed in range(args.seeds)
        for noise in NOISES
        for layout in LAYOUTS
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] layout={rec['layout']:<6} noise={rec['noise']:<5} seed={rec['seed']:<2} "
                f"alpha={rec['alpha_hat']:.4f} rel_err={rec['rel_error']:.3f} crlb={rec['crlb_rel_std']:.3f} "
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
