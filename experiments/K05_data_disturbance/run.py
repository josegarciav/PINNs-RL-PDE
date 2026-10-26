"""K05: how far does the estimate move under sensor noise, bias, dropout, delay and outliers?

Base case: the F01 heat problem with 200 observations at noise 0.05 (random
scatter) and, for the single-probe faults, the F05 layout with 4 probes. Each
disturbance is applied to the same data set as its base run (same seed, same
initial weights), so the shift of the estimate is paired, and it is compared
with the linear prediction delta_alpha = (J^T J)^-1 J^T delta_y from the exact
sensitivity (pinnrl.analysis.linear_response).

Usage: python experiments/K05_data_disturbance/run.py [--seeds 5] [--steps 5000] [--workers 4]
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
ALPHA_TRUE, ALPHA_GUESS, N_OBS, NOISE = 0.1, 0.5, 200, 0.05
CELLS = [
    ("scatter", "none", 0.0),
    ("scatter", "noise", 0.05),  # extra noise on top of the base 0.05
    ("scatter", "noise", 0.10),
    ("scatter", "bias", 0.01),
    ("scatter", "bias", 0.02),
    ("scatter", "bias", 0.05),
    ("scatter", "dropout", 0.25),
    ("scatter", "dropout", 0.50),
    ("scatter", "dropout", 0.75),
    ("scatter", "delay", 0.01),
    ("scatter", "delay", 0.02),
    ("scatter", "delay", 0.05),
    ("scatter", "outliers", 0.05),
    ("scatter", "outliers", 0.10),
    ("probes", "none", 0.0),
    ("probes", "bias_one", 0.02),  # bias on one of the four probes
    ("probes", "bias_one", 0.05),
    ("probes", "delay_one", 0.02),
    ("probes", "dropout_one", 1.0),  # one probe lost entirely
]


def base_observations(layout, seed):
    import torch

    gen = torch.Generator().manual_seed(seed)
    if layout == "scatter":
        x = torch.rand(N_OBS, 1, generator=gen)
        t = torch.rand(N_OBS, 1, generator=gen)
        sensor = torch.zeros(N_OBS, dtype=torch.long)
    else:
        m = N_OBS // 4
        pos = torch.arange(1, 5, dtype=torch.float32) / 5
        times = (torch.arange(m, dtype=torch.float32) + 0.5) / m
        x = pos.repeat_interleave(m)[:, None]
        t = times.repeat(4)[:, None]
        sensor = torch.arange(4).repeat_interleave(m)
    u_true = torch.exp(-ALPHA_TRUE * math.pi**2 * t) * torch.sin(math.pi * x)
    u = u_true + NOISE * torch.randn(
        u_true.shape, generator=torch.Generator().manual_seed(10_000 + seed)
    )
    return {"x": x, "t": t, "u": u}, u_true, sensor


def run_cell(job):
    layout, kind, magnitude, seed, steps = job
    import torch

    from pinnrl.analysis import exact_sensitivities, linear_response, perturb_observations
    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    obs, u_true, sensor = base_observations(layout, seed)
    one = sensor == 1  # the probe at x = 0.4
    if kind == "none":
        disturbed = obs
    elif kind == "bias_one":
        disturbed = perturb_observations(obs, "bias", magnitude, sensors=one)
    elif kind == "delay_one":
        disturbed = perturb_observations(obs, "delay", magnitude, sensors=one)
    elif kind == "dropout_one":
        disturbed = {k: v[~one] for k, v in obs.items()}
    else:
        disturbed = perturb_observations(obs, kind, magnitude, seed=100 + seed)

    # Linear prediction of the shift from the disturbance of the values.
    pde0 = from_catalog("heat")
    if kind in ("dropout", "dropout_one"):
        predicted = float("nan")  # no delta_y: the information changes, not the values
    else:
        J = exact_sensitivities(pde0, ["alpha"], disturbed["x"], disturbed["t"])
        with torch.no_grad():
            u_at = torch.exp(-ALPHA_TRUE * math.pi**2 * obs["t"]) * torch.sin(math.pi * obs["x"])
        # delta_y: the disturbed values minus the truth at the (possibly delayed) attributed points,
        # minus the base noise, so that only the disturbance remains.
        base_delta = obs["u"] - u_true
        with torch.no_grad():
            u_attr = torch.exp(-ALPHA_TRUE * math.pi**2 * disturbed["t"]) * torch.sin(
                math.pi * disturbed["x"]
            )
        delta_y = (disturbed["u"] - u_attr) - base_delta
        predicted = float(linear_response(J, NOISE, delta_y)[0])
        del u_at

    pde = from_catalog(
        "heat",
        parameters={"alpha": ALPHA_TRUE},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_GUESS},
        observation_data=disturbed,
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
    out = trainer.train_steps(num_steps=steps, resample_every=50)
    alpha_hat = pde.get_trainable_parameter_values()["alpha"]
    losses = pde.compute_loss(trainer.model, *pde.generate_collocation_points(2000))
    return {
        "layout": layout,
        "kind": kind,
        "magnitude": magnitude,
        "seed": seed,
        "steps": steps,
        "n_obs": int(disturbed["u"].shape[0]),
        "alpha_hat": alpha_hat,
        "rel_error": abs(alpha_hat - ALPHA_TRUE) / ALPHA_TRUE,
        "predicted_shift": predicted,
        "residual_loss": float(losses["residual"].detach()),
        "data_loss": float(losses["data"].detach()),
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "final_loss": float(out["loss"][-1]),
        "train_time_s": time.time() - t0,
    }


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (lay, kind, mag, seed, args.steps) for seed in range(args.seeds) for lay, kind, mag in CELLS
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['layout']:<7} {rec['kind']:<11} {rec['magnitude']:<5} seed {rec['seed']}: "
                f"alpha {rec['alpha_hat']:.4f} rel_err {rec['rel_error']:.3f} predicted shift {rec['predicted_shift']:+.4f} "
                f"res {rec['residual_loss']:.1e} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
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
