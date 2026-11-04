"""F11: can the time of a jump in a coefficient be identified as a parameter?

F09 showed that a smooth UnknownFunction turns a step in alpha(t) into a ramp.
Here the step is parametrised: alpha(t) = a + b * sigmoid((t - t0) / w) with
(a, b, t0) trainable and the width w fixed (0.02, a smoothed step), on the
heat equation with u0 = sin(pi x), Dirichlet 0, truth a = 0.05, b = 0.1,
t0 = 0.5 (F09's "step" truth, closed form). Started at a = 0.1, b = 0.05,
t0 = 0.3 (and t0 = 0.7 in a second arm), 200 observations, noise in
{0, 0.01, 0.05}, 5 seeds, 10,000 Adam steps; against the F09 network and the
Cramér-Rao bound on (a, b, t0) from the closed form.

Usage: python experiments/F11_change_point/run.py [--seeds 5] [--steps 10000] [--workers 5]
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
NOISES = [0.0, 0.01, 0.05]
N_OBS = 200
WIDTH = 0.02
TRUTH = {"a": 0.05, "b": 0.1, "t0": 0.5}
STARTS = {
    "from_0.3": {"a": 0.1, "b": 0.05, "t0": 0.3},
    "from_0.7": {"a": 0.1, "b": 0.05, "t0": 0.7},
}
EXACT = "exp(-pi^2*(0.05*t + 0.1*max(t - 0.5, 0)))*sin(pi*x)"


def build(model: str, start: dict):
    from pinnrl import CustomPDE

    common = dict(
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=EXACT,
    )
    if model == "network":
        return CustomPDE(
            "u_t - alpha*u_xx",
            unknown_functions={
                "alpha": {
                    "inputs": ["t"],
                    "hidden": [32, 32],
                    "positive": True,
                    "initial": 0.1,
                    "truth": "0.05 + 0.1*heaviside(t - 0.5)",
                }
            },
            name="heat step alpha(t), network",
            **common,
        )
    return CustomPDE(
        f"u_t - (a + b*sigmoid((t - t0)/{WIDTH}))*u_xx",
        parameters=dict(TRUTH),
        trainable_parameters=["a", "b", "t0"],
        parameter_initial_guesses=dict(start),
        name="heat step alpha(t), change point",
        **common,
    )


def run_cell(job):
    model, start_name, noise, seed, steps = job
    import numpy as np
    import torch

    torch.set_num_threads(2)
    start = STARTS[start_name]
    pde = build(model, start)
    pde.generate_synthetic_observations(n_points=N_OBS, noise_std=noise, seed=seed)
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
    trainer.model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    t0 = time.time()
    trajectory = []
    done = 0
    while done < steps:
        trainer.train_steps(num_steps=500, resample_every=50)
        done += 500
        trajectory.append(pde.get_trainable_parameter_values() if model == "change_point" else {})
    grid = torch.linspace(0, 1, 101).reshape(-1, 1)
    truth = 0.05 + 0.1 * (grid.ravel().numpy() > 0.5)
    with torch.no_grad():
        if model == "network":
            profile = pde.unknown_functions["alpha"]({"t": grid}).ravel().numpy()
        else:
            p = pde.get_trainable_parameter_values()
            profile = p["a"] + p["b"] / (1 + np.exp(-(grid.ravel().numpy() - p["t0"]) / WIDTH))
    rec = {
        "model": model,
        "start": start_name,
        "noise": noise,
        "seed": seed,
        "steps": steps,
        "alpha_rel_l2": float(np.linalg.norm(profile - truth) / np.linalg.norm(truth)),
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "train_time_s": time.time() - t0,
        "profile": profile.tolist(),
    }
    if model == "change_point":
        rec.update({k: v for k, v in pde.get_trainable_parameter_values().items()})
        rec["t0_error"] = abs(rec["t0"] - TRUTH["t0"])
        rec["trajectory_t0"] = [tr["t0"] for tr in trajectory]
    return rec


def crlb(seed: int, noise: float) -> dict:
    """Cramér-Rao bound on (a, b, t0) for the smoothed step, from the closed form."""
    import torch

    from pinnrl.analysis import closed_form_bound
    from pinnrl.pdes.catalog import from_catalog

    if noise <= 0:
        return {"a": 0.0, "b": 0.0, "t0": 0.0}
    # the closed form with the smoothed step: the decay integral of the sigmoid
    integral = f"a*t + b*{WIDTH}*log((1 + exp((t - t0)/{WIDTH}))/(1 + exp(-t0/{WIDTH})))"
    pde = from_catalog(
        "heat",
        equations=f"u_t - (a + b*sigmoid((t - t0)/{WIDTH}))*u_xx",
        parameters=dict(TRUTH),
        exact_solution=f"exp(-pi^2*({integral}))*sin(pi*x)",
    )
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(N_OBS, 1, generator=gen)
    t = torch.rand(N_OBS, 1, generator=gen)
    s = closed_form_bound(pde, ["a", "b", "t0"], x, t, noise)
    return {k: float(v) for k, v in zip(["a", "b", "t0"], s.std, strict=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    bounds = {
        f"{seed}_{noise}": crlb(seed, noise) for seed in range(args.seeds) for noise in NOISES
    }
    (args.out / "crlb.json").write_text(json.dumps(bounds, indent=1))
    jobs = [
        ("change_point", start, noise, seed, args.steps)
        for seed in range(args.seeds)
        for noise in NOISES
        for start in STARTS
    ]
    jobs += [
        ("network", "from_0.3", noise, seed, args.steps)
        for seed in range(args.seeds)
        for noise in NOISES
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            extra = (
                f"t0={rec['t0']:.3f} a={rec['a']:.3f} b={rec['b']:.3f}"
                if rec["model"] == "change_point"
                else ""
            )
            print(
                f"[{i}/{len(jobs)}] {rec['model']:<12} {rec['start']:<8} noise={rec['noise']:<5} seed={rec['seed']} "
                f"alpha_err={rec['alpha_rel_l2']:.3f} {extra} l2={rec['l2_error']:.3f} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    fields = [k for k in records[0] if k not in ("profile", "trajectory_t0")]
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=sorted(
                {k for r in records for k in r if k not in ("profile", "trajectory_t0")}
            ),
        )
        writer.writeheader()
        for rec in records:
            writer.writerow({k: v for k, v in rec.items() if k not in ("profile", "trajectory_t0")})
    print("saved", args.out / "runs.csv", len(fields))


if __name__ == "__main__":
    main()
