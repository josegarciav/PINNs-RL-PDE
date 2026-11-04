"""F09: recovery of a time-varying diffusivity alpha(t) and a coefficient field alpha(x).

Four truths on the 1-D heat equation with u0 = sin(pi x) and Dirichlet 0:

* ``linear``: alpha(t) = 0.05 + 0.1 t (closed-form solution);
* ``step``:   alpha(t) = 0.05 + 0.1 H(t - 0.5), a fouling-like jump (closed form);
* ``bump``:   alpha(x) = 0.05 + 0.1 exp(-((x - 0.5)/0.15)^2), a centred bump
  (method-of-lines reference);
* ``layers``: alpha(x) = 0.02 + 0.18 sigmoid((x - 0.5)/0.05), a two-layer medium
  with a smooth interface (method-of-lines reference).

Each truth is identified with an ``UnknownFunction`` (a 2 x 32 tanh network of
``t`` or ``x`` through a softplus, started at the constant 0.1) for every
(noise, seed) cell, and with a scalar alpha at noise 0.01 as the control that
shows what a constant absorbs. Records the relative L2 error of the recovered
function on a grid, its trajectory, the field error, and the best constant
alpha for each truth (the constant whose solution is closest to the truth's).

Usage: python experiments/F09_field_unknowns/run.py [--seeds 5] [--steps 10000]
       [--workers 5] [--out experiments/F09_field_unknowns/results]
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
CONTROL_NOISE = 0.01
N_OBS = 200
ALPHA_START = 0.1
TRAJECTORY_EVERY = 500
GRID = 101

ARMS = {
    "linear": dict(
        variable="t",
        truth="0.05 + 0.1*t",
        exact="exp(-pi^2*(0.05*t + 0.05*t^2))*sin(pi*x)",
    ),
    "step": dict(
        variable="t",
        truth="0.05 + 0.1*heaviside(t - 0.5)",
        exact="exp(-pi^2*(0.05*t + 0.1*max(t - 0.5, 0)))*sin(pi*x)",
    ),
    "bump": dict(variable="x", truth="0.05 + 0.1*exp(-((x - 0.5)/0.15)^2)"),
    "layers": dict(variable="x", truth="0.02 + 0.18*sigmoid((x - 0.5)/0.05)"),
}


def _truth_numpy(arm: str):
    import numpy as np

    return {
        "linear": lambda x, t: 0.05 + 0.1 * t * np.ones_like(x),
        "step": lambda x, t: (0.05 + 0.1 * float(t > 0.5)) * np.ones_like(x),
        "bump": lambda x, t: 0.05 + 0.1 * np.exp(-(((x - 0.5) / 0.15) ** 2)),
        "layers": lambda x, t: 0.02 + 0.18 / (1.0 + np.exp(-(x - 0.5) / 0.05)),
    }[arm]


def build_pde(arm: str, model: str):
    """The heat problem of ``arm`` with alpha as a function (``model="function"``) or a scalar."""
    from pinnrl import CustomPDE
    from pinnrl.benchmarks import heat_variable_reference

    spec = ARMS[arm]
    if "exact" in spec:
        exact = spec["exact"]
    else:
        exact = heat_variable_reference(_truth_numpy(arm), nx=201, nt=201).as_exact_solution()
    common = dict(
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=exact,
        name=f"heat alpha({spec['variable']}) {arm}",
    )
    if model == "function":
        equation = (
            "u_t - alpha*u_xx" if spec["variable"] == "t" else "u_t - alpha_x*u_x - alpha*u_xx"
        )
        return CustomPDE(
            equation,
            unknown_functions={
                "alpha": {
                    "inputs": [spec["variable"]],
                    "hidden": [32, 32],
                    "positive": True,
                    "initial": ALPHA_START,
                    "truth": spec["truth"],
                }
            },
            **common,
        )
    return CustomPDE(
        "u_t - alpha*u_xx",
        parameters={"alpha": ALPHA_START},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_START},
        **common,
    )


def run_cell(job: tuple[str, str, float, int, int]) -> dict:
    """Train one inverse problem and return its record."""
    arm, model, noise, seed, steps = job
    import numpy as np
    import torch

    torch.set_num_threads(2)
    pde = build_pde(arm, model)
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
    trainer.model.apply(_reset)

    variable = ARMS[arm]["variable"]
    grid = torch.linspace(0.0, 1.0, GRID).reshape(-1, 1)
    truth_fn = _truth_numpy(arm)
    truth = np.array([truth_fn(np.array([g]), g)[0] for g in grid.ravel().numpy()])

    def profile() -> np.ndarray:
        with torch.no_grad():
            if model == "function":
                fn = pde.unknown_functions["alpha"]
                return fn({variable: grid}).ravel().numpy()
            return np.full(GRID, pde.get_trainable_parameter_values()["alpha"])

    def rel_error(values: np.ndarray) -> float:
        return float(np.linalg.norm(values - truth) / np.linalg.norm(truth))

    t0 = time.time()
    trajectory = []
    done = 0
    while done < steps:
        chunk = min(TRAJECTORY_EVERY, steps - done)
        out = trainer.train_steps(num_steps=chunk, resample_every=50)
        done += chunk
        trajectory.append(rel_error(profile()))
    final = profile()
    return {
        "arm": arm,
        "model": model,
        "variable": variable,
        "noise": noise,
        "seed": seed,
        "steps": steps,
        "alpha_rel_l2": rel_error(final),
        "alpha_max_err": float(np.max(np.abs(final - truth)) / np.max(truth)),
        "alpha_hat": float(final.mean()) if model == "constant" else None,
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "final_loss": float(out["loss"][-1]),
        "train_time_s": time.time() - t0,
        "trajectory": trajectory,
        "profile": final.tolist(),
        "truth": truth.tolist(),
    }


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def best_constants() -> dict:
    """For each truth, the constant alpha whose solution is closest to the truth's, and that distance."""
    import numpy as np

    from pinnrl.benchmarks import heat_variable_reference

    out = {}
    for arm in ARMS:
        ref = heat_variable_reference(_truth_numpy(arm), nx=101, nt=101, refine=2)
        best = (np.inf, None)
        for a in np.linspace(0.02, 0.2, 91):
            cand = heat_variable_reference(
                lambda x, t, a=a: a * np.ones_like(x), nx=101, nt=101, refine=2
            )
            dist = ref.relative_l2(cand.u)
            if dist < best[0]:
                best = (dist, float(a))
        grid = np.linspace(0, 1, GRID)
        truth = np.array([_truth_numpy(arm)(np.array([g]), g)[0] for g in grid])
        out[arm] = {
            "alpha": best[1],
            "u_distance": best[0],
            "alpha_rel_l2": float(np.linalg.norm(best[1] - truth) / np.linalg.norm(truth)),
        }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "best_constant.json").write_text(json.dumps(best_constants(), indent=1))
    jobs = [
        (arm, "function", noise, seed, args.steps)
        for seed in range(args.seeds)
        for arm in ARMS
        for noise in NOISES
    ] + [
        (arm, "constant", CONTROL_NOISE, seed, args.steps)
        for seed in range(args.seeds)
        for arm in ARMS
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['arm']:<7} {rec['model']:<8} noise={rec['noise']:<5} "
                f"seed={rec['seed']} alpha_err={rec['alpha_rel_l2']:.3f} "
                f"l2={rec['l2_error']:.3f} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    fields = [k for k in records[0] if k not in ("trajectory", "profile", "truth")]
    with open(args.out / "runs.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec[k] for k in fields})
    print("saved", args.out / "runs.csv")


if __name__ == "__main__":
    main()
