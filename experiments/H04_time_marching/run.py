"""H04: time-marching windows against training on the whole window at once.

The K09 failures (pendulum, SIR, Lorenz: the PINN returns a trivial solution),
Van der Pol as the control, and the J07 drying droplet (an 8-14 % forward
error at the critical-moisture transition), each trained with the clean recipe
(normalized to the reference's scales, 4 x 64, Adam then L-BFGS 500) on 1, 2,
4 and 8 windows of equal length with the same total budget of Adam steps
(``PDETrainer.train_windows``: collocation points beyond the current cutoff
are left out and the network carries over). 2 seeds. Metric: the relative L2
error against the reference over the whole window, per field.

Usage: python experiments/H04_time_marching/run.py [--seeds 2] [--steps 12000] [--workers 3]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
EXPERIMENTS = HERE.parent
SYSTEMS = ["pendulum", "sir", "lorenz", "van_der_pol", "droplet"]
WINDOWS = (1, 2, 4, 8)
LBFGS_STEPS = 500


def load_driver(folder: str, alias: str):
    """Import ``experiments/<folder>/run.py`` under a distinct module name."""
    spec = importlib.util.spec_from_file_location(alias, EXPERIMENTS / folder / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build(system, seed):
    """``(pde, evaluate)``: the problem and a function of the trainer returning per-field errors."""
    import torch

    if system == "droplet":
        J = load_driver("J07_droplet_drying", "j07_run")

        dr = J.plant()
        ref = dr.reference(J.T_MAX, nt=401)
        pde = dr.custom_pde(J.T_MAX)

        def evaluate(trainer):
            e = J.field_errors(trainer, ref)
            return [e["rel_l2_X"], e["rel_l2_T_minus_T0"]]

        return pde, evaluate
    from scipy.integrate import solve_ivp

    from pinnrl.pdes.catalog import from_catalog

    K = load_driver("K09_linearized_stability", "k09_run")

    names, f_fn, _j, s0, (t0, t1), pde0 = K.state_space(system)
    t_eval = np.linspace(t0, t1, 401)
    sol = solve_ivp(
        lambda t, s: np.asarray(f_fn(*s), dtype=float).ravel(),
        (t0, t1),
        s0,
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    fields = list(pde0.fields)
    ref = sol.y[[names.index(f) for f in fields]]
    scales = {
        f: (float(ref[i].mean()), float(max(ref[i].std(), 1e-2 * abs(ref[i]).max(), 1e-3)))
        for i, f in enumerate(fields)
    }
    pde = from_catalog(system, scales=scales)

    def evaluate(trainer):
        xt = torch.tensor(np.stack([np.full(t_eval.size, 0.5), t_eval], 1), dtype=torch.float32)
        with torch.no_grad():
            pred = trainer.model(xt).numpy().T
        return [
            float(np.linalg.norm(pred[i] - ref[i]) / np.linalg.norm(ref[i]))
            for i in range(len(fields))
        ]

    return pde, evaluate


def run_cell(job):
    system, n_windows, seed, steps = job
    import torch

    torch.set_num_threads(2)
    pde, evaluate = build(system, seed)
    torch.manual_seed(seed)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        normalize=True,
        num_collocation_points=2048,
        batch_size=2048,
        num_boundary_points=64,
        num_initial_points=256,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    t0, t1 = float(pde.time_domain[0]), float(pde.time_domain[1])
    cutoffs = [t0 + (t1 - t0) * k / n_windows for k in range(1, n_windows + 1)]
    tic = time.time()
    result = trainer.train_windows(
        cutoffs, steps_per_window=steps // n_windows, resample_every=50, lbfgs_steps=LBFGS_STEPS
    )
    errors = evaluate(trainer)
    return {
        "system": system,
        "windows": n_windows,
        "seed": seed,
        "steps": steps,
        "rel_l2": errors,
        "rel_l2_mean": float(np.mean(errors)),
        "final_loss": result["loss"][-1],
        "train_time_s": time.time() - tic,
        "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--systems", nargs="+", default=SYSTEMS)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [
        (s, w, seed, args.steps)
        for s in args.systems
        for w in WINDOWS
        for seed in range(args.seeds)
    ]
    runs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_cell, jobs):
            runs.append(r)
            print(
                f"{r['system']:<12} windows={r['windows']} seed={r['seed']} rel L2 {np.round(r['rel_l2'], 4)} "
                f"loss {r['final_loss']:.2e} {r['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
