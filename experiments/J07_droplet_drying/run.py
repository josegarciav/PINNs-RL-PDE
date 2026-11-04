"""J07: a drying droplet, its PINN forward solve, and the drying-rate closure identified from a record.

Part A, forward: the single-droplet model of ``pinnrl.process.DryingDroplet``
(1.5 mm, 20 % solids, air at 90 C with a +-5 K wander, 200 s) solved by a
PINN with the clean recipe (normalized, 4 x 64, Adam 12,000 steps then L-BFGS 500; 4,000 + 100 left a 20 % error), 3 seeds,
against the ODE reference.

Part B, the closure: the plant's drying curve is the characteristic curve
f(X) = min(1, ((X - X_eq)/(X_cr - X_eq))^n) with n = 1.5, X_cr = 1. The twin
sees the droplet mass every second (a balance, noise 0.5 % of the initial
mass) and its temperature every second (noise 0.5 K), and identifies f as an
UnknownFunction of X (bounded to [0, 1], increasing), 3 seeds. Controls on the
same records, by least squares on the ODE reference (scipy): the parametric
curve with the right form (n, X_cr free: the best a classical estimator can
do), and a misspecified linear form f = min(1, X / X_lin) (what a plant model
would assume). Metrics: the relative L2 error of f over [X_eq, X_0], the
field errors, the run time.

Usage: python experiments/J07_droplet_drying/run.py [--seeds 3] [--steps 12000] [--lbfgs-steps 500] [--initial 0.5] [--workers 3]
"""

from __future__ import annotations

import argparse
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
T_MAX = 200.0
EVERY = 1.0
NOISE_M_REL, NOISE_T = 0.005, 0.5


def plant():
    from pinnrl.process import DryingDroplet

    return DryingDroplet(T_air=lambda t: 363.15 + 5.0 * math.sin(2 * math.pi * t / 80.0))


def records(seed):
    dr = plant()
    ref = dr.reference(T_MAX, nt=2001)
    times = np.arange(EVERY, T_MAX + 1e-9, EVERY)
    m = np.interp(times, ref.t, ref.m)
    T = np.interp(times, ref.t, ref.T)
    rm, rt = np.random.default_rng(seed), np.random.default_rng(seed + 1000)
    return (
        times,
        m + NOISE_M_REL * dr.m0 * rm.standard_normal(times.size),
        T + NOISE_T * rt.standard_normal(times.size),
        ref,
    )


def closure_error(dr, f_hat):
    """Relative L2 of a closure against the plant's over [X_eq, X_0]."""
    X = np.linspace(dr.X_eq, dr.X0, 400)
    truth = dr.drying_curve(X)
    return (
        float(np.linalg.norm(f_hat(X) - truth) / np.linalg.norm(truth)),
        X.tolist(),
        f_hat(X).tolist(),
    )


def trainer_for(pde, seed, steps, lbfgs_steps=100, data_weight=10.0):
    import torch

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
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": data_weight},
    )
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=lbfgs_steps, resample_every=50)
    return trainer, time.time() - t0


def field_errors(trainer, ref):
    import torch

    xt = torch.tensor(np.stack([np.full(ref.t.size, 0.5), ref.t], 1), dtype=torch.float32)
    with torch.no_grad():
        pred = trainer.model(xt).numpy()
    return {
        "rel_l2_X": float(np.linalg.norm(pred[:, 0] - ref.X) / np.linalg.norm(ref.X)),
        "rel_l2_T_minus_T0": float(
            np.linalg.norm(pred[:, 1] - ref.T) / np.linalg.norm(ref.T - ref.T[0])
        ),
        "max_abs_T": float(np.abs(pred[:, 1] - ref.T).max()),
    }


def run_forward(job):
    seed, steps, lbfgs_steps, _initial, _data_weight = job
    import torch

    torch.set_num_threads(2)
    dr = plant()
    ref = dr.reference(T_MAX, nt=401)
    pde = dr.custom_pde(T_MAX)
    trainer, train_time = trainer_for(pde, seed, steps)
    out = {
        "part": "forward",
        "seed": seed,
        "train_time_s": train_time,
        "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
    }
    out.update(field_errors(trainer, ref))
    return out


def run_closure(job):
    seed, steps, lbfgs_steps, initial, data_weight = job
    import torch

    from pinnrl.pdes import PointSensor

    torch.set_num_threads(2)
    dr = plant()
    times, m_rec, T_rec, ref = records(seed)
    sensors = [
        PointSensor(
            0.5,
            times,
            expression="m_s*(1 + X)",
            noise=NOISE_M_REL * dr.m0,
            scale=dr.m0,
            name="balance",
        ),
        PointSensor(0.5, times, expression="T", noise=NOISE_T, name="thermocouple"),
    ]
    pde = dr.custom_pde(
        T_MAX, unknown_rate={"hidden": [16, 16], "initial": initial}, sensors=sensors
    )
    pde.sensors[0].set_values(m_rec)
    pde.sensors[1].set_values(T_rec)
    trainer, train_time = trainer_for(pde, seed, steps, lbfgs_steps, data_weight)
    f = pde.unknown_functions["f"]

    def f_hat(X):
        with torch.no_grad():
            return f({"X": torch.tensor(X, dtype=torch.float32).reshape(-1, 1)}).numpy().ravel()

    err, grid, values = closure_error(dr, f_hat)
    out = {
        "part": "closure",
        "seed": seed,
        "train_time_s": train_time,
        "closure_rel_l2": err,
        "X_grid": grid,
        "f_hat": values,
        "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
    }
    out.update(field_errors(trainer, dr.reference(T_MAX, nt=401)))
    return out


def run_least_squares(seed, form):
    """Least squares of a parametric closure on the ODE reference: the classical control."""
    from scipy.optimize import least_squares

    from pinnrl.process import DryingDroplet

    dr = plant()
    times, m_rec, T_rec, _ = records(seed)
    sigma_m, sigma_T = NOISE_M_REL * dr.m0, NOISE_T

    def model(theta):
        if form == "cdc":
            n, X_cr = theta
            cand = DryingDroplet(T_air=dr.T_air, n=n, X_cr=X_cr)
        else:  # linear: f = min(1, X / X_lin)
            (X_lin,) = theta
            cand = DryingDroplet(
                T_air=dr.T_air,
                rate_closure=lambda X, X_lin=X_lin: np.minimum(1.0, np.maximum(X, 0.0) / X_lin),
            )
        ref = cand.reference(T_MAX, nt=401)
        return cand, np.concatenate(
            [
                (np.interp(times, ref.t, ref.m) - m_rec) / sigma_m,
                (np.interp(times, ref.t, ref.T) - T_rec) / sigma_T,
            ]
        )

    x0 = [1.0, 0.5] if form == "cdc" else [0.5]
    bounds = ([0.2, dr.X_eq + 0.05], [5.0, dr.X0]) if form == "cdc" else ([0.05], [dr.X0])
    t0 = time.time()
    res = least_squares(lambda th: model(th)[1], x0, bounds=bounds, xtol=1e-8, ftol=1e-10)
    cand = model(res.x)[0]
    err, grid, values = closure_error(dr, cand.drying_curve)
    return {
        "part": f"lsq_{form}",
        "seed": seed,
        "theta": res.x.tolist(),
        "closure_rel_l2": err,
        "misfit": float(np.sum(res.fun**2)),
        "time_s": time.time() - t0,
        "X_grid": grid,
        "f_hat": values,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--lbfgs-steps", type=int, default=500)
    parser.add_argument("--initial", type=float, default=0.5, help="start of the unknown closure")
    parser.add_argument(
        "--data-weight", type=float, default=10.0, help="weight of the sensor terms"
    )
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    runs = []
    for seed in range(args.seeds):
        for form in ("cdc", "linear"):
            r = run_least_squares(seed, form)
            runs.append(r)
            print(
                f"lsq {form:<6} seed={seed} theta={np.round(r['theta'], 3)} closure rel L2 {r['closure_rel_l2']:.3f} misfit {r['misfit']:.0f} {r['time_s']:.1f}s",
                flush=True,
            )
    jobs = [("forward", s) for s in range(args.seeds)] + [("closure", s) for s in range(args.seeds)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(
                run_forward if kind == "forward" else run_closure,
                (s, args.steps, args.lbfgs_steps, args.initial, args.data_weight),
            )
            for kind, s in jobs
        ]
        for fut in futures:
            r = fut.result()
            runs.append(r)
            extra = f" closure rel L2 {r['closure_rel_l2']:.3f}" if "closure_rel_l2" in r else ""
            print(
                f"pinn {r['part']:<8} seed={r['seed']} X {r['rel_l2_X']:.4f} T {r['rel_l2_T_minus_T0']:.4f} max|dT| {r['max_abs_T']:.2f} K{extra} {r['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
