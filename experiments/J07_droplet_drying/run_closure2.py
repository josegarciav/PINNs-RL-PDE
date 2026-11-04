"""J07b: making the free drying-rate closure train (second pass of J07's Part B).

Same plant, records and controls as ``run.py``. Four arms, 3 seeds:

* ``pointwise``: the closure is first estimated point by point from the
  smoothed record, f_i = -dX/dt m_s / (h_m A (rho_vs(T) - rho_va)) (the
  equation-discovery estimate), the UnknownFunction is pre-trained to it, and
  the PINN refines everything from there;
* ``two_stage``: the network first fits the record with f frozen at its
  start, then f is released;
* ``basis``: f as six Legendre modes of X (bounded to [0, 1]) instead of a
  monotone network, one-stage training;
* ``pointwise_only``: the pre-trained function alone, no PINN (the control
  the twin must beat).

Usage: python experiments/J07_droplet_drying/run_closure2.py [--seeds 3] [--steps 12000] [--workers 3]
"""

from __future__ import annotations

import argparse
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
DATA_WEIGHT = 300.0


def pointwise_estimate(dr, times, m_rec, T_rec):
    """``(X_i, f_i)`` from the smoothed records (``DryingDroplet.closure_from_record``)."""
    return dr.closure_from_record(times, m_rec, T_rec, window=21, order=3)


def pretrain(fn, X_pts, f_pts, steps=3000):
    """Pre-train the unknown function to the pointwise estimate (``UnknownFunction.fit``)."""
    return fn.fit({"X": X_pts}, f_pts, steps=steps, lr=1e-2)


def run_arm(job):
    arm, seed, steps = job
    import sys

    import torch

    sys.path.insert(0, str(HERE))
    import run as J  # noqa: E402

    from pinnrl.pdes import PointSensor

    torch.set_num_threads(2)
    dr = J.plant()
    times, m_rec, T_rec, ref = J.records(seed)
    sensors = [
        PointSensor(
            0.5,
            times,
            expression="m_s*(1 + X)",
            noise=J.NOISE_M_REL * dr.m0,
            scale=dr.m0,
            name="balance",
        ),
        PointSensor(0.5, times, expression="T", noise=J.NOISE_T, name="thermocouple"),
    ]
    spec = {"hidden": [16, 16], "initial": 0.5}
    if arm == "basis":
        spec = {"basis": {"type": "legendre", "modes": 6}, "monotone": None, "initial": 0.5}
    pde = dr.custom_pde(J.T_MAX, unknown_rate=spec, sensors=sensors)
    pde.sensors[0].set_values(m_rec)
    pde.sensors[1].set_values(T_rec)
    f = pde.unknown_functions["f"]
    out = {"arm": arm, "seed": seed, "steps": steps}
    t0 = time.time()
    if arm in ("pointwise", "pointwise_only"):
        X_pts, f_pts = pointwise_estimate(dr, times, m_rec, T_rec)
        torch.manual_seed(seed)
        out["pretrain_mse"] = pretrain(f, X_pts, f_pts)
        truth_pts = dr.drying_curve(X_pts)
        out["raw_estimate_rel_l2"] = float(
            np.linalg.norm(f_pts - truth_pts) / np.linalg.norm(truth_pts)
        )

        def f_hat0(X):
            with torch.no_grad():
                return f({"X": torch.tensor(X, dtype=torch.float32).reshape(-1, 1)}).numpy().ravel()

        out["pretrained_rel_l2"], grid, values = J.closure_error(dr, f_hat0)
        if arm == "pointwise_only":
            out.update(
                {
                    "closure_rel_l2": out["pretrained_rel_l2"],
                    "X_grid": grid,
                    "f_hat": values,
                    "train_time_s": time.time() - t0,
                }
            )
            return out
    if arm == "two_stage":
        for p in f.parameters():
            p.requires_grad_(False)
        trainer, _ = J.trainer_for(pde, seed, steps // 2, lbfgs_steps=0, data_weight=DATA_WEIGHT)
        for p in f.parameters():
            p.requires_grad_(True)
        trainer.train_steps(num_steps=steps - steps // 2, resample_every=50)
        trainer._switch_to_lbfgs()
        for g in trainer.optimizer.param_groups:
            g["lr"] = 1.0
        trainer.train_steps(num_steps=500, resample_every=50)
    else:
        trainer, _ = J.trainer_for(pde, seed, steps, lbfgs_steps=500, data_weight=DATA_WEIGHT)

    def f_hat(X):
        with torch.no_grad():
            return f({"X": torch.tensor(X, dtype=torch.float32).reshape(-1, 1)}).numpy().ravel()

    err, grid, values = J.closure_error(dr, f_hat)
    out.update(
        {
            "closure_rel_l2": err,
            "X_grid": grid,
            "f_hat": values,
            "train_time_s": time.time() - t0,
            "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
        }
    )
    out.update(J.field_errors(trainer, ref))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--arms", nargs="+", default=["pointwise_only", "pointwise", "two_stage", "basis"]
    )
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(arm, s, args.steps) for arm in args.arms for s in range(args.seeds)]
    runs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_arm, jobs):
            runs.append(r)
            extra = "".join(
                f" {k} {r[k]:.3f}"
                for k in ("raw_estimate_rel_l2", "pretrained_rel_l2", "rel_l2_X")
                if k in r
            )
            print(
                f"{r['arm']:<15} seed={r['seed']} closure rel L2 {r['closure_rel_l2']:.3f}{extra} {r['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "closure2.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "closure2.json")


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(HERE))
    main()
