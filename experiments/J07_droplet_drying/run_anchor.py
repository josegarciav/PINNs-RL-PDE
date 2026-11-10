"""J07e: the closure anchored at the equilibrium moisture, f(X_eq) = 0.

J07c traced the pointwise start's misfit (60 to 90 times the noise) to one
number: the bounded network cannot reach zero, so f(X_eq) is 0.01 and the
droplet keeps drying below equilibrium. Here the closure is a monotone network
with ``anchor={"at": X_eq, "value": 0}`` and no sigmoid: zero at equilibrium
by construction, non-negative above it. Two arms, same records as J07b:

* ``anchored_only``: the function pre-trained to the pointwise estimate;
* ``anchored``: that start refined by the PINN (J07b's ``pointwise`` schedule).

Each is graded by the closure's relative L2 error, by the record's misfit
through the reference solver (``DryingDroplet.record_misfit``) and, for the
PINN, by the network's own sensor misfit (``pde.sensor_misfit``).

Usage: python experiments/J07_droplet_drying/run_anchor.py [--seeds 3] [--steps 12000] [--workers 3]
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


def run_seed(job):
    seed, steps = job
    import sys

    import torch

    sys.path.insert(0, str(HERE))
    import run as J  # noqa: E402

    from pinnrl.pdes import PointSensor

    torch.set_num_threads(2)
    dr = J.plant()
    times, m_rec, T_rec, ref = J.records(seed)
    sigma_m, sigma_T = J.NOISE_M_REL * dr.m0, J.NOISE_T
    sensors = [
        PointSensor(
            0.5, times, expression="m_s*(1 + X)", noise=sigma_m, scale=dr.m0, name="balance"
        ),
        PointSensor(0.5, times, expression="T", noise=sigma_T, name="thermocouple"),
    ]
    spec = {"hidden": [16, 16], "bounds": None, "anchor": {"at": dr.X_eq, "value": 0.0}}
    pde = dr.custom_pde(J.T_MAX, unknown_rate=spec, sensors=sensors)
    pde.sensors[0].set_values(m_rec)
    pde.sensors[1].set_values(T_rec)
    f = pde.unknown_functions["f"]

    def f_hat(X):
        with torch.no_grad():
            return f({"X": torch.tensor(X, dtype=torch.float32).reshape(-1, 1)}).numpy().ravel()

    def grade(arm, t0, extra=None):
        err, grid, values = J.closure_error(dr, f_hat)
        g, v = np.array(grid), np.array(values)
        chi = dr.record_misfit(
            times,
            m_rec,
            T_rec,
            sigma_m,
            sigma_T,
            closure=lambda X: np.interp(X, g, v),
        )
        return {
            "arm": arm,
            "seed": seed,
            "closure_rel_l2": err,
            "X_grid": grid,
            "f_hat": values,
            "f_at_equilibrium": float(f_hat(np.array([dr.X_eq]))[0]),
            "record_misfit": chi["total"],
            "train_time_s": time.time() - t0,
            **(extra or {}),
        }

    t0 = time.time()
    X_pts, f_pts = dr.closure_from_record(times, m_rec, T_rec, window=21, order=3)
    torch.manual_seed(seed)
    f.fit({"X": X_pts}, f_pts, steps=3000, lr=1e-2)
    out = [grade("anchored_only", t0)]
    trainer, _ = J.trainer_for(pde, seed, steps, lbfgs_steps=500, data_weight=DATA_WEIGHT)
    own = pde.sensor_misfit(trainer.model)
    extra = {"sensor_chi2": own.chi2, "sensor_reduced": own.reduced, "sensor_limit": own.limit}
    extra.update(J.field_errors(trainer, ref))
    out.append(grade("anchored", t0, extra))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rows in pool.map(run_seed, [(s, args.steps) for s in range(args.seeds)]):
            for r in rows:
                runs.append(r)
                own = (
                    f" own reduced chi2 {r['sensor_reduced']:.2f}" if "sensor_reduced" in r else ""
                )
                print(
                    f"{r['arm']:<14} seed={r['seed']} closure rel L2 {r['closure_rel_l2']:.3f} "
                    f"record misfit {r['record_misfit']:.2f}{own} {r['train_time_s']:.0f}s",
                    flush=True,
                )
    (args.out / "anchor.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "anchor.json")


if __name__ == "__main__":
    main()
