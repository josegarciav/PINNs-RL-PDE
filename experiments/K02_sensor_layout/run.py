"""K02: sensor layout vs conditioning, with the sensor operators of pinnrl.pdes.sensors.

Part A (closed forms, seconds): the Fisher information of a layout from
``sensor_jacobian`` on the exact solution.

* heat, alpha: probes at n in {1, 2, 4, 8} evenly spaced positions, 50 readings
  each, noise 0.05; one probe swept over x0; and three sensor kinds at equal
  record count (a probe, the left wall flux -alpha*u_x, the domain mean).
* KdV (alpha, beta): probes at n in {1, 2, 4, 8} positions on [-15, 15], 50
  readings each on t in [0, 2]; one probe swept over x0 relative to the soliton
  path x = t; condition number, correlation and the per-parameter bound.

Part B (training): heat alpha identified from 1, 2, 4 and 8 probes at noise
0.05, 5 seeds, against the layout's Cramér-Rao bound.

Usage: python experiments/K02_sensor_layout/run.py [--seeds 5] [--steps 5000] [--workers 5]
       [--fisher-only] [--out experiments/K02_sensor_layout/results]
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
COUNTS = [1, 2, 4, 8]
READINGS = 50
NOISE = 0.05
ALPHA_TRUE, ALPHA_GUESS = 0.1, 0.5
TRAJECTORY_EVERY = 50


def even_positions(n: int, lo: float, hi: float) -> list[float]:
    """``n`` positions at the centres of ``n`` equal cells of ``[lo, hi]``."""
    h = (hi - lo) / n
    return [lo + h * (k + 0.5) for k in range(n)]


def heat_probes(n: int, noise: float = NOISE):
    import numpy as np

    from pinnrl.pdes import PointSensor

    times = np.linspace(0.0, 1.0, READINGS)
    return [
        PointSensor(x0, times, noise=noise, name=f"probe{k}")
        for k, x0 in enumerate(even_positions(n, 0, 1))
    ]


def fisher_part() -> dict:
    """Part A: Cramér-Rao bounds and condition numbers of layouts from the closed forms."""
    import numpy as np

    from pinnrl.analysis import fisher_information
    from pinnrl.pdes import BoundarySensor, IntegralSensor, PointSensor, sensor_jacobian
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.pdes.sensors import sensor_noise_vector

    def bound(pde, params):
        J = sensor_jacobian(pde, params)
        return fisher_information(J, sensor_noise_vector(pde.sensors), params)

    out: dict = {"heat": {}, "kdv": {}}
    times = np.linspace(0.0, 1.0, READINGS)

    # heat: probe count
    rows = []
    for n in COUNTS:
        pde = from_catalog("heat", sensors=heat_probes(n))
        s = bound(pde, ["alpha"])
        rows.append(
            {"n_sensors": n, "records": n * READINGS, "cr_rel_std": float(s.std[0]) / ALPHA_TRUE}
        )
    out["heat"]["count"] = rows
    # heat: one probe swept
    rows = []
    for x0 in np.linspace(0.05, 0.95, 19):
        pde = from_catalog("heat", sensors=[PointSensor(float(x0), times, noise=NOISE)])
        rows.append(
            {"x0": float(x0), "cr_rel_std": float(bound(pde, ["alpha"]).std[0]) / ALPHA_TRUE}
        )
    out["heat"]["position"] = rows
    # heat: sensor kinds at equal record count
    kinds = {
        "probe at 0.5": [PointSensor(0.5, times, noise=NOISE)],
        "probe at 0.25": [PointSensor(0.25, times, noise=NOISE)],
        "left flux -alpha*u_x": [BoundarySensor("left", times, "-alpha*u_x", noise=NOISE)],
        "domain mean": [IntegralSensor(times, "u", normalize=True, noise=NOISE)],
        "probe 0.25 + left flux": [
            PointSensor(0.25, times[::2], noise=NOISE),
            BoundarySensor("left", times[1::2], "-alpha*u_x", noise=NOISE),
        ],
    }
    out["heat"]["kinds"] = [
        {
            "layout": name,
            "records": sum(s.n_records for s in sensors),
            "cr_rel_std": float(bound(from_catalog("heat", sensors=sensors), ["alpha"]).std[0])
            / ALPHA_TRUE,
        }
        for name, sensors in kinds.items()
    ]

    # KdV two-parameter: count and position
    kdv_times = np.linspace(0.0, 2.0, READINGS)
    params = ["alpha", "beta"]
    rows = []
    for n in COUNTS:
        sensors = [PointSensor(x0, kdv_times, noise=NOISE) for x0 in even_positions(n, -15, 15)]
        s = bound(from_catalog("kdv_two_parameter", sensors=sensors), params)
        rows.append(
            {
                "n_sensors": n,
                "condition_number": float(s.condition_number),
                "correlation": float(s.correlation[0, 1]),
                "cr_rel_std_alpha": float(s.std[0]) / 6.0,
                "cr_rel_std_beta": float(s.std[1]) / 1.0,
            }
        )
    out["kdv"]["count"] = rows
    rows = []
    for x0 in np.linspace(-6, 8, 29):
        s = bound(
            from_catalog(
                "kdv_two_parameter", sensors=[PointSensor(float(x0), kdv_times, noise=NOISE)]
            ),
            params,
        )
        rows.append(
            {
                "x0": float(x0),
                "condition_number": float(s.condition_number),
                "correlation": float(s.correlation[0, 1]),
                "cr_rel_std_alpha": float(s.std[0]) / 6.0,
                "cr_rel_std_beta": float(s.std[1]) / 1.0,
            }
        )
    out["kdv"]["position"] = rows
    # KdV: probes placed on the soliton path vs off it
    on_path = [PointSensor(x0, kdv_times, noise=NOISE) for x0 in (0.0, 0.5, 1.0, 1.5, 2.0)]
    off_path = [PointSensor(x0, kdv_times, noise=NOISE) for x0 in (-12.0, -9.0, 6.0, 9.0, 12.0)]
    for name, sensors in (("on the soliton path", on_path), ("far from the path", off_path)):
        s = bound(from_catalog("kdv_two_parameter", sensors=sensors), params)
        out["kdv"][name] = {
            "condition_number": float(s.condition_number),
            "cr_rel_std_alpha": float(s.std[0]) / 6.0,
            "cr_rel_std_beta": float(s.std[1]) / 1.0,
            "identifiable": [bool(b) for b in s.identifiable],
        }
    return out


def run_cell(job: tuple[int, int, int]) -> dict:
    """Part B: identify alpha from ``n`` probes; one seed."""
    n, seed, steps = job
    import torch

    from pinnrl.analysis import fisher_information
    from pinnrl.pdes import sensor_jacobian
    from pinnrl.pdes.catalog import from_catalog
    from pinnrl.pdes.sensors import sensor_noise_vector

    torch.set_num_threads(2)
    pde = from_catalog(
        "heat",
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": ALPHA_GUESS},
        sensors=heat_probes(n),
    )
    pde.observe(seed=seed)
    cr = float(
        fisher_information(
            sensor_jacobian(pde, ["alpha"]), sensor_noise_vector(pde.sensors), ["alpha"]
        ).std[0]
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
    trajectory, done = [], 0
    while done < steps:
        chunk = min(TRAJECTORY_EVERY, steps - done)
        out = trainer.train_steps(num_steps=chunk, resample_every=TRAJECTORY_EVERY)
        done += chunk
        trajectory.append(pde.get_trainable_parameter_values()["alpha"])
    alpha_hat = trajectory[-1]
    return {
        "n_sensors": n,
        "seed": seed,
        "steps": steps,
        "alpha_hat": alpha_hat,
        "rel_error": abs(alpha_hat - ALPHA_TRUE) / ALPHA_TRUE,
        "crlb_rel_std": cr / ALPHA_TRUE,
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
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--fisher-only", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    fisher = fisher_part()
    (args.out / "fisher.json").write_text(json.dumps(fisher, indent=1))
    print(f"fisher part done in {time.time() - t0:.1f}s")
    for row in fisher["heat"]["count"]:
        print(f"  heat {row['n_sensors']} probes: CR rel std {row['cr_rel_std']:.4f}")
    for row in fisher["kdv"]["count"]:
        print(
            f"  kdv {row['n_sensors']} probes: kappa {row['condition_number']:.1f} corr {row['correlation']:.3f}"
        )
    if args.fisher_only:
        return

    jobs = [(n, seed, args.steps) for seed in range(args.seeds) for n in COUNTS]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] n={rec['n_sensors']} seed={rec['seed']} alpha={rec['alpha_hat']:.4f} "
                f"rel_err={rec['rel_error']:.3f} cr={rec['crlb_rel_std']:.3f} {rec['train_time_s']:.0f}s "
                f"(elapsed {time.time() - t0:.0f}s)",
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
