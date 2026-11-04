"""J03: recover a time-varying fouling resistance R_f(t) from synthetic outlet sensors.

The falling-film evaporator of J01 over a 20-minute window with the fouling
drifting R_f(t) = 2e-4 t / T (an 8-hour drift compressed), the steam
temperature wandering by +-1 K (a slow sine, measured) and the feed steady.
The twin sees the steam and feed records as inputs and two outlet sensors,
X(L, t) every 5 s with noise 0.002 and W(L, t) every 5 s with noise 0.001.
R_f is an UnknownFunction of t bounded to [0, 5e-4] (sigmoid output, so the
pre-activation is of order one), started at 1e-4; the control is a scalar R_f.
Training: the J02 recipe (normalized, Adam 4,000 steps then a safeguarded
L-BFGS finish of 100 steps, no network regularization), 3 seeds. Metrics: the
relative L2 error of R_f(t) on the window, its error at the end of the window,
and the field errors against the reference.

Usage: python experiments/J03_fouling_estimation/run.py [--seeds 3] [--steps 4000] [--workers 3]
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
T_MAX = 1200.0
R_END = 2e-4
SENSOR_EVERY = 5.0
NOISE_X, NOISE_W = 0.002, 0.001


def plant():
    from pinnrl.process import FallingFilmEvaporator

    return FallingFilmEvaporator(
        T_s=lambda t: 355.15 + 1.0 * math.sin(2 * math.pi * t / 400.0),
        R_f=lambda t: R_END * t / T_MAX,
    )


def truth_rf(t):
    return R_END * np.asarray(t) / T_MAX


def run_cell(job):
    model, seed, steps = job
    import torch

    from pinnrl.pdes import PointSensor

    torch.set_num_threads(2)
    ev = plant()
    ref = ev.reference(t_max=T_MAX, nz=201, nt=241)
    times = np.arange(SENSOR_EVERY, T_MAX + 1e-9, SENSOR_EVERY)
    sensors = [
        PointSensor(ev.length, times, expression="X", noise=NOISE_X, name="outlet X"),
        PointSensor(ev.length, times, expression="W", noise=NOISE_W, name="outlet W"),
    ]
    if model == "function":
        pde = ev.custom_pde(
            t_max=T_MAX,
            unknown_fouling={
                "hidden": [16, 16],
                "bounds": [0.0, 5e-4],
                "positive": False,
                "initial": 1e-4,
                "truth": f"{R_END}*t/{T_MAX}",
            },
            sensors=sensors,
        )
    else:  # a scalar fouling resistance, the control
        pde = ev.custom_pde(t_max=T_MAX, sensors=sensors)
        pde_dict = pde.to_dict()
        from pinnrl import CustomPDE

        pde_dict["inputs"].pop("R_f")
        pde_dict["parameters"]["R_f"] = R_END / 2  # the true mean, used only to fill observations
        pde_dict["trainable_parameters"] = ["R_f"]
        pde_dict["parameter_initial_guesses"] = {"R_f": 1e-4}
        pde_dict["initial_condition"] = None
        pde = CustomPDE.from_dict(pde_dict)
        ss = ev.steady_state(201, t=0.0)
        zg = ss["z"]

        def initial(x):
            zz = x[:, 0].detach().cpu().numpy()
            return torch.stack(
                [
                    torch.as_tensor(np.interp(zz, zg, ss["W"]), dtype=x.dtype),
                    torch.as_tensor(np.interp(zz, zg, ss["X"]), dtype=x.dtype),
                ],
                dim=1,
            )

        pde.initial_condition = lambda x, t: initial(x)
        for s in pde.sensors:
            s.bind(pde)
    # Records from the reference solver (the true plant), through the operators.
    for s in pde.sensors:
        zt = s.points()[1].numpy().ravel()
        field = ref.X if s.expression == "X" else ref.W
        clean = field.interpolate(np.full_like(zt, ev.length), zt)
        rng = np.random.default_rng(seed + (0 if s.expression == "X" else 1000))
        s.set_values(clean + s.noise * rng.standard_normal(clean.size))
    torch.manual_seed(seed)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        normalize=True,
        num_collocation_points=4096,
        batch_size=4096,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
    )
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=100, resample_every=50)
    train_time = time.time() - t0
    grid = torch.linspace(0, T_MAX, 121).reshape(-1, 1)
    if model == "function":
        with torch.no_grad():
            rf = pde.unknown_functions["R_f"]({"t": grid}).ravel().numpy()
    else:
        rf = np.full(121, pde.get_trainable_parameter_values()["R_f"])
    truth = truth_rf(grid.ravel().numpy())
    zz, tt = np.meshgrid(ref.z, ref.t, indexing="ij")
    with torch.no_grad():
        pred = trainer.model(
            torch.tensor(np.stack([zz.ravel(), tt.ravel()], 1), dtype=torch.float32)
        ).numpy()
    X_pred = pred[:, 1].reshape(zz.shape)
    return {
        "model": model,
        "seed": seed,
        "steps": steps,
        "train_time_s": train_time,
        "rf_rel_l2": float(np.linalg.norm(rf - truth) / np.linalg.norm(truth)),
        "rf_end_error": float(abs(rf[-1] - truth[-1]) / truth[-1]),
        "rf_mean_abs_error": float(np.mean(np.abs(rf - truth))),
        "rel_l2_X_minus_feed": float(
            np.linalg.norm(X_pred - ref.X.u) / np.linalg.norm(ref.X.u - 0.1)
        ),
        "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
        "profile": rf.tolist(),
        "truth": truth.tolist(),
        "t": grid.ravel().tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(m, s, args.steps) for s in range(args.seeds) for m in ("function", "scalar")]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(run_cell, jobs):
            records.append(rec)
            print(
                f"{rec['model']:<8} seed={rec['seed']} R_f rel L2 {rec['rf_rel_l2']:.3f} end err {rec['rf_end_error']:.3f} "
                f"X err {rec['rel_l2_X_minus_feed']:.3f} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
