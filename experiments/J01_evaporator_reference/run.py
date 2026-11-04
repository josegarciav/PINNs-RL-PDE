"""J01 / J02: the falling-film evaporator reference simulator, and the PINN forward solve of it.

J01 (seconds). The model of pinnrl.process.FallingFilmEvaporator:
* the steady profile and its sensitivity to steam temperature, feed and fouling;
* grid convergence of the dynamic solver (nz = 26 ... 801) on a steam-temperature
  step, against a Richardson extrapolation of the outlet;
* three operating scenarios on a 20-minute window: a 5 K steam step, a 20 %
  feed ramp, and a fouling drift R_f: 0 -> 2e-4 m^2 K/W over the window (an
  8-hour drift compressed, so the transport and the drift both show).

J02 (minutes). The CustomPDE form of the model trained forward for the steam-
step scenario, normalized (W around 0.1 kg/s, X around 0.1, z in metres, t in
seconds), Adam 4,000 steps then L-BFGS 100 steps, 3 seeds, scored against the
reference on W and X and on the outlet X(L, t).

Usage: python experiments/J01_evaporator_reference/run.py [--skip-pinn] [--seeds 3] [--workers 3]
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
T_MAX = 1200.0


def scenarios():
    from pinnrl.process import FallingFilmEvaporator

    return {
        "steam step +5 K at 300 s": FallingFilmEvaporator(
            T_s=lambda t: 355.15 + (5.0 if t > 300 else 0.0)
        ),
        "feed ramp +20 % over 300-600 s": FallingFilmEvaporator(
            W_in=lambda t: 0.1 * (1.0 + 0.2 * min(max((t - 300) / 300, 0.0), 1.0))
        ),
        "fouling drift 0 -> 2e-4 over the window": FallingFilmEvaporator(
            R_f=lambda t: 2e-4 * t / T_MAX
        ),
    }


def reference_part() -> dict:
    from pinnrl.process import FallingFilmEvaporator

    out: dict = {}
    base = FallingFilmEvaporator()
    ss = base.steady_state()
    out["steady"] = {
        "residence_time_s": base.residence_time(),
        "outlet_W": float(ss["W"][-1]),
        "outlet_X": float(ss["X"][-1]),
        "evaporated_fraction": float(1 - ss["W"][-1] / 0.1),
        "flux_inlet": float(ss["q"][0]),
        "flux_outlet": float(ss["q"][-1]),
    }
    sens = {}
    for name, ev in {
        "steam +1 K": FallingFilmEvaporator(T_s=356.15),
        "feed +10 %": FallingFilmEvaporator(W_in=0.11),
        "feed solids +0.01": FallingFilmEvaporator(X_in=0.11),
        "fouling 1e-4": FallingFilmEvaporator(R_f=1e-4),
        "fouling 2e-4": FallingFilmEvaporator(R_f=2e-4),
    }.items():
        s = ev.steady_state()
        sens[name] = {"outlet_X": float(s["X"][-1]), "d_outlet_X": float(s["X"][-1] - ss["X"][-1])}
    out["steady_sensitivity"] = sens

    step = scenarios()["steam step +5 K at 300 s"]
    finals = {}
    t0 = time.time()
    for nz in (26, 51, 101, 201, 401, 801):
        t1 = time.time()
        r = step.reference(t_max=600.0, nz=nz, nt=121)
        finals[nz] = {"outlet_X_600": float(r.outlet()["X"][-1]), "time_s": time.time() - t1}
    vals = [finals[n]["outlet_X_600"] for n in (201, 401, 801)]
    rich = vals[2] + (vals[2] - vals[1])  # first-order Richardson extrapolation
    out["grid_convergence"] = {
        "finals": finals,
        "richardson_outlet_X_600": float(rich),
        "errors": {n: float(abs(finals[n]["outlet_X_600"] - rich)) for n in finals},
        "total_time_s": time.time() - t0,
    }
    out["scenarios"] = {}
    for name, ev in scenarios().items():
        r = ev.reference(t_max=T_MAX, nz=201, nt=241)
        o = r.outlet()
        out["scenarios"][name] = {
            "t": o["t"].tolist(),
            "outlet_W": o["W"].tolist(),
            "outlet_X": o["X"].tolist(),
            "X_range": [float(o["X"].min()), float(o["X"].max())],
        }
    return out


def run_pinn(job):
    seed, steps = job
    import torch

    torch.set_num_threads(2)
    ev = scenarios()["steam step +5 K at 300 s"]
    ref = ev.reference(t_max=T_MAX, nz=201, nt=241)
    pde = ev.custom_pde(t_max=T_MAX)
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
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=100, resample_every=50)
    train_time = time.time() - t0
    zz, tt = np.meshgrid(ref.z, ref.t, indexing="ij")
    inp = torch.tensor(np.stack([zz.ravel(), tt.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        pred = trainer.model(inp).numpy()
    W_pred, X_pred = pred[:, 0].reshape(zz.shape), pred[:, 1].reshape(zz.shape)
    out = ref.outlet()
    x_out_pred = X_pred[-1]
    return {
        "seed": seed,
        "steps": steps,
        "train_time_s": train_time,
        "rel_l2_W": float(np.linalg.norm(W_pred - ref.W.u) / np.linalg.norm(ref.W.u)),
        "rel_l2_X": float(np.linalg.norm(X_pred - ref.X.u) / np.linalg.norm(ref.X.u)),
        "rel_l2_X_minus_feed": float(
            np.linalg.norm(X_pred - ref.X.u) / np.linalg.norm(ref.X.u - 0.1)
        ),
        "outlet_X_max_abs_error": float(np.max(np.abs(x_out_pred - out["X"]))),
        "outlet_X_pred": x_out_pred.tolist(),
        "outlet_X_ref": out["X"].tolist(),
        "t": ref.t.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--skip-pinn", action="store_true")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ref = reference_part()
    (args.out / "reference.json").write_text(json.dumps(ref, indent=1))
    print("steady:", ref["steady"])
    print("grid errors:", ref["grid_convergence"]["errors"])
    for name, s in ref["scenarios"].items():
        print(f"{name}: outlet X range {np.round(s['X_range'], 4)}")
    if args.skip_pinn:
        return
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(run_pinn, [(s, args.steps) for s in range(args.seeds)]):
            records.append(rec)
            print(
                f"pinn seed={rec['seed']} W {rec['rel_l2_W']:.4f} X {rec['rel_l2_X']:.4f} "
                f"(X - feed) {rec['rel_l2_X_minus_feed']:.3f} outlet max err {rec['outlet_X_max_abs_error']:.4f} "
                f"{rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
