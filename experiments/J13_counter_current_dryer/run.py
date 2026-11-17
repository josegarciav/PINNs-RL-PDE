"""J13: the counter-current tower, the first unit whose steady state is not an initial-value problem.

Three units so far (the evaporator tube, the droplet, the co-current tower)
had a forward solve of milliseconds, and on each the classical estimator
through the reference solver beat the PINN twin. The roadmap's condition for
the PINN's case is a unit without a cheap trusted forward solve. The
counter-current tower (``pinnrl.process.CounterCurrentDryer``) is the first
candidate: the air enters at the bottom against the falling droplets, so the
steady state is a two-point boundary-value problem and the transient a
hyperbolic system with characteristics running both ways. This experiment
asks:

1. ``cost``: what the steady and transient solves cost against the co-current
   tower, by grid and tolerance, on a smooth step and on a plant-like record;
2. ``forward``: how well and how fast the PINN solves the step response of
   the four fields in (z, t), against the method-of-lines reference;
3. ``identify``: the critical moisture and exponent of the drying curve and
   the wall-loss coefficient from a noisy transient record (five air
   thermocouples and the outlet humidity every 10 s, the powder's moisture
   every minute), by bounded least squares through the reference and by the
   PINN twin with the three parameters trainable, on three data sets.

Usage: python experiments/J13_counter_current_dryer/run.py [--arms cost,forward,identify]
           [--datasets 3] [--seeds 2] [--steps 2000] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
T_MAX = 600.0
NZ_PLANT = 201  # the plant and the fitted model share the grid (a "known model" scenario)
PROBES = (0.5, 2.0, 3.5, 5.0, 6.5)
NOISE = {"T_air": 0.5, "Y_out": 1e-3, "X_out": 3e-3}
FAST_TIMES = np.arange(10.0, T_MAX + 1e-9, 10.0)  # thermocouples and hygrometer
LAB_TIMES = np.arange(60.0, T_MAX + 1e-9, 60.0)  # powder samples
TRUTH = {"Xcr": 1.0, "ncdc": 1.5, "U_w": 1.0}
GUESS = {"Xcr": 0.7, "ncdc": 1.0, "U_w": 1.3}
BOUNDS = {"Xcr": (0.3, 1.45), "ncdc": (0.5, 3.0), "U_w": (0.0, 3.0)}
NAMES = ("Xcr", "ncdc", "U_w")


def inlet_step(t: float) -> float:
    """The inlet air: 160 degC, then +15 K from t = 10 s with a 5 s time constant."""
    return 433.15 + 15.0 * (1.0 - np.exp(-max(t - 10.0, 0.0) / 5.0))


def tower(theta=None, **changes):
    """The counter-current tower with the step on its inlet air and, optionally, other parameters."""
    from dataclasses import replace

    from pinnrl.process import CounterCurrentDryer

    dr = CounterCurrentDryer(T_in=inlet_step, **changes)
    if theta is not None:
        Xcr, n, Uw = theta
        dr = replace(dr, droplet=replace(dr.droplet, X_cr=float(Xcr), n=float(n)), U_wall=float(Uw))
    return dr


def records_of(trans, dr):
    """The record vector of one transient: thermocouples, humidity (every 10 s), powder (every minute)."""
    rec = dr.records(trans, PROBES)
    fast = np.searchsorted(trans.t, FAST_TIMES)
    lab = np.searchsorted(trans.t, LAB_TIMES)
    return np.concatenate([rec["T_air"][:, fast].ravel(), rec["Y_out"][fast], rec["X_out"][lab]])


def sigma_vector():
    return np.concatenate(
        [
            np.full(len(PROBES) * FAST_TIMES.size, NOISE["T_air"]),
            np.full(FAST_TIMES.size, NOISE["Y_out"]),
            np.full(LAB_TIMES.size, NOISE["X_out"]),
        ]
    )


def simulate(theta, nz=NZ_PLANT, rtol=1e-7):
    dr = tower(theta)
    trans = dr.transient(T_MAX, nz=nz, nt=int(T_MAX / 5) + 1, rtol=rtol, atol=rtol * 1e-3)
    return records_of(trans, dr), trans


def dataset(seed):
    """Noisy records of the true plant."""
    clean, trans = simulate([TRUTH[k] for k in NAMES])
    rng = np.random.default_rng(100 + seed)
    return clean + sigma_vector() * rng.standard_normal(clean.size), clean, trans


# ------------------------------------------------------------------------ cost
def best_of(fn, repeats=3):
    best = np.inf
    for _ in range(repeats):
        tic = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - tic)
    return best


def run_cost():
    from pinnrl.process import CounterCurrentDryer, SprayDryer

    out = {}
    co = SprayDryer()
    out["co_current_steady_ms"] = 1e3 * best_of(co.reference)
    dr = CounterCurrentDryer()
    out["counter_steady_ms"] = 1e3 * best_of(dr.steady)
    exact = dr.steady().outlet()["X"]
    out["counter_steady_outlet"] = dr.steady().outlet()
    out["co_current_outlet"] = co.reference().outlet()
    rng = np.random.default_rng(0)
    knots = np.arange(0.0, T_MAX + 10.0, 10.0)
    values = 433.15 + 3.0 * rng.standard_normal(knots.size)
    record = CounterCurrentDryer(T_in=lambda t: float(np.interp(t, knots, values)))
    step = tower()
    grids = {}
    for nz in (51, 101, 201, 401):
        g = {"mol_steady_ms": 1e3 * best_of(lambda n=nz: dr.steady_cells(n), 2)}
        s = dr.steady_cells(nz)
        g["X_out_error_pct"] = 100.0 * (s[nz - 1] / exact - 1.0)
        for label, model in (("step", step), ("record", record)):
            for rtol in (1e-6, 1e-8):
                tr = model.transient(T_MAX, nz=nz, nt=121, rtol=rtol, atol=rtol * 1e-3)
                tr2 = model.transient(T_MAX, nz=nz, nt=121, rtol=rtol, atol=rtol * 1e-3)
                g[f"{label}_rtol{rtol:g}_s"] = min(tr.stats["seconds"], tr2.stats["seconds"])
                g[f"{label}_rtol{rtol:g}_steps"] = tr.stats["steps"]
                g[f"{label}_rtol{rtol:g}_jacobians"] = tr.stats["jacobians"]
        grids[str(nz)] = g
        print(f"cost nz={nz}: {g}", flush=True)
    out["grids"] = grids
    print(
        f"cost: co-current steady {out['co_current_steady_ms']:.1f} ms, "
        f"counter-current steady {out['counter_steady_ms']:.0f} ms",
        flush=True,
    )
    return out


# --------------------------------------------------------------------- the PINN
def _trainer(pde, seed, steps, derivatives="native", lbfgs_steps=200):
    """J06's recipe with the residual weight lowered to 1e-3 and a longer L-BFGS finish.

    The package multiplies a residual by the time span over the field's scale,
    which assumes the fields change over the whole span; here the droplets
    cross the tower in 5 s and the air in 100 s of a 600 s window, and the
    droplet-temperature equation is a stiff balance (``custom_pde`` divides
    each equation by its own scale). With the residual at weight 1 the
    initial and boundary terms are never fitted (moisture 90 % off); at 1e-3
    the forward solve reaches about 10 %, and L-BFGS does the learning.
    """
    import torch

    torch.manual_seed(seed)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        normalize=True,
        derivatives=derivatives,
        num_collocation_points=4096,
        batch_size=4096,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1e-3, "boundary": 10.0, "initial": 10.0, "data": 10.0},
    )
    native = pde.native_active(trainer.model)
    tic = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=lbfgs_steps, resample_every=50)
    return trainer, time.time() - tic, native


def _field_errors(model, trans):
    import torch

    sols = trans.as_reference_solutions()
    zz, tt = np.meshgrid(trans.z, trans.t, indexing="ij")
    inp = torch.tensor(np.stack([zz.ravel(), tt.ravel()], 1), dtype=torch.float32)
    with torch.no_grad():
        u = model(inp).numpy()
    return {name: sols[name].relative_l2(u[:, k]) for k, name in enumerate(("X", "T", "Y", "Ta"))}


def _sensors():
    from pinnrl.pdes import PointSensor

    s = [
        PointSensor(z, FAST_TIMES, expression="Ta", noise=NOISE["T_air"], name=f"T_air@{z}")
        for z in PROBES
    ]
    s.append(PointSensor(0.0, FAST_TIMES, expression="Y", noise=NOISE["Y_out"], name="Y_out"))
    s.append(PointSensor(8.0, LAB_TIMES, expression="X", noise=NOISE["X_out"], name="X_out"))
    return s


def _set_records(pde, y):
    k = 0
    for s in pde.sensors:
        n = len(s.times)
        s.set_values(y[k : k + n])
        k += n


def run_forward(job):
    """The PINN forward solve of the step response, graded against a 401-cell reference."""
    import torch

    logging.disable(logging.WARNING)
    torch.set_num_threads(job["threads"])
    dr = tower()
    fine = dr.transient(T_MAX, nz=401, nt=121, rtol=1e-8, atol=1e-11)
    coarse = dr.transient(T_MAX, nz=NZ_PLANT, nt=121, rtol=1e-8, atol=1e-11)
    pde = dr.custom_pde(T_MAX)
    trainer, seconds, native = _trainer(pde, job["seed"], job["steps"])
    errors = _field_errors(trainer.model, fine)
    # the discretisation gap between the two grids, in the same norm, as the yardstick
    gap = {
        name: fine.as_reference_solutions()[name].relative_l2(
            np.stack(
                [
                    np.interp(fine.z, coarse.z, coarse.as_reference_solutions()[name].u[:, j])
                    for j in range(fine.t.size)
                ],
                axis=1,
            ).ravel()
        )
        for name in ("X", "T", "Y", "Ta")
    }
    rec_true = dr.records(fine, PROBES)
    zz = np.array(PROBES)
    with torch.no_grad():
        pts = torch.tensor(
            np.stack([np.repeat(zz, fine.t.size), np.tile(fine.t, zz.size)], 1), dtype=torch.float32
        )
        Ta = trainer.model(pts)[:, 3].numpy().reshape(zz.size, fine.t.size)
        bottom = torch.tensor(np.stack([np.full(fine.t.size, 8.0), fine.t], 1), dtype=torch.float32)
        X_out = trainer.model(bottom)[:, 0].numpy()
    out = {
        **job,
        "train_time_s": seconds,
        "native": native,
        "rel_l2": errors,
        "grid_gap_rel_l2": gap,
        "T_air_rmse_K": float(np.sqrt(np.mean((Ta - rec_true["T_air"]) ** 2))),
        "X_out_rmse": float(np.sqrt(np.mean((X_out - rec_true["X_out"]) ** 2))),
        "T_air_pinn": Ta.tolist(),
        "T_air_ref": rec_true["T_air"].tolist(),
        "X_out_pinn": X_out.tolist(),
        "X_out_ref": rec_true["X_out"].tolist(),
        "t": fine.t.tolist(),
    }
    print(
        f"forward seed={job['seed']}: {seconds:.0f} s, rel L2 {errors}, grid gap {gap}", flush=True
    )
    return out


def run_pinn_identify(job):
    """The PINN twin with Xcr, ncdc and U_w trainable on data set ``dataset``."""
    import torch

    logging.disable(logging.WARNING)
    torch.set_num_threads(job["threads"])
    y, _clean, trans = dataset(job["dataset"])
    dr = tower()
    pde = dr.custom_pde(T_MAX, identify=list(NAMES), initial_guesses=GUESS, sensors=_sensors())
    _set_records(pde, y)
    trainer, seconds, native = _trainer(pde, 1000 * job["dataset"] + job["seed"], job["steps"])
    est = pde.get_trainable_parameter_values()
    misfit = pde.sensor_misfit(trainer.model)
    errors = _field_errors(trainer.model, trans)
    out = {
        **job,
        "train_time_s": seconds,
        "native": native,
        "estimate": {k: float(est[k]) for k in NAMES},
        "error": {k: float(est[k] - TRUTH[k]) for k in NAMES},
        "chi2_reduced": float(misfit.reduced),
        "accepted": bool(misfit.accepted),
        "rel_l2": errors,
    }
    print(
        f"pinn identify dataset={job['dataset']} seed={job['seed']}: {seconds:.0f} s, "
        f"estimate {out['estimate']}, chi2 {misfit.reduced:.2f}",
        flush=True,
    )
    return out


# ----------------------------------------------------------------- least squares
def run_lsq(seed):
    """Bounded least squares through the method-of-lines reference, from the same cold start."""
    from scipy.optimize import least_squares

    y, _clean, _trans = dataset(seed)
    sig = sigma_vector()
    calls = {"n": 0}

    def residual(theta):
        calls["n"] += 1
        return (simulate(theta)[0] - y) / sig

    tic = time.perf_counter()
    res = least_squares(
        residual,
        [GUESS[k] for k in NAMES],
        bounds=([BOUNDS[k][0] for k in NAMES], [BOUNDS[k][1] for k in NAMES]),
        x_scale=[0.1, 0.1, 0.1],
        diff_step=1e-3,
        xtol=1e-8,
        ftol=1e-10,
    )
    seconds = time.perf_counter() - tic
    est = dict(zip(NAMES, res.x.tolist(), strict=True))
    out = {
        "dataset": seed,
        "seconds": seconds,
        "solves": calls["n"],
        "estimate": est,
        "error": {k: est[k] - TRUTH[k] for k in NAMES},
        "chi2_reduced": float(np.mean(res.fun**2)),
        "success": bool(res.success),
    }
    print(
        f"lsq dataset={seed}: {seconds:.0f} s, {calls['n']} solves, estimate {est}, "
        f"chi2 {out['chi2_reduced']:.2f}",
        flush=True,
    )
    return out


def linear_prediction():
    """The linear-Gaussian standard deviations of the three parameters from the record's sensitivities."""
    from pinnrl.analysis import posterior_covariance

    sig = sigma_vector()
    theta0 = np.array([TRUTH[k] for k in NAMES])
    cols = []
    for j, h in enumerate([0.01, 0.01, 0.01]):
        up, dn = theta0.copy(), theta0.copy()
        up[j] += h
        dn[j] -= h
        cols.append((simulate(up)[0] - simulate(dn)[0]) / (2 * h))
    J = np.stack(cols, axis=1) / sig[:, None]
    cov = posterior_covariance(J.T @ J)
    std = np.sqrt(np.diag(cov))
    corr = cov / np.outer(std, std)
    return {"std": dict(zip(NAMES, std.tolist(), strict=True)), "correlation": corr.tolist()}


# ------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="cost,forward,identify")
    parser.add_argument("--datasets", type=int, default=3)
    parser.add_argument("--seeds", type=int, default=2, help="PINN seeds per data set")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    arms = args.arms.split(",")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "j13.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    if "cost" in arms:
        results["cost"] = run_cost()
        path.write_text(json.dumps(results, indent=1))

    profile = tower().steady(n=201)
    results["steady_profile"] = {
        k: getattr(profile, k).tolist() for k in ("z", "X", "T_p", "Y", "T_a")
    }
    from pinnrl.process import SprayDryer

    co = SprayDryer().reference(n=201)
    results["co_current_profile"] = {
        k: getattr(co, k).tolist() for k in ("z", "X", "T_p", "Y", "T_a")
    }

    jobs_f = [
        {"seed": s, "steps": args.steps, "threads": args.threads} for s in range(args.datasets)
    ]
    jobs_i = [
        {"dataset": d, "seed": s, "steps": args.steps, "threads": args.threads}
        for d in range(args.datasets)
        for s in range(args.seeds)
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if "forward" in arms:
            results["forward"] = list(pool.map(run_forward, jobs_f))
            path.write_text(json.dumps(results, indent=1))
        if "identify" in arms:
            futures = [pool.submit(run_pinn_identify, j) for j in jobs_i]
            results["linear_prediction"] = linear_prediction()
            print("linear prediction", results["linear_prediction"], flush=True)
            results["lsq"] = [run_lsq(d) for d in range(args.datasets)]
            results["pinn_identify"] = [f.result() for f in futures]
            path.write_text(json.dumps(results, indent=1))
    path.write_text(json.dumps(results, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
