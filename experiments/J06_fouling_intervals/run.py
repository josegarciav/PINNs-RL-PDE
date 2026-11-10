"""J06: are the uncertainty intervals of the fouling resistance calibrated?

The plant and records of J03 (the J01 evaporator over 20 minutes, the steam
wandering +-1 K and recorded, outlet X and W every 5 s with noise 0.002 and
0.001), over many noise draws, with every estimator reporting a 90 % interval
for three numbers: the fouling at the end of the window R_f(T), the fouling
two minutes earlier R_f(0.9 T), and the time until R_f reaches the cleaning
threshold 3e-4 m2K/W (600 s after T on the ramp). Coverage of the truth over
the data sets, width and bias are the metrics (F06's protocol).

Scenarios: "known" (the ramp R_f = 2e-4 t/T, inputs exact), "bias" (the same
with the estimator's steam record offset by b ~ N(0, 0.3 K), drawn per data
set) and "saturating" (a Kern-Seaton curve to the same end value, inputs exact).

Estimators:
  batch-2, batch-6   Gauss-Newton on the reference solver (the plant's own grid)
                     with R_f as 2 or 6 Legendre modes of time; the interval is
                     the Gaussian of the Fisher covariance at the estimate;
  ... + bias         the same with the steam offset among the unknowns under its
                     prior N(0, 0.3 K);
  ekf-rw             the J05 filter (11 cells), R_f a random walk;
  ekf-llt            the same with R_f a local linear trend (level and rate, the
                     rate a random walk), which is what gives a time to clean;
  ... + bias         the filters with the steam offset as a constant state;
                     (J05 clips the fouling at zero after each update; the
                     fine-grid filters of this scenario are also run unclipped);
  pinn               the J03 twin (R_f an UnknownFunction of t), an ensemble of
                     initializations per data set, with the Fisher width, the
                     ensemble spread and both as candidate intervals.

Usage: python experiments/J06_fouling_intervals/run.py [--datasets 100] [--workers 8]
       python experiments/J06_fouling_intervals/run.py --part pinn [--pinn-datasets 12] [--members 3]
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

os.environ.setdefault("TQDM_DISABLE", "1")

HERE = Path(__file__).parent
T_MAX = 1200.0
R_END = 2e-4
EVERY = 5.0
NOISE_X, NOISE_W = 0.002, 0.001
NZ, NT = 201, 241  # the plant's grid; nt gives one output per record time
C_SCALE = 1e-4  # the modes are in units of the mean fouling
R_CLEAN = 3e-4  # the cleaning threshold
BIAS_STD = 0.3  # K, the prior of the steam record's offset
TAU = 400.0  # s, the time constant of the saturating curve
LEVEL = 0.9
Z = 1.6448536269514722  # the 95 % normal quantile
T90 = 0.9 * T_MAX
TIMES = np.arange(EVERY, T_MAX + 1e-9, EVERY)
I90 = int(round(T90 / EVERY)) - 1  # index of T90 in TIMES

TRUTHS = {
    "ramp": lambda t: R_END * np.asarray(t, dtype=float) / T_MAX,
    "saturating": lambda t: (
        R_END * (1.0 - np.exp(-np.asarray(t, dtype=float) / TAU)) / (1.0 - math.exp(-T_MAX / TAU))
    ),
}
SCENARIOS = {  # name: (truth, the steam record carries an offset)
    "known": ("ramp", False),
    "bias": ("ramp", True),
    "saturating": ("saturating", False),
}


def steam(t):
    return 355.15 + 1.0 * math.sin(2 * math.pi * t / 400.0)


@functools.cache
def clean_records(truth):
    """The plant's noise-free outlet records ``(K, 2)`` = [X, W] at ``TIMES``."""
    from pinnrl.process import FallingFilmEvaporator

    fn = TRUTHS[truth]
    ev = FallingFilmEvaporator(T_s=steam, R_f=lambda t: float(fn(t)))
    ref = ev.reference(t_max=T_MAX, nz=NZ, nt=NT)
    return np.stack([ref.X.u[-1, 1:], ref.W.u[-1, 1:]], axis=1)


def dataset(scenario, seed):
    """The records of one data set and the offset of its steam record (J03's noise seeds)."""
    truth, biased = SCENARIOS[scenario]
    clean = clean_records(truth)
    y = clean.copy()
    y[:, 0] += NOISE_X * np.random.default_rng(seed).standard_normal(TIMES.size)
    y[:, 1] += NOISE_W * np.random.default_rng(seed + 1000).standard_normal(TIMES.size)
    b = BIAS_STD * float(np.random.default_rng(seed + 2000).standard_normal()) if biased else 0.0
    return y, b


def target_truths(scenario):
    fn = TRUTHS[SCENARIOS[scenario][0]]
    out = {"end": float(fn(T_MAX)), "t90": float(fn(T90))}
    if SCENARIOS[scenario][0] == "ramp":
        out["ttc"] = T_MAX * R_CLEAN / R_END - T_MAX
    return out


def gaussian(est, std):
    return {
        "est": float(est),
        "std": float(std),
        "lo": float(est - Z * std),
        "hi": float(est + Z * std),
    }


def time_to_clean(level, rate, cov, seed=0):
    """The time for a level growing at ``rate`` to reach ``R_CLEAN``: median and 90 % interval."""
    from pinnrl.process import time_to_threshold

    tt = time_to_threshold(level, rate, cov, R_CLEAN, level=LEVEL, rising=True, seed=seed)
    return {"est": tt.median, "lo": tt.lower, "hi": tt.upper}


# ------------------------------------------------------------------ batch
def model_records(c, beta, record_bias):
    """Outlet records of the reference model with fouling modes ``c`` and steam ``record - beta``."""
    from pinnrl.process import FallingFilmEvaporator

    c = np.asarray(c, dtype=float)
    shift = record_bias - beta
    ev = FallingFilmEvaporator(
        T_s=lambda t: steam(t) + shift,
        R_f=lambda t: C_SCALE * float(legendre.legval(2.0 * t / T_MAX - 1.0, c)),
    )
    ref = ev.reference(t_max=T_MAX, nz=NZ, nt=NT)
    return np.concatenate([ref.X.u[-1, 1:], ref.W.u[-1, 1:]])


def run_batch(job):
    scenario, seed, modes, with_bias = job["scenario"], job["dataset"], job["modes"], job["bias"]
    from scipy.optimize import least_squares

    y, b = dataset(scenario, seed)
    yv = np.concatenate([y[:, 0], y[:, 1]])
    sig = np.concatenate([np.full(TIMES.size, NOISE_X), np.full(TIMES.size, NOISE_W)])

    def resid(theta):
        beta = theta[modes] if with_bias else 0.0
        r = (model_records(theta[:modes], beta, b) - yv) / sig
        return np.append(r, beta / BIAS_STD) if with_bias else r

    x0 = np.zeros(modes + int(with_bias))
    x0[0] = 1.0  # the filters' and the twin's start, 1e-4
    tic = time.perf_counter()
    res = least_squares(resid, x0, method="trf", diff_step=0.02, xtol=1e-7, ftol=1e-10)
    cov = np.linalg.inv(res.jac.T @ res.jac)
    c, cc = res.x[:modes], cov[:modes, :modes]
    targets = {}
    for name, s in (("end", 1.0), ("t90", 2.0 * T90 / T_MAX - 1.0)):
        a = legendre.legvander(np.array([s]), modes - 1)[0]
        targets[name] = gaussian(C_SCALE * a @ c, C_SCALE * math.sqrt(a @ cc @ a))
    if modes == 2 and "ttc" in target_truths(scenario):
        # R_f = 1e-4 (c0 + c1 s), s = 2t/T - 1: level c0 + c1 at T, rate 2 c1 / T.
        A = C_SCALE * np.array([[1.0, 1.0], [0.0, 2.0 / T_MAX]])
        targets["ttc"] = time_to_clean(*(A @ c), A @ cc @ A.T, seed=seed)
    out = {
        "targets": targets,
        "time_s": time.perf_counter() - tic,
        "nfev": int(res.nfev),
        "chi2": float(2 * res.cost),
    }
    if with_bias:
        out["beta"] = float(res.x[modes])
        out["beta_std"] = float(math.sqrt(cov[modes, modes]))
        out["bias_truth"] = b
    if seed == 0:
        grid = np.linspace(0.0, T_MAX, 121)
        V = legendre.legvander(2.0 * grid / T_MAX - 1.0, modes - 1)
        out["profile"] = (C_SCALE * V @ c).tolist()
        out["profile_std"] = (C_SCALE * np.sqrt(np.einsum("ij,jk,ik->i", V, cc, V))).tolist()
        out["t"] = grid.tolist()
    return {**job, **out}


# ------------------------------------------------------------------ filters
def run_ekf(job):
    scenario, seed = job["scenario"], job["dataset"]
    nz, drift, sigma, with_bias = job["nz"], job["drift"], job["sigma"], job["bias"]
    from pinnrl.process import ExtendedKalmanFilter, FallingFilmEvaporator

    y, b = dataset(scenario, seed)
    hr, hb = [0.0], [0.0]
    ev = FallingFilmEvaporator(T_s=lambda t: steam(t) + b - hb[0], R_f=lambda t: hr[0])
    rhs = ev.mol_rhs(nz)
    n = 2 * (nz - 1)
    k_rate = n + 1 if drift == "llt" else None
    k_bias = n + 1 + int(drift == "llt") if with_bias else None
    dim = n + 1 + int(drift == "llt") + int(with_bias)

    def f(t, x):
        hr[0] = float(x[n])
        hb[0] = float(x[k_bias]) if with_bias else 0.0
        full = np.concatenate([[0.1], x[: nz - 1], [0.1], x[nz - 1 : n]])
        d = rhs(t, full)
        out = np.zeros(dim)
        out[: nz - 1], out[nz - 1 : n] = d[1:nz], d[nz + 1 :]
        if k_rate is not None:
            out[n] = x[k_rate]
        return out

    def h(t, x):
        return np.array([x[2 * nz - 3], x[nz - 2]])

    ss = ev.steady_state(nz, t=0.0)  # at R_f = 0 and the recorded steam, as in J05
    x0, P0, Q = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    x0[:n], P0[:n], Q[:n] = np.concatenate([ss["W"][1:], ss["X"][1:]]), 1e-6, 1e-12
    x0[n], P0[n] = 1e-4, (1e-4) ** 2
    if drift == "llt":
        P0[k_rate], Q[k_rate] = (1e-6) ** 2, sigma**2
    else:
        Q[n] = sigma**2
    if with_bias:
        P0[k_bias] = BIAS_STD**2
    lower = np.full(dim, -np.inf)
    if job.get("clip", True):  # R_f >= 0, as in J05
        lower[n] = 0.0
    ekf = ExtendedKalmanFilter(f, h, Q=Q, R=np.array([NOISE_X**2, NOISE_W**2]))
    res = ekf.run(TIMES, y, x0, np.diag(P0), smooth=True, lower=lower)
    truth = TRUTHS[SCENARIOS[scenario][0]](TIMES)
    sm, sm_std = res.smoothed[:, n], res.std(smoothed=True)[:, n]
    sel = TIMES >= 120.0
    targets = {
        "end": gaussian(res.filtered[-1, n], res.std()[-1, n]),
        "t90": gaussian(sm[I90], sm_std[I90]),
    }
    if drift == "llt" and "ttc" in target_truths(scenario):
        idx = [n, k_rate]
        targets["ttc"] = time_to_clean(
            res.filtered[-1, n],
            res.filtered[-1, k_rate],
            res.filtered_cov[-1][np.ix_(idx, idx)],
            seed=seed,
        )
    out = {
        "targets": targets,
        "time_s": res.time_s,
        "rel_l2_smoother": float(np.linalg.norm((sm - truth)[sel]) / np.linalg.norm(truth[sel])),
        "band_coverage": float(np.mean(np.abs(sm - truth)[sel] <= Z * sm_std[sel])),
    }
    if with_bias:
        out["beta"] = float(res.filtered[-1, k_bias])
        out["beta_std"] = float(res.std()[-1, k_bias])
        out["bias_truth"] = b
    if seed == 0:
        out["profile"] = sm.tolist()
        out["profile_std"] = sm_std.tolist()
        out["profile_filter"] = res.filtered[:, n].tolist()
        out["profile_filter_std"] = res.std()[:, n].tolist()
        out["t"] = TIMES.tolist()
    return {**job, **out}


def run_classical(job):
    return run_batch(job) if job["kind"] == "batch" else run_ekf(job)


def classical_jobs(n_datasets):
    arms = []

    def batch(scenario, modes, bias=False):
        name = f"batch-{modes}" + (" + bias" if bias else "")
        arms.append(
            {"kind": "batch", "arm": name, "scenario": scenario, "modes": modes, "bias": bias}
        )

    def ekf(scenario, drift, sigma, nz=11, bias=False, clip=True):
        name = (
            f"ekf-{drift} {sigma:.0e}"
            + (f" nz{nz}" if nz != 11 else "")
            + (" + bias" if bias else "")
            + ("" if clip else ", no clip")
        )
        arms.append(
            {
                "kind": "ekf",
                "arm": name,
                "scenario": scenario,
                "drift": drift,
                "sigma": sigma,
                "nz": nz,
                "bias": bias,
                "clip": clip,
            }
        )

    for scenario in ("known", "saturating"):
        batch(scenario, 2)
        batch(scenario, 6)
        for nz in (11, 51):
            for sig in (1e-7, 1e-6, 1e-5):
                ekf(scenario, "rw", sig, nz=nz)
            for sig in (1e-10, 1e-9, 1e-8):
                ekf(scenario, "llt", sig, nz=nz)
    for bias in (False, True):
        batch("bias", 2, bias)
        batch("bias", 6, bias)
        for nz in (11, 51):
            ekf("bias", "rw", 1e-6, nz=nz, bias=bias)
            ekf("bias", "llt", 1e-9, nz=nz, bias=bias)
        # J05 clips the fouling at zero after each update; with an offset state that
        # leaves the two inconsistent, so the fine-grid filters are also run unclipped.
        ekf("bias", "rw", 1e-6, nz=51, bias=bias, clip=False)
        ekf("bias", "llt", 1e-9, nz=51, bias=bias, clip=False)
    return [{**a, "dataset": d} for a in arms for d in range(n_datasets)]


# ------------------------------------------------------------------ the PINN twin
def run_pinn(job):
    """J03's function-valued twin on data set ``dataset`` from initialization ``member``."""
    seed, member, steps = job["dataset"], job["member"], job["steps"]
    import torch

    from pinnrl.pdes import PointSensor
    from pinnrl.process import FallingFilmEvaporator

    logging.disable(logging.WARNING)
    torch.set_num_threads(2)
    ev = FallingFilmEvaporator(T_s=steam, R_f=lambda t: R_END * t / T_MAX)
    y, _ = dataset("known", seed)
    sensors = [
        PointSensor(ev.length, TIMES, expression="X", noise=NOISE_X, name="outlet X"),
        PointSensor(ev.length, TIMES, expression="W", noise=NOISE_W, name="outlet W"),
    ]
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
    for s in pde.sensors:
        s.set_values(y[:, 0] if s.expression == "X" else y[:, 1])
    torch.manual_seed(1000 * seed + member)
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
    tic = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=100, resample_every=50)
    grid = torch.linspace(0, T_MAX, 121).reshape(-1, 1)
    with torch.no_grad():
        rf = pde.unknown_functions["R_f"]({"t": grid}).ravel().numpy()
        pred = torch.cat([s.predict(trainer.model) for s in pde.sensors]).ravel().numpy()
    truth = TRUTHS["ramp"](grid.ravel().numpy())
    sig = np.concatenate([np.full(TIMES.size, NOISE_X), np.full(TIMES.size, NOISE_W)])
    return {
        **job,
        "train_time_s": time.time() - tic,
        "end": float(rf[-1]),
        "t90": float(rf[108]),
        "rel_l2": float(np.linalg.norm(rf - truth) / np.linalg.norm(truth)),
        "chi2": float((((pred - np.concatenate([y[:, 0], y[:, 1]])) / sig) ** 2).sum()),
        "lbfgs_failures": int(getattr(trainer, "_lbfgs_failures", 0)),
        "profile": rf.tolist(),
        "t": grid.ravel().tolist(),
    }


# ------------------------------------------------------------------ summaries
def summarize_classical(records):
    out = {}
    keys = sorted({(r["scenario"], r["arm"]) for r in records})
    for scenario, arm in keys:
        rows = [r for r in records if r["scenario"] == scenario and r["arm"] == arm]
        truths = target_truths(scenario)
        entry = {"n": len(rows), "time_s": float(np.mean([r["time_s"] for r in rows]))}
        for name, truth in truths.items():
            tg = [r["targets"][name] for r in rows if name in r["targets"]]
            if not tg:
                continue
            est = np.array([t["est"] for t in tg])
            lo, hi = np.array([t["lo"] for t in tg]), np.array([t["hi"] for t in tg])
            finite = np.isfinite(hi)
            entry[name] = {
                "coverage": float(np.mean((lo <= truth) & (truth <= hi))),
                # the median width of the bounded intervals; "unbounded" counts the others
                "width_rel": float(np.median((hi - lo)[finite]) / truth) if finite.any() else None,
                "bias_rel": float((np.median(est) - truth) / truth),
                "rmse_rel": float(np.sqrt(np.mean((est[finite] - truth) ** 2)) / truth),
                "spread_rel": float(np.std(est[finite]) / truth),
                "unbounded": float(1.0 - finite.mean()),
            }
            if "std" in tg[0]:
                entry[name]["std_rel"] = float(np.mean([t["std"] for t in tg]) / truth)
        if "band_coverage" in rows[0]:
            entry["band_coverage"] = float(np.mean([r["band_coverage"] for r in rows]))
            entry["rel_l2_smoother"] = float(np.mean([r["rel_l2_smoother"] for r in rows]))
        if "beta" in rows[0]:
            err = np.array([r["beta"] - r["bias_truth"] for r in rows])
            std = np.array([r["beta_std"] for r in rows])
            entry["beta"] = {
                "rmse": float(np.sqrt(np.mean(err**2))),
                "std": float(std.mean()),
                "coverage": float(np.mean(np.abs(err) <= Z * std)),
            }
        out[f"{scenario} | {arm}"] = entry
    return out


def summarize_pinn(runs, fisher_std):
    """Candidate intervals for the twin's estimate, per target; ``fisher_std`` from batch-6.

    A member is accepted by the misfit guard when the chi-square of its own
    sensor predictions against the records is below the 99.9 % quantile for
    their number (a run that collapsed to a near-constant fouling does not
    explain the records). Every interval is reported for the accepted members
    and, as the control, for all of them.
    """
    from scipy.stats import chi2
    from scipy.stats import t as student

    truths = target_truths("known")
    limit = float(chi2.ppf(0.999, 2 * TIMES.size))
    datasets = sorted({r["dataset"] for r in runs})
    accepted = [r for r in runs if r["chi2"] <= limit]
    out = {
        "n_datasets": len(datasets),
        "n_runs": len(runs),
        "chi2_limit": limit,
        "accepted_fraction": len(accepted) / len(runs),
        "datasets_with_an_accepted_member": len({r["dataset"] for r in accepted}),
        "chi2_accepted": [min(r["chi2"] for r in accepted), max(r["chi2"] for r in accepted)],
        "rel_l2_accepted": [min(r["rel_l2"] for r in accepted), max(r["rel_l2"] for r in accepted)],
        "train_time_s": float(np.mean([r["train_time_s"] for r in runs])),
    }
    rejected = [r for r in runs if r["chi2"] > limit]
    if rejected:
        out["chi2_rejected"] = [min(r["chi2"] for r in rejected), max(r["chi2"] for r in rejected)]
        out["rel_l2_rejected"] = [
            min(r["rel_l2"] for r in rejected),
            max(r["rel_l2"] for r in rejected),
        ]
    for name in ("end", "t90"):
        truth, sf = truths[name], fisher_std[name]
        entry = {"fisher_std_rel": float(sf / truth)}
        for label, pool in (("accepted", accepted), ("all", runs)):
            groups = [
                np.array(
                    [r[name] for r in sorted(pool, key=lambda r: r["member"]) if r["dataset"] == d]
                )
                for d in datasets
            ]
            groups = [g for g in groups if g.size]
            err = np.concatenate(groups) / truth - 1.0
            multi = [g for g in groups if g.size >= 2]
            cands = {"fisher at one member": [(g[0] - Z * sf, g[0] + Z * sf) for g in groups]}
            ens, both = [], []
            for g in multi:
                tq, sd = float(student.ppf(0.95, g.size - 1)), g.std(ddof=1)
                ens.append((g.mean() - tq * sd, g.mean() + tq * sd))
                wide = tq * math.sqrt(sd**2 + sf**2)
                both.append((g.mean() - wide, g.mean() + wide))
            cands["ensemble"], cands["ensemble + fisher"] = ens, both
            entry[label] = {
                "n_runs": int(err.size),
                "bias_rel": float(err.mean()),
                "rmse_rel": float(np.sqrt((err**2).mean())),
                "abs_error_q90_rel": float(np.quantile(np.abs(err), 0.9)),
                "within_dataset_std_rel": float(
                    np.sqrt(np.mean([g.var(ddof=1) for g in multi])) / truth
                )
                if multi
                else None,
                "between_dataset_std_rel": float(np.std([g.mean() for g in groups], ddof=1) / truth)
                if len(groups) > 1
                else None,
                "intervals": {
                    k: {
                        "n": len(v),
                        "coverage": float(np.mean([lo <= truth <= hi for lo, hi in v])),
                        "width_rel": float(np.median([hi - lo for lo, hi in v]) / truth),
                    }
                    for k, v in cands.items()
                    if v
                },
            }
        out[name] = entry
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--part", choices=["classical", "pinn", "summary"], default="classical")
    parser.add_argument("--datasets", type=int, default=100)
    parser.add_argument("--pinn-datasets", type=int, default=12)
    parser.add_argument("--members", type=int, default=3)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--only", default="", help="classical: rerun only the arms containing this text"
    )
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.part == "classical":
        jobs = classical_jobs(args.datasets)
        records = []
        if args.only:  # rerun the arms whose name contains the text, keep the rest
            jobs = [j for j in jobs if args.only in j["arm"]]
            keep = json.loads((args.out / "classical.json").read_text())
            records = [r for r in keep if args.only not in r["arm"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for i, rec in enumerate(pool.map(run_classical, jobs, chunksize=4), 1):
                records.append(rec)
                if i % (4 * args.datasets) == 0 or i == len(jobs):
                    print(
                        f"[{i}/{len(jobs)}] {rec['scenario']} | {rec['arm']} (elapsed {time.time() - t0:.0f}s)",
                        flush=True,
                    )
        (args.out / "classical.json").write_text(json.dumps(records))
    elif args.part == "pinn":
        jobs = [
            {"dataset": d, "member": m, "steps": args.steps}
            for d in range(args.pinn_datasets)
            for m in range(args.members)
        ]
        runs = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for rec in pool.map(run_pinn, jobs):
                runs.append(rec)
                print(
                    f"pinn data set {rec['dataset']} member {rec['member']}: end {rec['end'] / R_END:.3f} "
                    f"t90 {rec['t90'] / (0.9 * R_END):.3f} rel L2 {rec['rel_l2']:.3f} chi2 {rec['chi2']:.0f} "
                    f"{rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                    flush=True,
                )
                (args.out / "pinn.json").write_text(json.dumps(runs))
    summary = {}
    if (args.out / "classical.json").exists():
        summary["classical"] = summarize_classical(
            json.loads((args.out / "classical.json").read_text())
        )
    if (args.out / "pinn.json").exists() and "classical" in summary:
        b6 = summary["classical"]["known | batch-6"]
        truths = target_truths("known")
        fisher_std = {k: b6[k]["std_rel"] * truths[k] for k in ("end", "t90")}
        summary["pinn"] = summarize_pinn(
            json.loads((args.out / "pinn.json").read_text()), fisher_std
        )
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    for key, e in summary.get("classical", {}).items():
        cells = [
            f"{name}: cov {e[name]['coverage']:.2f} width {e[name]['width_rel']:.3f} bias {e[name]['bias_rel']:+.3f}"
            for name in ("end", "t90", "ttc")
            if name in e
        ]
        print(f"{key:<36} " + " | ".join(cells))
    if "pinn" in summary:
        print(json.dumps(summary["pinn"], indent=1))
    print("saved", args.out / "summary.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
