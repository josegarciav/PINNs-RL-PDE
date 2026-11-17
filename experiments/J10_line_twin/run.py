"""J10: the line-level twin. A three-pass evaporator feeds the spray dryer; whose error does the powder inherit?

``pinnrl.process.ProcessLine`` chains three falling-film passes and hands the
concentrate to the co-current tower. Over a four-hour shift the passes foul
at different rates, so the concentrate thins and the powder leaves wetter.
The plant records each pass's outlet solids and flow and the dryer's air
thermocouples and outlet humidity every 30 s, and samples the powder hourly.

1. ``monitors``: J06's fouling monitor on each pass with the measured inlet
   of the previous pass as its input; the end-of-shift fouling bands,
   propagated with the dryer's campaign posterior (J11) and the feed's
   accuracy to the powder's moisture and stickiness margin: the variance
   budget by unit, linear and by Monte Carlo.
2. ``joint``: the linear-Gaussian posterior of twelve unknowns (the fouling
   level and growth of each pass, the drying curve's two parameters, the wall
   loss and the droplet size, the feed's flow and solids) from the
   evaporator's records, the dryer's, the lab's and all together: what each
   unit's instruments determine of the other unit's unknowns, and of the
   powder at the end of the shift.
3. ``validate``: simulated shifts fitted jointly through the line's reference
   (bounded least squares with the priors as rows) against the two-stage
   estimate (monitors, then propagation): the error and the interval of the
   predicted powder moisture, and the cost.

Usage: python experiments/J10_line_twin/run.py [--arms monitors,joint,validate] [--datasets 10] [--workers 3]
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
T_SHIFT = 4 * 3600.0
RECORD_EVERY = 30.0
TIMES = np.arange(RECORD_EVERY, T_SHIFT + 1e-9, RECORD_EVERY)
LAB_TIMES = np.arange(3600.0, T_SHIFT + 1e-9, 3600.0)
FOULING_END = (0.2e-4, 0.4e-4, 0.6e-4)  # m²K/W reached by each pass at the end of the shift
NOISE = {"X_pass": 0.002, "W_pass": 0.001, "T_air": 0.5, "Y_out": 1e-3, "X_powder": 0.003}
PROBES = (1.0, 2.0, 3.5, 5.0, 6.5, 8.0)
DRYER_PRIOR = {
    "X_cr": 0.106,
    "n": 0.064,
    "U_wall": 0.30,
    "d0": 25e-6,
}  # J11: five points, nuisances uncertain
FEED_PRIOR = {"W_feed": 0.002, "X_feed": 0.005}  # J12's instrument accuracies
NAMES = ["a_0", "a_1", "a_2", "b_0", "b_1", "b_2", "X_cr", "n", "U_wall", "d0", "W_feed", "X_feed"]
TRUTH = np.array([0.0, 0.0, 0.0, *FOULING_END, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
STEPS = np.array([1e-5] * 6 + [0.02, 0.02, 0.05, 2.5e-6, 5e-4, 1e-3])
PRIOR_STD = np.array(
    [np.inf] * 6
    + [DRYER_PRIOR[k] for k in ("X_cr", "n", "U_wall", "d0")]
    + [FEED_PRIOR["W_feed"], FEED_PRIOR["X_feed"]]
)
DRYER_EVERY = 4  # the quasi-steady dryer every 2 min in the reference runs, interpolated between


def make_line(theta=TRUTH):
    """The line for the parameter vector: fouling ``a_k + b_k t/T`` per pass, the dryer's and the feed's deviations."""
    from pinnrl.process import ProcessLine

    th = np.asarray(theta, dtype=float)
    fouling = [lambda t, a=th[k], b=th[3 + k]: a + b * t / T_SHIFT for k in range(3)]
    line = ProcessLine.default(fouling=fouling)
    passes = list(line.passes)
    passes[0] = replace(passes[0], W_in=0.1 + th[10], X_in=0.1 + th[11])
    droplet = line.dryer.droplet
    droplet = replace(
        droplet, X_cr=droplet.X_cr + th[6], n=droplet.n + th[7], d0=droplet.d0 + th[9]
    )
    dryer = replace(line.dryer, droplet=droplet, U_wall=line.dryer.U_wall + th[8])
    return replace(line, passes=passes, dryer=dryer)


def record_vector(theta=TRUTH, nz=51):
    """All records of a shift as one vector: passes (X then W), dryer thermocouples, humidity, lab samples."""
    line = make_line(theta)
    ref = line.reference(
        T_SHIFT, nz=nz, nt=TIMES.size + 1, dryer_every=DRYER_EVERY, probe_heights=PROBES
    )
    rec = line.records(ref)
    k = np.searchsorted(ref.t, TIMES)
    lab = np.searchsorted(ref.t, LAB_TIMES)
    parts = [
        rec["X_pass"][:, k].ravel(),
        rec["W_pass"][:, k].ravel(),
        rec["T_air"][:, k].ravel(),
        rec["Y_out"][k],
        rec["X_powder"][lab],
    ]
    return np.concatenate(parts), ref


def sigma_vector():
    n = TIMES.size
    return np.concatenate(
        [
            np.full(3 * n, NOISE["X_pass"]),
            np.full(3 * n, NOISE["W_pass"]),
            np.full(len(PROBES) * n, NOISE["T_air"]),
            np.full(n, NOISE["Y_out"]),
            np.full(LAB_TIMES.size, NOISE["X_powder"]),
        ]
    )


def blocks():
    """Row ranges of the record vector by instrument group."""
    n = TIMES.size
    e = 6 * n
    d = e + len(PROBES) * n + n
    return {"evaporator": slice(0, e), "dryer": slice(e, d), "lab": slice(d, d + LAB_TIMES.size)}


def dataset(seed, draw=True):
    """Noisy records of a plant whose nuisances (dryer parameters, feed) are drawn from their priors.

    Returns ``(records, clean records, reference, truth)``; the fouling is the
    same in every data set, the nuisances differ (as J11's validation did), so
    a fit that starts at the nominal values is tested on what it cannot see.
    ``draw=False`` gives the nominal plant (the monitors arm and the figure).
    """
    rng = np.random.default_rng(500 + seed)
    truth = TRUTH.copy()
    finite = np.isfinite(PRIOR_STD)
    if draw:
        truth[finite] += PRIOR_STD[finite] * rng.standard_normal(int(finite.sum()))
        truth[9] = max(truth[9], -100e-6)  # the droplet stays larger than 150 µm
    clean, ref = record_vector(truth)
    return clean + sigma_vector() * rng.standard_normal(clean.size), clean, ref, truth


def split(y):
    """The record vector back into named arrays."""
    n = TIMES.size
    out = {"X_pass": y[: 3 * n].reshape(3, n), "W_pass": y[3 * n : 6 * n].reshape(3, n)}
    k = 6 * n
    out["T_air"] = y[k : k + len(PROBES) * n].reshape(len(PROBES), n)
    k += len(PROBES) * n
    out["Y_out"] = y[k : k + n]
    out["X_powder"] = y[k + n :]
    return out


# --------------------------------------------------------------------- monitors
def smooth(x, window=11):
    """A centred running mean over ``window`` records, the ends padded with the edge values."""
    pad = window // 2
    xp = np.concatenate([np.full(pad, x[0]), x, np.full(pad, x[-1])])
    return np.convolve(xp, np.ones(window) / window, mode="valid")


def run_monitors(y, level=0.9):
    """J06's fouling monitor on each pass, the measured inlet of the previous pass as its input."""
    from scipy.stats import norm

    from pinnrl.process import monitor_fouling

    rec = split(y)
    line = make_line()
    W_in, X_in = line.passes[0].W_in, line.passes[0].X_in  # the feed instruments are trusted here
    out = []
    z = norm.ppf(0.5 + level / 2)
    for k, tube in enumerate(line.passes):
        model = replace(tube, W_in=W_in, X_in=X_in, R_f=0.0)
        tic = time.perf_counter()
        mon = monitor_fouling(
            model,
            TIMES,
            np.stack([rec["X_pass"][k], rec["W_pass"][k]], 1),
            nz=51,
            noise=(NOISE["X_pass"], NOISE["W_pass"]),
            level=level,
        )
        out.append(
            {
                "pass": k,
                "seconds": time.perf_counter() - tic,
                "end": float(mon.fouling[-1]),
                "end_std": float((mon.upper[-1] - mon.lower[-1]) / (2 * z)),
                "smoothed_end": float(mon.smoothed[-1]),
                "truth_end": FOULING_END[k],
                "misfit": list(mon.misfit),
                "track": mon.fouling.tolist(),
                "lower": mon.lower.tolist(),
                "upper": mon.upper.tolist(),
            }
        )
        # the next pass is fed with what this pass's instruments recorded, smoothed over 5 min
        # (the fouling moves over hours; the raw record's noise would enter the filter as an exact input)
        w_smooth, x_smooth = smooth(rec["W_pass"][k]), smooth(rec["X_pass"][k])
        W_in = lambda t, w=w_smooth: float(np.interp(t, TIMES, w))  # noqa: E731
        X_in = lambda t, x=x_smooth: float(np.interp(t, TIMES, x))  # noqa: E731
    return out


def run_propagation(monitors, samples=300):
    """The end-of-shift budget: the monitors' fouling bands, the dryer's campaign posterior, the feed's accuracy."""
    line = make_line()
    fouling_std = {f"R_f_{m['pass']}": m["end_std"] for m in monitors}
    cases = {
        "monitored fouling + dryer + feed": {**fouling_std, **DRYER_PRIOR, **FEED_PRIOR},
        "monitored fouling only": fouling_std,
        "dryer + feed only": {**DRYER_PRIOR, **FEED_PRIOR},
        "unmonitored fouling + dryer + feed": {
            **{f"R_f_{k}": FOULING_END[k] for k in range(3)},
            **DRYER_PRIOR,
            **FEED_PRIOR,
        },
    }
    out = {}
    for label, std in cases.items():
        tic = time.perf_counter()
        prop = line.propagate(std, t=T_SHIFT, samples=samples if "only" not in label else 0)
        prop["seconds"] = time.perf_counter() - tic
        out[label] = prop
        print(
            f"propagate [{label}]: powder std {prop['linear']['powder']['std']:.4f}, "
            f"margin std {prop['linear']['margin']['std']:.2f} K, "
            f"shares {dict((k, round(v, 2)) for k, v in prop['linear']['powder']['shares'].items() if v > 0.03)}",
            flush=True,
        )
    return out


# ------------------------------------------------------------------------ joint
def jacobian():
    """Central differences of the record vector and of the end-of-shift targets at the truth."""
    cols, powder_g, margin_g = [], [], []
    for j in range(len(NAMES)):
        up, dn = TRUTH.copy(), TRUTH.copy()
        up[j] += STEPS[j]
        dn[j] -= STEPS[j]
        yu, ru = record_vector(up)
        yd, rd = record_vector(dn)
        cols.append((yu - yd) / (2 * STEPS[j]))
        powder_g.append((ru.powder[-1] - rd.powder[-1]) / (2 * STEPS[j]))
        margin_g.append((ru.margin[-1] - rd.margin[-1]) / (2 * STEPS[j]))
        print(f"jacobian column {NAMES[j]} done", flush=True)
    return np.stack(cols, 1), np.array(powder_g), np.array(margin_g)


def run_joint():
    from pinnrl.analysis import posterior_covariance, target_std

    tic = time.perf_counter()
    J, g_powder, g_margin = jacobian()
    sig = sigma_vector()
    Jw = J / sig[:, None]
    B = blocks()
    info = {k: Jw[s].T @ Jw[s] for k, s in B.items()}
    combos = {
        "prior only": [],
        "evaporator": ["evaporator"],
        "dryer online": ["dryer"],
        "dryer online + lab": ["dryer", "lab"],
        "evaporator + dryer online": ["evaporator", "dryer"],
        "all": ["evaporator", "dryer", "lab"],
    }
    end_vectors = [
        np.eye(12)[k] + np.eye(12)[3 + k] for k in range(3)
    ]  # fouling at the end of pass k
    out = {
        "seconds": time.perf_counter() - tic,
        "columns": NAMES,
        "powder_gradient": g_powder.tolist(),
    }
    for label, parts in combos.items():
        F = sum((info[p] for p in parts), np.zeros((12, 12)))
        try:
            cov = posterior_covariance(F, PRIOR_STD)
        except np.linalg.LinAlgError:
            out[label] = None
            continue
        std = np.sqrt(np.diag(cov))
        out[label] = {
            "std": dict(zip(NAMES, std.tolist(), strict=True)),
            "fouling_end_std": [float(target_std(cov, v)) for v in end_vectors],
            "powder_end_std": float(target_std(cov, g_powder)),
            "margin_end_std": float(target_std(cov, g_margin)),
        }
        print(
            f"joint [{label}]: fouling end std {[f'{s:.1e}' for s in out[label]['fouling_end_std']]}, "
            f"powder end std {out[label]['powder_end_std']:.4f}, margin {out[label]['margin_end_std']:.2f} K",
            flush=True,
        )
    return out


# --------------------------------------------------------------------- validate
def run_fit(seed):
    """The joint MAP fit of the twelve unknowns through the line's reference, from a cold start."""
    from scipy.optimize import least_squares

    y, _clean, ref_true, truth = dataset(seed)
    sig = sigma_vector()
    finite = np.isfinite(PRIOR_STD)
    calls = {"n": 0}

    def residual(theta):
        calls["n"] += 1
        r = (record_vector(theta)[0] - y) / sig
        prior = (theta[finite] - TRUTH[finite] * 0.0) / PRIOR_STD[
            finite
        ]  # priors centred on the nominal
        return np.concatenate([r, prior])

    x0 = np.zeros(12)
    scale = np.array([2e-5] * 6 + [0.05, 0.05, 0.1, 1e-5, 1e-3, 2e-3])
    lower = np.array([-1e-4] * 3 + [-1e-4] * 3 + [-0.5, -0.8, -0.9, -1e-4, -0.02, -0.03])
    upper = np.array([5e-4] * 6 + [0.4, 1.5, 1.5, 1e-4, 0.02, 0.03])
    tic = time.perf_counter()
    res = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        x_scale=scale,
        diff_step=1e-2,
        xtol=1e-8,
        ftol=1e-9,
        max_nfev=400,
    )
    seconds = time.perf_counter() - tic
    est = res.x
    powder = record_vector(est)[1].powder[-1]  # the plant's own reference: no discretisation bias
    return {
        "dataset": seed,
        "seconds": seconds,
        "solves": calls["n"],
        "success": bool(res.success),
        "estimate": dict(zip(NAMES, est.tolist(), strict=True)),
        "truth": dict(zip(NAMES, truth.tolist(), strict=True)),
        "fouling_end": [float(est[k] + est[3 + k]) for k in range(3)],
        "powder_end": float(powder),
        "powder_truth": float(ref_true.powder[-1]),
        "chi2_reduced": float(np.mean(res.fun[: y.size] ** 2)),
    }


def run_two_stage(seed):
    """Monitors on the passes, then the line run forward at their estimates with the nominal dryer."""
    y, _clean, ref_true, truth = dataset(seed)
    tic = time.perf_counter()
    mons = run_monitors(y)
    theta = TRUTH.copy()  # the nominal dryer and feed: this estimate knows nothing of the nuisances
    theta[:3] = 0.0
    theta[3:6] = [m["end"] for m in mons]  # the fouling at the end, as the monitors have it
    line = make_line(theta)
    powder = record_vector(theta)[1].powder[-1]
    prop = line.propagate(
        {**{f"R_f_{m['pass']}": m["end_std"] for m in mons}, **DRYER_PRIOR, **FEED_PRIOR}, t=T_SHIFT
    )
    return {
        "dataset": seed,
        "seconds": time.perf_counter() - tic,
        "truth": dict(zip(NAMES, truth.tolist(), strict=True)),
        "fouling_end": [m["end"] for m in mons],
        "fouling_end_std": [m["end_std"] for m in mons],
        "misfit": [m["misfit"] for m in mons],
        "powder_end": float(powder),
        "powder_std": prop["linear"]["powder"]["std"],
        "powder_truth": float(ref_true.powder[-1]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default="monitors,joint,validate")
    parser.add_argument("--datasets", type=int, default=10)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    arms = args.arms.split(",")
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "j10.json"
    results = json.loads(path.read_text()) if path.exists() else {}

    tic = time.perf_counter()
    y0, clean, ref, truth0 = dataset(
        0, draw=False
    )  # the nominal plant for the monitors and the figure
    results["setup"] = {
        "line_solve_s": time.perf_counter() - tic,
        "t": ref.t.tolist(),
        "X_concentrate": ref.X.tolist(),
        "powder": ref.powder.tolist(),
        "margin": ref.margin.tolist(),
        "X_pass": [r.X.u[-1].tolist() for r in ref.passes],
        "steady_start": make_line().steady(t=0.0) | {"dryer": None},
        "steady_end": make_line().steady(t=T_SHIFT) | {"dryer": None},
        "records_noisy": {k: np.asarray(v).tolist() for k, v in split(y0).items()},
        "truth_0": dict(zip(NAMES, truth0.tolist(), strict=True)),
    }
    print(
        f"line: concentrate {ref.X[0]:.3f} -> {ref.X[-1]:.3f}, powder {ref.powder[0]:.3f} -> {ref.powder[-1]:.3f}, "
        f"margin {ref.margin[0]:.1f} -> {ref.margin[-1]:.1f} K; one shift solve {results['setup']['line_solve_s']:.1f} s",
        flush=True,
    )
    if "monitors" in arms:
        results["monitors"] = run_monitors(y0)
        for m in results["monitors"]:
            print(
                f"monitor pass {m['pass']}: end {m['end']:.2e} ± {m['end_std']:.1e} (truth {m['truth_end']:.1e}), "
                f"misfit {m['misfit']}, {m['seconds']:.1f} s",
                flush=True,
            )
        results["propagation"] = run_propagation(results["monitors"])
        path.write_text(json.dumps(results, indent=1))
    if "joint" in arms:
        results["joint"] = run_joint()
        path.write_text(json.dumps(results, indent=1))
    if "validate" in arms:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fits = list(pool.map(run_fit, range(args.datasets)))
            two = list(pool.map(run_two_stage, range(args.datasets)))
        results["fits"] = fits
        results["two_stage"] = two
        err_j = np.array([f["powder_end"] - f["powder_truth"] for f in fits])
        err_t = np.array([f["powder_end"] - f["powder_truth"] for f in two])
        fe_j = np.array([f["fouling_end"] for f in fits]) - np.array(FOULING_END)
        fe_t = np.array([f["fouling_end"] for f in two]) - np.array(FOULING_END)
        print(
            f"validate: fouling end rms error, joint {np.sqrt(np.mean(fe_j**2, axis=0))}, "
            f"two-stage {np.sqrt(np.mean(fe_t**2, axis=0))}",
            flush=True,
        )
        print(
            f"validate: joint powder error rms {np.sqrt(np.mean(err_j**2)):.4f} ({np.mean([f['seconds'] for f in fits]):.0f} s a fit), "
            f"two-stage rms {np.sqrt(np.mean(err_t**2)):.4f} ({np.mean([f['seconds'] for f in two]):.0f} s)",
            flush=True,
        )
        path.write_text(json.dumps(results, indent=1))
    print("saved", path)


if __name__ == "__main__":
    main()
