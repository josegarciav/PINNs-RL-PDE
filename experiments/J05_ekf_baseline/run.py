"""J05: an extended Kalman filter on a lumped evaporator model against the PINN twin of J03.

The same plant and records as J03 (the J01 evaporator over 20 minutes, the
steam wandering +-1 K and recorded, the fouling drifting 0 -> 2e-4 m2K/W,
outlet X and W every 5 s with noise 0.002 and 0.001, the same noise seeds).
The estimator is `pinnrl.process.ExtendedKalmanFilter` on the method-of-lines
model with n_z in {6, 11, 21, 51} cells, R_f appended to the state as a
random walk of intensity sigma_rw^2 per second, sigma_rw in {1e-7, 1e-6, 1e-5}
(the truth's slope is 1.7e-7 per second), started at R_f = 1e-4 like the PINN;
the Rauch-Tung-Striebel smoother gives the batch estimate the PINN is
comparable to. Model mismatch: the filter's steam record biased by +0.3 K, its
feed flow by +1 % (J04's nuisances). Metrics: relative L2 of R_f(t) after a
2-minute warm-up, the error at the end of the window, the lag of the filter
behind the ramp, the coverage of the +-2 sigma band, the run time.

Usage: python experiments/J05_ekf_baseline/run.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
T_MAX = 1200.0
R_END = 2e-4
EVERY = 5.0
NOISE_X, NOISE_W = 0.002, 0.001
GRIDS = (6, 11, 21, 51)
SIGMAS = (1e-7, 1e-6, 1e-5)
MISMATCH = {"steam +0.3 K": {"T_s_bias": 0.3}, "feed +1 %": {"W_in_bias": 0.01}}


def steam(t):
    return 355.15 + 1.0 * math.sin(2 * math.pi * t / 400.0)


def plant_records(seed):
    from pinnrl.process import FallingFilmEvaporator

    ev = FallingFilmEvaporator(T_s=steam, R_f=lambda t: R_END * t / T_MAX)
    ref = ev.reference(t_max=T_MAX, nz=201, nt=241)
    times = np.arange(EVERY, T_MAX + 1e-9, EVERY)
    X = ref.X.interpolate(np.full_like(times, ev.length), times)
    W = ref.W.interpolate(np.full_like(times, ev.length), times)
    rx = np.random.default_rng(seed)
    rw = np.random.default_rng(seed + 1000)
    return times, np.stack(
        [
            X + NOISE_X * rx.standard_normal(times.size),
            W + NOISE_W * rw.standard_normal(times.size),
        ],
        axis=1,
    )


def run_filter(times, records, nz, sigma_rw, mismatch=None):
    from pinnrl.process import ExtendedKalmanFilter, FallingFilmEvaporator, augment_random_walk

    mismatch = dict(mismatch or {})
    b = mismatch.get("T_s_bias", 0.0)
    model = FallingFilmEvaporator(
        T_s=lambda t: steam(t) + b, W_in=0.1 * (1.0 + mismatch.get("W_in_bias", 0.0)), R_f=0.0
    )
    f, h, s0 = model.fouling_state_space(nz)
    n = s0.size
    ekf = ExtendedKalmanFilter(
        augment_random_walk(f, n, 1),
        lambda t, x: h(t, x[:n]),
        Q=np.concatenate([np.full(n, 1e-12), [sigma_rw**2]]),
        R=np.array([NOISE_X**2, NOISE_W**2]),
    )
    x0 = np.concatenate([s0, [1e-4]])
    P0 = np.diag(np.concatenate([np.full(n, 1e-6), [(1e-4) ** 2]]))
    res = ekf.run(times, records, x0, P0, smooth=True, lower=np.r_[np.full(n, -np.inf), 0.0])
    truth = R_END * times / T_MAX
    sel = times >= 120.0
    slope = R_END / T_MAX

    def metrics(est, std):
        err = est - truth
        band = np.abs(err[sel]) <= 2 * std[sel]
        return {
            "rel_l2": float(np.linalg.norm(err[sel]) / np.linalg.norm(truth[sel])),
            "end_error": float(abs(err[-1]) / truth[-1]),
            "lag_s": float(np.mean(-err[times >= 300.0]) / slope),
            "coverage_2std": float(band.mean()),
            "std_end": float(std[-1]),
        }

    out = {
        "nz": nz,
        "sigma_rw": sigma_rw,
        "mismatch": mismatch,
        "time_s": res.time_s,
        "filter": metrics(res.filtered[:, -1], res.std()[:, -1]),
        "smoother": metrics(res.smoothed[:, -1], res.std(smoothed=True)[:, -1]),
        "profile_filter": res.filtered[:, -1].tolist(),
        "profile_smoother": res.smoothed[:, -1].tolist(),
        "std_filter": res.std()[:, -1].tolist(),
        "t": times.tolist(),
    }
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records_by_seed = {s: plant_records(s) for s in range(args.seeds)}
    runs = []
    t0 = time.time()
    for nz in GRIDS:
        for sigma in SIGMAS:
            for seed, (times, records) in records_by_seed.items():
                r = run_filter(times, records, nz, sigma)
                r["seed"] = seed
                runs.append(r)
                print(
                    f"nz={nz:<3} sigma_rw={sigma:.0e} seed={seed} filter relL2 {r['filter']['rel_l2']:.3f} end {r['filter']['end_error']:.3f} "
                    f"lag {r['filter']['lag_s']:.0f}s cov {r['filter']['coverage_2std']:.2f} | smoother relL2 {r['smoother']['rel_l2']:.3f} "
                    f"end {r['smoother']['end_error']:.3f} | {r['time_s']:.1f}s (elapsed {time.time() - t0:.0f}s)",
                    flush=True,
                )
    for label, mm in MISMATCH.items():
        for seed, (times, records) in records_by_seed.items():
            r = run_filter(times, records, 11, 1e-6, mm)
            r["seed"] = seed
            r["label"] = label
            runs.append(r)
            print(
                f"mismatch {label:<12} seed={seed} filter relL2 {r['filter']['rel_l2']:.3f} end {r['filter']['end_error']:.3f} "
                f"| smoother relL2 {r['smoother']['rel_l2']:.3f} end {r['smoother']['end_error']:.3f}",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "runs.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
