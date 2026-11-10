"""J05b: moving-horizon estimation against the extended Kalman filter on the evaporator.

J05's plant, records and seeds (outlet X and W every 5 s for 20 minutes, the
fouling drifting 0 -> 2e-4 m2K/W, the steam wandering +-1 K and recorded). Both
estimators use the 11-cell method-of-lines model with R_f appended as a random
walk and the same Q, R and prior; `pinnrl.process.MovingHorizonEstimator`
re-solves the last ``horizon`` records at every second record (stride 2).

Cases:
  tuned        sigma_rw = 1e-6 (J05's best), horizons 6, 12 and 24 records;
  noisy        sigma_rw = 1e-5 (the filter follows the noise), horizon 12, with
               and without the constraint that the fouling only grows
               (disturbance >= 0);
  monotone     sigma_rw = 1e-6, horizon 12, the same constraint;
  bad start    R_f started at 4e-4 with a prior standard deviation of 2e-5 (an
               over-confident wrong prior), sigma_rw = 1e-6, horizon 12 and 24.
Metrics as J05: relative L2 of R_f(t) after a 2-minute warm-up, the error at
the end of the window, the coverage of the 90 % band, the run time.

Usage: python experiments/J05_ekf_baseline/run_mhe.py [--seeds 3] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run as J  # noqa: E402

NZ = 11
Z = 1.6448536269514722
STRIDE = 2

CASES = [
    # label, sigma_rw, horizon (0 = the filter), monotone, bad start
    ("tuned", 1e-6, 0, False, False),
    ("tuned", 1e-6, 6, False, False),
    ("tuned", 1e-6, 12, False, False),
    ("tuned", 1e-6, 24, False, False),
    ("monotone", 1e-6, 12, True, False),
    ("noisy", 1e-5, 0, False, False),
    ("noisy", 1e-5, 12, False, False),
    ("noisy", 1e-5, 12, True, False),
    ("bad start", 1e-6, 0, False, True),
    ("bad start", 1e-6, 12, False, True),
    ("bad start", 1e-6, 24, False, True),
]


def run_case(job):
    label, sigma_rw, horizon, monotone, bad_start, seed = job
    from pinnrl.process import (
        ExtendedKalmanFilter,
        FallingFilmEvaporator,
        MovingHorizonEstimator,
        augment_random_walk,
    )

    times, records = J.plant_records(seed)
    model = FallingFilmEvaporator(T_s=J.steam, R_f=0.0)
    f, h, s0 = model.fouling_state_space(NZ)
    n = s0.size
    f_aug = augment_random_walk(f, n, 1)

    def h_aug(t, x):
        return h(t, x[:n])

    # The physical states carry no process noise: inside a window they follow the model.
    Q = np.concatenate([np.zeros(n), [sigma_rw**2]])
    R = np.array([J.NOISE_X**2, J.NOISE_W**2])
    rf0, rf_std = (4e-4, 2e-5) if bad_start else (1e-4, 1e-4)
    x0 = np.concatenate([s0, [rf0]])
    P0 = np.diag(np.concatenate([np.full(n, 1e-6), [rf_std**2]]))
    lower = np.r_[np.full(n, -np.inf), 0.0]
    if horizon == 0:
        res = ExtendedKalmanFilter(f_aug, h_aug, Q, R).run(times, records, x0, P0, lower=lower)
        res_t, est, std = times, res.filtered[:, -1], res.std()[:, -1]
        evaluations = times.size
    else:
        mhe = MovingHorizonEstimator(f_aug, h_aug, Q, R, horizon=horizon)
        res = mhe.run(
            times,
            records,
            x0,
            P0,
            lower=lower,
            noise_lower=0.0 if monotone else None,
            stride=STRIDE,
        )
        res_t, est, std = res.t, res.filtered[:, -1], res.std()[:, -1]
        evaluations = mhe.evaluations
    truth = J.R_END * res_t / J.T_MAX
    sel = res_t >= 120.0
    err = est - truth
    return {
        "label": label,
        "sigma_rw": sigma_rw,
        "horizon": horizon,
        "monotone": monotone,
        "seed": seed,
        "rel_l2": float(np.linalg.norm(err[sel]) / np.linalg.norm(truth[sel])),
        "end_error": float(abs(err[-1]) / truth[-1]),
        "coverage_90": float(np.mean(np.abs(err[sel]) <= Z * std[sel])),
        "std_end_rel": float(std[-1] / truth[-1]),
        "roughness": float(np.std(np.diff(est[sel])) / (J.R_END / J.T_MAX * (res_t[1] - res_t[0]))),
        "recovery_s": float(res_t[np.argmax(np.abs(err) < 0.2e-4)])
        if np.any(np.abs(err) < 0.2e-4)
        else None,
        "time_s": res.time_s,
        "evaluations": int(evaluations),
        "t": res_t.tolist(),
        "profile": est.tolist(),
        "std": std.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(*case, seed) for case in CASES for seed in range(args.seeds)]
    runs = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for r in pool.map(run_case, jobs):
            runs.append(r)
            name = (
                "EKF"
                if r["horizon"] == 0
                else f"MHE N={r['horizon']}" + (" monotone" if r["monotone"] else "")
            )
            print(
                f"{r['label']:<10} {name:<18} seed={r['seed']} rel L2 {r['rel_l2']:.3f} end {r['end_error']:.3f} "
                f"cov90 {r['coverage_90']:.2f} rough {r['roughness']:.1f} recovery {r['recovery_s']} "
                f"{r['time_s']:.1f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "mhe.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "mhe.json")


if __name__ == "__main__":
    main()
