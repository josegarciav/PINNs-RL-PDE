"""J07d: the free-form closure by shooting, the classical control for J07b.

The same records as J07 and J07b (mass and temperature every second, 3 seeds).
The closure f(X) is a monotone piecewise-linear function on ``--knots`` knots
uniform in X over [X_eq, X_0], written as non-negative increments (so it
increases) and capped at 1, with no knowledge of the plant's functional form;
the droplet ODE is solved by the reference solver for each candidate and the
increments are fitted by bounded least squares on the weighted record
residuals (scipy ``least_squares``, finite-difference Jacobian). Two variants:
``anchored`` (f(X_eq) = 0, which the definition of the equilibrium moisture
gives) and ``free`` (the value at X_eq is a parameter too). Two starts: a
straight line from 0 at X_eq to 1 at X_0 (a cold start, far from the plant's
curve) and the pointwise estimate of J07b. Metrics as in J07b: the relative L2
error of f over [X_eq, X_0], the record misfit under the reference solver
(``DryingDroplet.record_misfit``), the run time.

Usage: python experiments/J07_droplet_drying/run_shooting.py [--seeds 3] [--knots 24]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run as J  # noqa: E402


def closure_from(theta, knots, anchored):
    """The piecewise-linear closure with increments ``theta`` (first value 0 when anchored)."""
    values = np.cumsum(theta) if not anchored else np.concatenate([[0.0], np.cumsum(theta)])
    values = np.minimum(values, 1.0)
    return lambda X: np.interp(X, knots, values)


def fit(dr, times, m_rec, T_rec, knots, anchored, start):
    sigma_m, sigma_T = J.NOISE_M_REL * dr.m0, J.NOISE_T

    def resid(theta):
        model = replace(dr, rate_closure=closure_from(theta, knots, anchored))
        with np.errstate(invalid="ignore", over="ignore"):
            ref = model.reference(J.T_MAX, nt=2001)
        r = np.concatenate(
            [
                (np.interp(times, ref.t, ref.m) - m_rec) / sigma_m,
                (np.interp(times, ref.t, ref.T) - T_rec) / sigma_T,
            ]
        )
        return np.where(np.isfinite(r), r, 1e3)

    values = np.clip(start(knots), 0.0, 1.0)
    values = np.maximum.accumulate(values)
    if anchored:
        theta0 = np.diff(np.concatenate([[0.0], values[1:]]))
    else:
        theta0 = np.diff(np.concatenate([[0.0], values]))
    theta0 = np.clip(theta0, 1e-4, 1.0)
    tic = time.perf_counter()
    res = least_squares(
        resid, theta0, bounds=(0.0, 1.0), method="trf", diff_step=1e-3, x_scale=0.05, max_nfev=200
    )
    return closure_from(res.x, knots, anchored), {
        "nfev": int(res.nfev),
        "time_s": time.perf_counter() - tic,
        "chi2_fit": float(2 * res.cost / (2 * times.size)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--knots", type=int, default=24)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dr = J.plant()
    knots = np.linspace(dr.X_eq, dr.X0, args.knots)
    sigma_m, sigma_T = J.NOISE_M_REL * dr.m0, J.NOISE_T
    runs = []
    for seed in range(args.seeds):
        times, m_rec, T_rec, _ = J.records(seed)
        X_pts, f_pts = dr.closure_from_record(times, m_rec, T_rec, window=21, order=3)
        order = np.argsort(X_pts)
        starts = {
            "cold": lambda X: (X - dr.X_eq) / (dr.X0 - dr.X_eq),
            "pointwise": lambda X, xp=X_pts[order], fp=f_pts[order]: np.interp(X, xp, fp),
        }
        for anchored in (True, False):
            for start_name, start in starts.items():
                f_hat, info = fit(dr, times, m_rec, T_rec, knots, anchored, start)
                err, grid, values = J.closure_error(dr, f_hat)
                chi = dr.record_misfit(times, m_rec, T_rec, sigma_m, sigma_T, closure=f_hat)
                rec = {
                    "arm": "anchored" if anchored else "free",
                    "start": start_name,
                    "seed": seed,
                    "knots": args.knots,
                    "closure_rel_l2": err,
                    "misfit": chi,
                    **info,
                    "X_grid": grid,
                    "f_hat": values,
                }
                runs.append(rec)
                print(
                    f"{rec['arm']:<9} start={start_name:<10} seed={seed} f rel L2 {err:.4f} "
                    f"misfit m {chi['mass']:.2f} T {chi['temperature']:.2f} nfev {info['nfev']} {info['time_s']:.1f}s",
                    flush=True,
                )
    (args.out / "shooting.json").write_text(json.dumps(runs, indent=1))
    print("saved", args.out / "shooting.json")


if __name__ == "__main__":
    main()
