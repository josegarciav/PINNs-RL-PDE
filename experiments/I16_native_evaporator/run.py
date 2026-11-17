"""I16: the evaporator's reference solve in C++ against SciPy, per solve, per fit and per filter run.

Three measurements on the falling-film evaporator (a one-hour window, a steam record with 10 s
knots and 0.05 K noise, a fouling ramp), one thread:

1. One solve of ``FallingFilmEvaporator.reference`` at 11, 51 and 201 cells: SciPy's BDF with a
   dense Jacobian (the path before I16), BDF with the banded pattern declared, and the compiled
   SDIRK integrator, at two tolerances.
2. A J06-style batch fit: ``scipy.optimize.least_squares`` over four Legendre modes of the fouling
   and a steam offset, each residual evaluation one solve at 51 cells, through either path.
3. The fouling monitor (an extended Kalman filter with the trend model at 51 cells over 360
   records) with ``integrator="scipy"`` and ``"native"``.

Usage: python experiments/I16_native_evaporator/run.py [--repeats 3]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from pinnrl.process import FallingFilmEvaporator, monitor_fouling

HERE = Path(__file__).parent
T_MAX = 3600.0
C_SCALE = 1e-4


def steam_record(seed=0):
    tt = np.arange(0.0, T_MAX + 1.0, 10.0)
    values = (
        355.15
        + 2.0 * np.sin(tt / 600.0)
        + 0.05 * np.random.default_rng(seed).standard_normal(tt.size)
    )
    return lambda t: float(np.interp(t, tt, values))


def model(c, bias=0.0, steam=None):
    steam = steam or steam_record()
    return FallingFilmEvaporator(
        T_s=lambda t: steam(t) + bias,
        R_f=lambda t: C_SCALE * float(legendre.legval(2.0 * t / T_MAX - 1.0, c)),
    )


def timed(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        tic = time.perf_counter()
        out = fn()
        best = min(best, time.perf_counter() - tic)
    return best, out


def _dense_bdf(m, nz, rtol, atol):
    """The reference solve as it was before I16: SciPy's BDF with a dense Jacobian."""
    ss = m.steady_state(nz, t=0.0)
    s0 = np.concatenate([ss["W"], ss["X"]])
    t = np.linspace(0.0, T_MAX, 61)
    sol = solve_ivp(m.mol_rhs(nz), (0.0, T_MAX), s0, t_eval=t, method="BDF", rtol=rtol, atol=atol)
    assert sol.success
    return sol.y[nz:]


def per_solve(repeats):
    rows = []
    smooth = lambda t: 355.15 + 2.0 * np.sin(t / 600.0)  # noqa: E731
    for inputs, steam in (("smooth steam", smooth), ("steam record, 10 s knots", steam_record())):
        m = model(np.array([1.0, 0.5]), steam=steam)
        for nz in (11, 51, 201):
            ref = m.reference(T_MAX, nz=nz, nt=61, rtol=1e-11, atol=1e-14)
            cases = [
                ("BDF dense, 1e-8", lambda nz=nz, m=m: _dense_bdf(m, nz, 1e-8, 1e-11)),
                ("BDF dense, 1e-6", lambda nz=nz, m=m: _dense_bdf(m, nz, 1e-6, 1e-9)),
                (
                    "native, 1e-8",
                    lambda nz=nz, m=m: m.reference(T_MAX, nz=nz, nt=61, method="native").X.u,
                ),
                (
                    "native, 1e-6",
                    lambda nz=nz, m=m: (
                        m.reference(T_MAX, nz=nz, nt=61, method="native", rtol=1e-6, atol=1e-9).X.u
                    ),
                ),
            ]
            if nz > 64:
                cases.insert(
                    2, ("BDF banded, 1e-8", lambda nz=nz, m=m: m.reference(T_MAX, nz=nz, nt=61).X.u)
                )
            for label, fn in cases:
                seconds, X = timed(fn, repeats)
                err = float(np.abs(X[1:] - ref.X.u[1:]).max() / np.abs(ref.X.u).max())
                rows.append(
                    {
                        "part": "solve",
                        "inputs": inputs,
                        "nz": nz,
                        "method": label,
                        "ms": 1e3 * seconds,
                        "rel_err_X": err,
                    }
                )
                print(
                    f"solve [{inputs}] nz={nz:<4} {label:<18} {1e3 * seconds:8.1f} ms  X error vs tight BDF {err:.1e}",
                    flush=True,
                )
    return rows


def batch_fit(repeats):
    """Four Legendre modes of the fouling and a steam offset from hourly outlet records."""
    nz, modes = 51, 4
    truth_c = np.array([0.8, 0.6, -0.1, 0.05])
    truth = model(truth_c, bias=0.4).reference(T_MAX, nz=nz, nt=361)
    rng = np.random.default_rng(1)
    y = np.concatenate(
        [
            truth.X.u[-1, 1:] + 0.002 * rng.standard_normal(360),
            truth.W.u[-1, 1:] + 0.001 * rng.standard_normal(360),
        ]
    )
    sig = np.concatenate([np.full(360, 0.002), np.full(360, 0.001)])
    rows = []
    for method, kw in (
        ("BDF", dict(method="BDF")),
        ("native", dict(method="native", rtol=1e-6, atol=1e-9)),
    ):
        calls = [0]

        def resid(theta, calls=calls, kw=kw):
            calls[0] += 1
            ref = model(theta[:modes], bias=theta[modes]).reference(T_MAX, nz=nz, nt=361, **kw)
            r = (np.concatenate([ref.X.u[-1, 1:], ref.W.u[-1, 1:]]) - y) / sig
            return np.append(r, theta[modes] / 0.5)

        x0 = np.zeros(modes + 1)
        x0[0] = 1.0
        seconds, res = timed(
            lambda resid=resid, x0=x0: least_squares(
                resid, x0, method="trf", diff_step=0.02, xtol=1e-7, ftol=1e-10
            ),
            repeats,
        )
        rows.append(
            {
                "part": "fit",
                "method": method,
                "seconds": seconds,
                "solves": calls[0] // repeats,
                "estimate": res.x.tolist(),
                "cost": float(res.cost),
            }
        )
        print(
            f"fit {method:<7} {seconds:6.2f} s  ({calls[0] // repeats} solves)  modes {np.round(res.x[:modes], 3)} bias {res.x[modes]:.3f}  cost {res.cost:.1f}",
            flush=True,
        )
    print(f"    truth modes {truth_c} bias 0.4")
    return rows


def monitor(repeats):
    nz = 51
    truth = model(np.array([1.0, 1.0])).reference(T_MAX, nz=nz, nt=361)
    rng = np.random.default_rng(2)
    times = truth.t[1:]
    records = np.stack(
        [
            truth.X.u[-1, 1:] + 0.002 * rng.standard_normal(360),
            truth.W.u[-1, 1:] + 0.001 * rng.standard_normal(360),
        ],
        1,
    )
    m = model(np.array([0.0]))
    rows = []
    out = {}
    for integrator in ("scipy", "native"):
        seconds, res = timed(
            lambda integrator=integrator: monitor_fouling(
                m, times, records, nz=nz, steam_bias_std=0.5, integrator=integrator
            ),
            repeats,
        )
        out[integrator] = res
        rows.append(
            {
                "part": "monitor",
                "integrator": integrator,
                "seconds": seconds,
                "fouling_end": float(res.fouling[-1]),
                "misfit": [float(v) for v in res.misfit],
            }
        )
        print(
            f"monitor {integrator:<7} {seconds:6.2f} s  fouling at the end {res.fouling[-1]:.3e} (truth {2e-4:.3e})  misfit {np.round(res.misfit, 2)}",
            flush=True,
        )
    diff = float(np.abs(out["native"].fouling - out["scipy"].fouling).max())
    print(f"    max difference between the two fouling tracks {diff:.1e}")
    rows.append({"part": "monitor", "integrator": "difference", "max_abs_fouling_diff": diff})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = (
        per_solve(args.repeats)
        + batch_fit(max(1, args.repeats // 3))
        + monitor(max(1, args.repeats // 3))
    )
    (args.out / "timings.json").write_text(json.dumps(rows, indent=1))
    print("saved", args.out / "timings.json")


if __name__ == "__main__":
    main()
