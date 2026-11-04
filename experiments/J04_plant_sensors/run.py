"""J04: is the fouling resistance identifiable from outlet measurements only?

The J01 evaporator over J03's 20-minute window (steam wandering +-1 K, feed
steady, fouling drifting 0 -> 2e-4 m2K/W), with the fouling written as six
Legendre modes of time, R_f(t) = sum_k c_k P_k(2t/T - 1) (truth: c_0 = c_1 =
1e-4, the linear drift). The sensitivities of the records to the modes and to
four nuisance parameters (the clean coefficient U, a steam-temperature bias,
a feed-flow bias and a feed-solids bias) come from central finite differences
on the reference solver, and `pinnrl.analysis.fisher_information` gives the
Cramer-Rao bound, the correlations and the condition number.

Part A, layouts: outlet X every 5 s (noise 0.002); outlet X and W (noise
0.001); plus a mid-tube X probe; outlet X and W every 1 s. Modes only (the
inputs exactly known).
Part B, nuisances: outlet X and W, the modes together with each nuisance in
turn and with all four.
Part C, profile likelihood (F07's method, on the reference): the misfit of 5
noisy data sets as a function of the drift's end value r_end (R_f = r_end t/T),
with the steam bias fixed at zero and re-optimised at each grid point, against
the Fisher parabola.

Usage: python experiments/J04_plant_sensors/run.py [--datasets 5] [--modes 6]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from numpy.polynomial import legendre
from scipy.optimize import minimize_scalar

HERE = Path(__file__).parent
T_MAX = 1200.0
R_END = 2e-4
C_SCALE = 1e-4  # the modes are reported in units of the mean fouling
NOISE_X, NOISE_W = 0.002, 0.001
NZ, NT = 201, 241
MID = 5.0

NUISANCES = {  # name: (step for the finite difference, unit for the report)
    "U": (0.01, "relative"),
    "T_s_bias": (0.1, "K"),
    "W_in_bias": (0.01, "relative"),
    "X_in_bias": (0.001, "absolute"),
}


def steam(t):
    return 355.15 + 1.0 * math.sin(2 * math.pi * t / 400.0)


def plant(c, nuis=None):
    """The evaporator with fouling modes ``c`` (in units of C_SCALE) and nuisance offsets."""
    from pinnrl.process import FallingFilmEvaporator

    nuis = dict(nuis or {})
    c = np.asarray(c, dtype=float)

    def rf(t):
        return C_SCALE * float(legendre.legval(2.0 * t / T_MAX - 1.0, c))

    b = nuis.get("T_s_bias", 0.0)
    return FallingFilmEvaporator(
        U=2000.0 * (1.0 + nuis.get("U", 0.0)),
        W_in=0.1 * (1.0 + nuis.get("W_in_bias", 0.0)),
        X_in=0.1 + nuis.get("X_in_bias", 0.0),
        T_s=lambda t: steam(t) + b,
        R_f=rf,
    )


def records(ev, layout):
    """The sensor records of ``layout`` from one reference solve: (values, sigma)."""
    ref = ev.reference(t_max=T_MAX, nz=NZ, nt=NT)
    out = []
    for z, field, every, sigma in layout:
        times = np.arange(every, T_MAX + 1e-9, every)
        sol = ref.X if field == "X" else ref.W
        out.append((sol.interpolate(np.full_like(times, z), times), np.full(times.size, sigma)))
    return np.concatenate([v for v, _ in out]), np.concatenate([s for _, s in out])


LAYOUTS = {
    "outlet X": [(10.0, "X", 5.0, NOISE_X)],
    "outlet X + W": [(10.0, "X", 5.0, NOISE_X), (10.0, "W", 5.0, NOISE_W)],
    "outlet X + W + mid X": [
        (10.0, "X", 5.0, NOISE_X),
        (10.0, "W", 5.0, NOISE_W),
        (MID, "X", 5.0, NOISE_X),
    ],
    "outlet X + W every 1 s": [(10.0, "X", 1.0, NOISE_X), (10.0, "W", 1.0, NOISE_W)],
}


def jacobian(layout, modes, nuisances, truth_c):
    """Central finite differences of the records with respect to the modes and nuisances."""
    cols, names = [], []
    h_c = 0.1  # 10 % of the mean fouling
    for k in range(modes):
        cp, cm = truth_c.copy(), truth_c.copy()
        cp[k] += h_c
        cm[k] -= h_c
        yp, _ = records(plant(cp), layout)
        ym, sigma = records(plant(cm), layout)
        cols.append((yp - ym) / (2 * h_c))
        names.append(f"c{k}")
    for name in nuisances:
        h = NUISANCES[name][0]
        yp, _ = records(plant(truth_c, {name: h}), layout)
        ym, sigma = records(plant(truth_c, {name: -h}), layout)
        cols.append((yp - ym) / (2 * h))
        names.append(name)
    return np.stack(cols, axis=1), sigma, names


def fisher(J, sigma, names):
    from pinnrl.analysis import fisher_information

    s = fisher_information(
        torch.as_tensor(J, dtype=torch.float64),
        torch.as_tensor(sigma, dtype=torch.float64),
        names,
        rank_tol=1e-12,
    )
    cov = s.covariance.numpy()
    n_modes = sum(1 for n in names if n.startswith("c"))
    ones = np.ones(n_modes)  # R_f(T) = sum_k c_k P_k(1) = sum_k c_k
    end_var = (
        float(ones @ cov[:n_modes, :n_modes] @ ones)
        if np.isfinite(cov[:n_modes, :n_modes]).all()
        else math.inf
    )
    return {
        "names": list(names),
        "std": [float(v) for v in s.std],
        "correlation": [[float(v) for v in row] for row in s.correlation],
        "condition_number": float(s.condition_number),
        "identifiable": [bool(v) for v in s.identifiable],
        "eigenvalues": [float(v) for v in s.eigenvalues],
        "std_end_value_rel": math.sqrt(end_var) * C_SCALE / R_END
        if np.isfinite(end_var)
        else math.inf,
    }


def misfit_at(misfit, r_end):
    """The misfit as a function of the steam bias at a fixed drift."""

    def f(b):
        return misfit(r_end, b)

    return f


def profile(n_datasets, seed0=0):
    """Misfit of noisy outlet records vs the drift's end value, steam bias fixed and free."""
    layout = LAYOUTS["outlet X + W"]
    truth_c = np.zeros(2)
    truth_c[0] = truth_c[1] = R_END / 2 / C_SCALE
    clean, sigma = records(plant(truth_c), layout)
    grid = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]) * R_END
    cache: dict[tuple[float, float], np.ndarray] = {}

    def model(r_end, b):
        key = (round(float(r_end), 12), round(float(b), 9))
        if key not in cache:
            c = np.array([r_end / 2 / C_SCALE, r_end / 2 / C_SCALE])
            cache[key] = records(plant(c, {"T_s_bias": b}), layout)[0]
        return cache[key]

    # Fisher information of (r_end, b) at the truth, for the parabola.
    h_r, h_b = 0.1 * R_END, 0.1
    J = np.stack(
        [
            (model(R_END + h_r, 0.0) - model(R_END - h_r, 0.0)) / (2 * h_r),
            (model(R_END, h_b) - model(R_END, -h_b)) / (2 * h_b),
        ],
        axis=1,
    )
    fish = fisher(J, sigma, ["r_end", "T_s_bias"])
    F = (J / sigma[:, None]).T @ (J / sigma[:, None])
    out = {
        "grid_rel": (grid / R_END).tolist(),
        "fisher": fish,
        "F_rr": float(F[0, 0]),
        "F_profile_rr": float(F[0, 0] - F[0, 1] ** 2 / F[1, 1]),
        "datasets": [],
    }
    for d in range(n_datasets):
        rng = np.random.default_rng(seed0 + d)
        y = clean + sigma * rng.standard_normal(clean.size)

        def misfit(r_end, b, y=y):
            return float((((model(r_end, b) - y) / sigma) ** 2).sum())

        fixed = [misfit(r, 0.0) for r in grid]
        free, b_hat = [], []
        for r in grid:
            res = minimize_scalar(
                misfit_at(misfit, r), bounds=(-3.0, 3.0), method="bounded", options={"xatol": 1e-3}
            )
            free.append(float(res.fun))
            b_hat.append(float(res.x))
        out["datasets"].append({"fixed": fixed, "free": free, "b_hat": b_hat})
        print(
            f"profile data set {d}: fixed min at {grid[int(np.argmin(fixed))] / R_END:.2f}, free min at {grid[int(np.argmin(free))] / R_END:.2f}",
            flush=True,
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", type=int, default=5)
    parser.add_argument("--modes", type=int, default=6)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    truth_c = np.zeros(args.modes)
    truth_c[0] = truth_c[1] = R_END / 2 / C_SCALE
    t0 = time.time()
    result = {"modes": args.modes, "layouts": {}, "nuisances": {}}
    for name, layout in LAYOUTS.items():
        J, sigma, names = jacobian(layout, args.modes, [], truth_c)
        result["layouts"][name] = fisher(J, sigma, names)
        f = result["layouts"][name]
        print(
            f"{name:<26} std of modes {np.round(f['std'], 3)} end value {f['std_end_value_rel']:.3f} kappa {f['condition_number']:.1e} ({time.time() - t0:.0f}s)",
            flush=True,
        )
    layout = LAYOUTS["outlet X + W"]
    J_all, sigma, names_all = jacobian(layout, args.modes, list(NUISANCES), truth_c)
    biases = [n for n in NUISANCES if n != "U"]
    cases = [([n], False) for n in NUISANCES] + [(biases, False), (list(NUISANCES), True)]
    for case, drop_constant in cases:
        # U and the constant mode are confounded (1/U + R_f); with U unknown only the drift
        # (modes 1 and up) is asked for.
        keep = [
            i
            for i, n in enumerate(names_all)
            if (n.startswith("c") and not (drop_constant and n == "c0")) or n in case
        ]
        f = fisher(J_all[:, keep], sigma, [names_all[i] for i in keep])
        key = "+".join(case) + (" (drift only)" if drop_constant else "")
        result["nuisances"][key] = f
        print(
            f"with {key:<50} std {np.round(f['std'], 3)} end value {f['std_end_value_rel']:.3f} "
            f"kappa {f['condition_number']:.1e}",
            flush=True,
        )
    result["profile"] = profile(args.datasets)
    (args.out / "summary.json").write_text(json.dumps(result, indent=1))
    print("saved", args.out / "summary.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
