"""J11: the drying-rate closure at dryer scale. What do a tower's instruments determine?

J07 identified the closure f(X) from a single droplet's mass and temperature
record, 400 readings of the droplet itself. A spray dryer gives nothing of the
kind: air thermocouples at a few heights, the outlet air's humidity and the
powder's moisture from the lab, at a handful of operating points. This
experiment asks, on ``pinnrl.process.SprayDryer`` (the droplet coupled to its
air, co-current plug flow):

1. ``cost``: what a forward solve and a fit through it cost (the question the
   roadmap set before any network is trained at this scale);
2. ``parametric``: how well the two numbers of the characteristic drying curve
   (critical moisture, exponent) are determined, with the wall-loss coefficient
   and the droplet size known or uncertain (30 % and 10 %);
3. ``free``: how much of a free-form closure (monotone, piecewise linear on 12
   knots, zero at equilibrium) the instruments determine: the posterior band of
   f(X) and the number of directions the data move, for the plant's instruments
   and with powder sampled inside the tower;
4. ``prediction``: whether that matters for what a twin is asked, the powder
   moisture at an operating point outside the campaign.

Linear-Gaussian predictions (``pinnrl.analysis``) are checked by fitting
simulated campaigns through the reference solver.

Usage: python experiments/J11_dryer_closure/run.py [--fits 40] [--workers 6]
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
PROBES = (1.0, 2.0, 3.5, 5.0, 6.5, 8.0)
NOISE = {"T_air": 0.5, "Y_out": 1e-3, "X_out": 3e-3, "X_tower": 0.01}
CAMPAIGNS = {  # (inlet air temperature K, solids flow kg/s)
    "1 point": [(453.15, 0.010)],
    "5 points": [
        (453.15, 0.010),
        (438.15, 0.010),
        (468.15, 0.010),
        (453.15, 0.008),
        (453.15, 0.012),
    ],
    "9 points": [(T, F) for T in (438.15, 453.15, 468.15) for F in (0.008, 0.010, 0.012)],
}
NEW_POINT = (428.15, 0.013)  # colder and more loaded than anything in the campaign
N_KNOTS = 12
PRIOR_F = 0.3  # prior std of each knot value of the free-form closure
PRIOR_NUISANCE = {"U_wall": 0.30, "d0": 0.10}  # relative


def tower():
    from pinnrl.process import SprayDryer

    return SprayDryer()


def knots():
    d = tower().droplet
    return np.linspace(d.X_eq, d.X0, N_KNOTS)


def closure_free(values):
    """Piecewise linear through ``(knots, [0, *values])``; no cap, for the linear analysis."""
    k = knots()
    v = np.concatenate([[0.0], np.asarray(values, dtype=float)])
    return lambda X: np.interp(X, k, v)


def truth_free():
    return tower().droplet.drying_curve(knots())[1:]


def simulate(point, closure=None, cdc=None, nuisance=(0.0, 0.0), sample_tower=False):
    """The records of one operating point as one vector (and the outlet moisture)."""
    dr = tower()
    drop = dr.droplet
    if cdc is not None:
        drop = replace(drop, X_cr=cdc[0], n=cdc[1])
    drop = replace(drop, d0=drop.d0 * (1.0 + nuisance[1]))
    if closure is not None:
        drop = replace(drop, rate_closure=closure)
    dr = replace(
        dr, droplet=drop, T_in=point[0], solids_flow=point[1], U_wall=dr.U_wall * (1 + nuisance[0])
    )
    ref = dr.reference()
    rec = dr.records(PROBES, ref)
    parts = [rec["T_air"], rec["Y_out"], rec["X_out"]]
    if sample_tower:
        parts.append(np.interp(PROBES[:-1], ref.z, ref.X))
    return np.concatenate(parts), float(ref.X[-1])


def sigma(n_points, sample_tower=False):
    one = [NOISE["T_air"]] * len(PROBES) + [NOISE["Y_out"], NOISE["X_out"]]
    if sample_tower:
        one += [NOISE["X_tower"]] * (len(PROBES) - 1)
    return np.tile(one, n_points)


def campaign(points, theta, kind, sample_tower=False):
    """All records of a campaign for ``theta = [closure parameters, U_wall rel, d0 rel]``."""
    n = 2 if kind == "cdc" else N_KNOTS - 1
    nuisance = tuple(theta[n : n + 2])
    kwargs = {"cdc": theta[:n]} if kind == "cdc" else {"closure": closure_free(theta[:n])}
    return np.concatenate(
        [simulate(p, nuisance=nuisance, sample_tower=sample_tower, **kwargs)[0] for p in points]
    )


def nominal(kind):
    d = tower().droplet
    core = np.array([d.X_cr, d.n]) if kind == "cdc" else truth_free()
    return np.concatenate([core, [0.0, 0.0]])


def jacobian(fn, x0, steps):
    cols = []
    for j, h in enumerate(steps):
        up, dn = x0.copy(), x0.copy()
        up[j] += h
        dn[j] -= h
        cols.append((np.atleast_1d(fn(up)) - np.atleast_1d(fn(dn))) / (2 * h))
    return np.stack(cols, axis=-1)


def steps_for(kind):
    return np.array(([0.01, 0.01] if kind == "cdc" else [0.01] * (N_KNOTS - 1)) + [0.01, 0.005])


def prior_for(kind, nuisances_known):
    core = [np.inf, np.inf] if kind == "cdc" else [PRIOR_F] * (N_KNOTS - 1)
    nz = [1e-6, 1e-6] if nuisances_known else [PRIOR_NUISANCE["U_wall"], PRIOR_NUISANCE["d0"]]
    return np.array(core + nz)


def new_point_target(kind):
    """d X_out(new point) / d theta at the nominal point."""
    n = 2 if kind == "cdc" else N_KNOTS - 1

    def fn(theta):
        kwargs = {"cdc": theta[:n]} if kind == "cdc" else {"closure": closure_free(theta[:n])}
        return simulate(NEW_POINT, nuisance=tuple(theta[n : n + 2]), **kwargs)[1]

    return jacobian(fn, nominal(kind), steps_for(kind))[0]


# ------------------------------------------------------------------ fits
def fit_campaign(job):
    """One noisy campaign fitted through the reference solver (bounded least squares)."""
    from scipy.optimize import least_squares

    kind, points, seed, sample_tower, known = (
        job[k] for k in ("kind", "points", "seed", "sample_tower", "known")
    )
    x_true = nominal(kind)
    rng = np.random.default_rng(seed)
    ps = prior_for(kind, known)
    truth = x_true.copy()
    if not known:
        truth[-2:] = rng.normal(0.0, ps[-2:])
    sig = sigma(len(points), sample_tower)
    y = campaign(points, truth, kind, sample_tower) + sig * rng.standard_normal(sig.size)
    n = x_true.size - 2
    k = knots()

    if kind == "cdc":

        def unpack_full(z):
            return np.concatenate([z[:2], z[2:] * ps[-2:]])

        def resid_full(z):
            r = (campaign(points, unpack_full(z), kind, sample_tower) - y) / sig
            return np.concatenate([r, z[2:]])

        z0 = np.array([0.7, 1.0, 0.0, 0.0])  # a cold start: critical moisture 0.7, a linear fall
        bounds = ([0.2, 0.3, -5, -5], [1.45, 4.0, 5, 5])
        prior_mean = None
    else:
        prior_mean = (k[1:] - k[0]) / (
            k[-1] - k[0]
        )  # a straight line: nothing of the plant's curve

        def unpack_full(z):  # non-negative increments -> increasing values, capped at 1
            return np.concatenate([np.minimum(np.cumsum(z[:n]), 1.0), z[n:] * ps[-2:]])

        def resid_full(z):
            th = unpack_full(z)
            r = (campaign(points, th, kind, sample_tower) - y) / sig
            return np.concatenate([r, (th[:n] - prior_mean) / PRIOR_F, z[n:]])

        z0 = np.concatenate([np.diff(np.concatenate([[0.0], prior_mean])), [0.0, 0.0]])
        bounds = ([0.0] * n + [-5, -5], [1.0] * n + [5, 5])
    if known:  # the nuisances are fixed at zero
        m = len(z0) - 2

        def unpack(z):
            return unpack_full(np.concatenate([z, [0.0, 0.0]]))

        def resid(z):
            return resid_full(np.concatenate([z, [0.0, 0.0]]))

        z0, bounds = z0[:m], (bounds[0][:m], bounds[1][:m])
    else:
        unpack, resid = unpack_full, resid_full
    tic = time.perf_counter()
    res = least_squares(resid, z0, bounds=bounds, diff_step=1e-3, x_scale=0.1, max_nfev=200)
    elapsed = time.perf_counter() - tic
    theta = unpack(res.x)
    kwargs = {"cdc": theta[:n]} if kind == "cdc" else {"closure": closure_free(theta[:n])}
    kw_true = {"cdc": truth[:n]} if kind == "cdc" else {"closure": closure_free(truth[:n])}
    pred = simulate(NEW_POINT, nuisance=tuple(theta[n:]), **kwargs)[1]
    actual = simulate(NEW_POINT, nuisance=tuple(truth[n:]), **kw_true)[1]
    d = tower().droplet
    grid = np.linspace(d.X_eq, d.X0, 300)
    f_true = d.drying_curve(grid)
    f_hat = (
        replace(d, X_cr=theta[0], n=theta[1]).drying_curve(grid)
        if kind == "cdc"
        else closure_free(theta[:n])(grid)
    )
    return {
        **{k_: job[k_] for k_ in ("kind", "campaign", "seed", "sample_tower", "known")},
        "theta": theta.tolist(),
        "truth": truth.tolist(),
        "closure_rel_l2": float(np.linalg.norm(f_hat - f_true) / np.linalg.norm(f_true)),
        "new_point_error": pred - actual,
        "new_point_truth": actual,
        "chi2_reduced": float(np.sum(resid(res.x)[: sig.size] ** 2) / sig.size),
        "time_s": elapsed,
        "nfev": int(res.nfev),
        "f_hat": f_hat.tolist() if seed < 3 else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fits", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from pinnrl.analysis import information, posterior_covariance, target_std

    dr = tower()
    out: dict = {"probes": PROBES, "noise": NOISE, "campaigns": CAMPAIGNS, "new_point": NEW_POINT}

    # 1. cost of the forward solve
    tic = time.perf_counter()
    for _ in range(20):
        dr.reference()
    out["solve_ms"] = 1e3 * (time.perf_counter() - tic) / 20
    out["balances"] = dr.balances(dr.reference(n=801))
    profiles = {}
    for name, p in (("base", CAMPAIGNS["5 points"][0]), ("new", NEW_POINT)):
        r = replace(dr, T_in=p[0], solids_flow=p[1]).reference()
        profiles[name] = {k: getattr(r, k).tolist() for k in ("z", "X", "T_p", "Y", "T_a")}
    out["profiles"] = profiles
    print(f"1. one reference solve: {out['solve_ms']:.1f} ms; balances {out['balances']}")

    # 2-4. linear predictions
    linear = []
    targets = {kind: new_point_target(kind) for kind in ("cdc", "free")}
    x_new = simulate(NEW_POINT)[1]
    out["new_point_X"] = x_new
    for kind in ("cdc", "free"):
        x0, h = nominal(kind), steps_for(kind)
        n = x0.size - 2
        for camp, points in CAMPAIGNS.items():
            for sample_tower in (False, True):
                J = jacobian(lambda th: campaign(points, th, kind, sample_tower), x0, h)  # noqa: B023
                F = information(J, sigma(len(points), sample_tower))
                for known in (True, False):
                    ps = prior_for(kind, known)
                    try:
                        cov = posterior_covariance(F, ps)
                    except np.linalg.LinAlgError:
                        continue
                    row = {
                        "kind": kind,
                        "campaign": camp,
                        "sample_tower": sample_tower,
                        "known": known,
                        "std": np.sqrt(np.diag(cov)).tolist(),
                        "new_point_std": target_std(cov, targets[kind]),
                    }
                    if kind == "free":
                        # directions of the closure the data move: prior-whitened information > 1
                        Fc = np.linalg.inv(cov[:n, :n]) - np.eye(n) / PRIOR_F**2
                        eig = np.linalg.eigvalsh(PRIOR_F**2 * Fc)
                        row["determined"] = int(np.sum(eig > 1.0))
                        row["eigenvalues"] = np.sort(eig)[::-1].tolist()
                    linear.append(row)
    out["linear"] = linear

    def lin(kind, camp, sample_tower, known):
        return next(
            r
            for r in linear
            if (r["kind"], r["campaign"], r["sample_tower"], r["known"])
            == (kind, camp, sample_tower, known)
        )

    print("\n2. the characteristic curve's two numbers (std of X_cr | n; truth 1.0, 1.5)")
    for r in (r for r in linear if r["kind"] == "cdc" and not r["sample_tower"]):
        print(
            f"   {r['campaign']:<9} nuisances {'known    ' if r['known'] else 'uncertain'}  "
            f"{r['std'][0]:.3f} | {r['std'][1]:.3f}   X_out at the new point +-{r['new_point_std']:.4f}"
        )
    print(
        "\n3. the free-form closure (prior 0.3 per knot): directions determined, mean posterior std of the knots"
    )
    for r in (r for r in linear if r["kind"] == "free"):
        print(
            f"   {r['campaign']:<9} {'+ tower samples' if r['sample_tower'] else 'plant only     '} "
            f"nuisances {'known    ' if r['known'] else 'uncertain'}  {r['determined']:2d} of {N_KNOTS - 1}"
            f"  knots +-{np.mean(r['std'][: N_KNOTS - 1]):.3f}   X_out(new) +-{r['new_point_std']:.4f}"
        )

    # fits of simulated campaigns
    configs = [
        ("cdc", "5 points", False, True),
        ("cdc", "5 points", False, False),
        ("cdc", "1 point", False, False),
        ("free", "5 points", False, True),
        ("free", "5 points", False, False),
        ("free", "5 points", True, False),
    ]
    jobs = [
        {
            "kind": k,
            "campaign": c,
            "points": CAMPAIGNS[c],
            "seed": s,
            "sample_tower": st,
            "known": kn,
        }
        for k, c, st, kn in configs
        for s in range(args.fits)
    ]
    tic = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fits = list(pool.map(fit_campaign, jobs))
    out["fits"] = fits
    print(f"\n4. {len(fits)} fits of simulated campaigns in {time.time() - tic:.0f} s")
    summary = []
    for k, c, st, kn in configs:
        rows = [
            f
            for f in fits
            if (f["kind"], f["campaign"], f["sample_tower"], f["known"]) == (k, c, st, kn)
        ]
        err = np.array([f["new_point_error"] for f in rows])
        pred = lin(k, c, st, kn)
        s = {
            "kind": k,
            "campaign": c,
            "sample_tower": st,
            "known": kn,
            "closure_rel_l2_median": float(np.median([f["closure_rel_l2"] for f in rows])),
            "closure_rel_l2_max": float(np.max([f["closure_rel_l2"] for f in rows])),
            "new_point_rmse": float(np.sqrt(np.mean(err**2))),
            "new_point_bias": float(err.mean()),
            "new_point_std_predicted": pred["new_point_std"],
            "time_s_median": float(np.median([f["time_s"] for f in rows])),
            "nfev_median": float(np.median([f["nfev"] for f in rows])),
            "chi2_reduced_median": float(np.median([f["chi2_reduced"] for f in rows])),
        }
        if k == "cdc":
            th = np.array([f["theta"][:2] for f in rows])
            s["param_std_found"] = th.std(axis=0, ddof=1).tolist()
            s["param_bias"] = (th.mean(axis=0) - nominal("cdc")[:2]).tolist()
            s["param_std_predicted"] = pred["std"][:2]
        summary.append(s)
        label = f"{k:<4} {c:<9} {'+tower' if st else 'plant '} {'known    ' if kn else 'uncertain'}"
        extra = (
            f"  X_cr +-{s['param_std_found'][0]:.3f} (pred {s['param_std_predicted'][0]:.3f})"
            f" n +-{s['param_std_found'][1]:.3f} (pred {s['param_std_predicted'][1]:.3f})"
            if k == "cdc"
            else ""
        )
        print(
            f"   {label}  closure L2 {100 * s['closure_rel_l2_median']:.1f} % (max {100 * s['closure_rel_l2_max']:.0f})"
            f"  X_out(new) rmse {s['new_point_rmse']:.4f} (pred {s['new_point_std_predicted']:.4f})"
            f"  {s['time_s_median']:.1f} s{extra}"
        )
    out["summary"] = summary
    (args.out / "j11.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved", args.out / "j11.json")


if __name__ == "__main__":
    main()
