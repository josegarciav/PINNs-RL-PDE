"""J12: measurement and campaign design for the spray-dryer twin.

J11 identified the drying curve's two parameters from a five-point campaign
and found the droplet size, uncertain by 10 %, to be the nuisance that breaks
the estimate; J08 turned the result into a stickiness margin good to 1.0-1.5 K.
Both admitted two nuisances. Here every input of the tower is admitted with
the accuracy it is known to, as J09 did for the evaporator, and the remedies
are priced before any is bought, for the two things the twin is for: the
curve's parameters, and the stickiness margin at the operating limit.

Unknowns: the critical moisture and the exponent. Nuisances (1 sigma): wall
loss 30 %, droplet diameter 10 %, droplet velocity (residence time) 15 %, air
flow 3 %, solids flow 2 %, inlet air temperature 1 K, inlet humidity 0.001,
feed solids fraction 0.005.

1. ``nuisances``: the cost of each, alone and together.
2. ``remedies``: droplet size by laser diffraction, an air-flow calibration,
   more or better thermocouples, a thermocouple near the atomiser, replicate
   lab assays, powder sampled inside the tower, a larger campaign.
3. ``campaign``: which operating points to run: the next points chosen one at
   a time from a 5 x 5 window (``greedy_design``) against the factorial
   campaigns of J11.
4. ``validation``: baseline and the best design, 60 simulated campaigns each
   fitted through the nonlinear reference.

Usage: python experiments/J12_dryer_design/run.py [--fits 60] [--workers 6]
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
PARAMS = ["X_cr", "n", "U_wall", "d0", "v_p", "G", "F_s", "T_in", "Y_in", "w_s"]
PRIOR = {
    "U_wall": 0.30,
    "d0": 0.10,
    "v_p": 0.15,
    "G": 0.03,
    "F_s": 0.02,
    "T_in": 1.0,
    "Y_in": 1e-3,
    "w_s": 0.005,
}
LABEL = {
    "U_wall": "wall loss (30 %)",
    "d0": "droplet diameter (10 %)",
    "v_p": "droplet velocity (15 %)",
    "G": "air flow (3 %)",
    "F_s": "solids flow (2 %)",
    "T_in": "inlet air temperature (1 K)",
    "Y_in": "inlet humidity (0.001)",
    "w_s": "feed solids fraction (0.005)",
}
STEP = np.array([0.01, 0.01, 0.01, 0.005, 0.01, 0.005, 0.005, 0.2, 2e-4, 1e-3])
KNOWN = 1e-6
FACTORIAL5 = [(453.15, 0.010), (438.15, 0.010), (468.15, 0.010), (453.15, 0.008), (453.15, 0.012)]
FACTORIAL9 = [(T, F) for T in (438.15, 453.15, 468.15) for F in (0.008, 0.010, 0.012)]
WINDOW = [(T, F) for T in np.linspace(433.15, 473.15, 5) for F in np.linspace(0.008, 0.012, 5)]
LIMIT_POINT = (453.15, 0.011)  # J08's point at the sticky limit
Z90 = 1.6448536269514722


def nominal():
    return np.array([1.0, 1.5] + [0.0] * 8)


def dryer(point, theta):
    """The tower at a nominal operating point with the true inputs displaced by the nuisances."""
    from pinnrl.process import SprayDryer

    dr = SprayDryer()
    xcr, n, uw, d0, vp, g, fs, tin, yin, ws = theta
    drop = replace(
        dr.droplet,
        X_cr=xcr,
        n=n,
        d0=dr.droplet.d0 * (1 + d0),
        solids_fraction=dr.droplet.solids_fraction + ws,
    )
    return replace(
        dr,
        droplet=drop,
        U_wall=dr.U_wall * (1 + uw),
        particle_velocity=dr.particle_velocity * (1 + vp),
        air_flow=dr.air_flow * (1 + g),
        solids_flow=point[1] * (1 + fs),
        T_in=point[0] + tin,
        Y_in=dr.Y_in + yin,
    )


def point_records(point, theta, extra_probes=(), tower=False):
    """``{kind: values}`` at one operating point, and the outlet stickiness margin."""
    dr = dryer(point, theta)
    ref = dr.reference()
    out = {
        "T_air": np.interp(PROBES, ref.z, ref.T_a),
        "Y_out": np.array([ref.Y[-1]]),
        "X_out": np.array([ref.X[-1]]),
    }
    if extra_probes:
        out["T_extra"] = np.interp(extra_probes, ref.z, ref.T_a)
    if tower:
        out["X_tower"] = np.interp(PROBES[:-1], ref.z, ref.X)
    return out, float(dr.stickiness_margin(ref)[-1])


def layout(extra_probes=(), tower=False, noise=None):
    """The record kinds of one operating point in order, with the noise of each row."""
    nz = {**NOISE, **(noise or {})}
    kinds = [
        ("T_air", len(PROBES), nz["T_air"]),
        ("Y_out", 1, nz["Y_out"]),
        ("X_out", 1, nz["X_out"]),
    ]
    if extra_probes:
        kinds.append(("T_extra", len(extra_probes), nz["T_air"]))
    if tower:
        kinds.append(("X_tower", len(PROBES) - 1, nz["X_tower"]))
    return kinds


def vector(point, theta, extra_probes=(), tower=False):
    rec, _ = point_records(point, theta, extra_probes, tower)
    return np.concatenate([rec[k] for k, _, _ in layout(extra_probes, tower)])


def sigma_vector(extra_probes=(), tower=False, noise=None):
    return np.concatenate([np.full(m, s) for _, m, s in layout(extra_probes, tower, noise)])


def jacobian(fn, x0=None):
    x0 = nominal() if x0 is None else x0
    cols = []
    for j, h in enumerate(STEP):
        up, dn = x0.copy(), x0.copy()
        up[j] += h
        dn[j] -= h
        cols.append((np.atleast_1d(fn(up)) - np.atleast_1d(fn(dn))) / (2 * h))
    return np.stack(cols, axis=-1)


def point_jacobian(job):
    point, extra, tower = job
    return jacobian(lambda th: vector(point, th, extra, tower))


def prior_std(overrides=None):
    p = {**PRIOR, **(overrides or {})}
    return np.array([np.inf, np.inf] + [p[k] for k in PARAMS[2:]])


# ------------------------------------------------------------------ validation
def fit_campaign(job):
    """One simulated campaign of a design, fitted by least squares scaled by the priors."""
    from scipy.optimize import least_squares

    points, seed, extra, tower, noise, ps = (
        job[k] for k in ("points", "seed", "extra", "tower", "noise", "ps")
    )
    rng = np.random.default_rng(seed)
    truth = nominal()
    truth[2:] = rng.normal(0.0, ps[2:])
    sig = np.tile(sigma_vector(extra, tower, noise), len(points))
    clean = np.concatenate([vector(p, truth, extra, tower) for p in points])
    y = clean + sig * rng.standard_normal(sig.size)

    def unpack(z):
        return np.concatenate([z[:2], z[2:] * ps[2:]])

    def resid(z):
        th = unpack(z)
        try:
            model = np.concatenate([vector(p, th, extra, tower) for p in points])
        except RuntimeError:
            return np.full(sig.size + 8, 1e3)
        return np.concatenate([(model - y) / sig, z[2:]])

    z0 = np.concatenate([[0.8, 1.2], np.zeros(8)])
    lower = np.concatenate([[0.3, 0.4], np.full(8, -4.0)])
    upper = np.concatenate([[1.45, 4.0], np.full(8, 4.0)])
    tic = time.perf_counter()
    res = least_squares(resid, z0, bounds=(lower, upper), diff_step=1e-3, x_scale=0.1, max_nfev=60)
    theta = unpack(res.x)
    m_hat = point_records(LIMIT_POINT, theta)[1]
    m_true = point_records(LIMIT_POINT, truth)[1]
    return {
        "design": job["design"],
        "seed": seed,
        "theta": theta.tolist(),
        "truth": truth.tolist(),
        "margin_error": m_hat - m_true,
        "time_s": time.perf_counter() - tic,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fits", type=int, default=60)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from pinnrl.analysis import greedy_design, information, posterior_covariance, target_std

    NEAR = (0.3, 0.6)  # two thermocouples near the atomiser
    all_points = list(dict.fromkeys(FACTORIAL5 + FACTORIAL9 + WINDOW))
    tic = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jacs = list(pool.map(point_jacobian, [(p, NEAR, True) for p in all_points]))
    Jp = dict(zip(all_points, jacs, strict=True))  # rows: T_air(6), Y, X, T_extra(2), X_tower(5)
    rows_base = slice(0, 8)
    rows_extra = slice(8, 10)
    rows_tower = slice(10, 15)
    margin_target = jacobian(lambda th: point_records(LIMIT_POINT, th)[1])[0]
    print(f"sensitivities at {len(all_points)} operating points in {time.time() - tic:.0f} s")

    def info(points, extra=False, tower=False, noise=None):
        nz = {**NOISE, **(noise or {})}
        sig = np.array([nz["T_air"]] * 6 + [nz["Y_out"], nz["X_out"]])
        F = np.zeros((len(PARAMS), len(PARAMS)))
        for p in points:
            F += information(Jp[p][rows_base], sig)
            if extra:
                F += information(Jp[p][rows_extra], nz["T_air"])
            if tower:
                F += information(Jp[p][rows_tower], nz["X_tower"])
        return F

    def evaluate(F, overrides=None):
        cov = posterior_covariance(F, prior_std(overrides))
        return {
            "X_cr": float(np.sqrt(cov[0, 0])),
            "n": float(np.sqrt(cov[1, 1])),
            "margin": target_std(cov, margin_target),
            "nuisance_std": np.sqrt(np.diag(cov))[2:].tolist(),
        }

    out: dict = {"params": PARAMS, "prior": PRIOR, "limit_point": LIMIT_POINT}
    exact = {k: KNOWN * v for k, v in PRIOR.items()}
    F5 = info(FACTORIAL5)

    # 1. nuisances
    rows = [{"name": "all inputs exact", **evaluate(F5, exact)}]
    for k in PARAMS[2:]:
        rows.append({"name": LABEL[k], "key": k, **evaluate(F5, {**exact, k: PRIOR[k]})})
    rows.append(
        {
            "name": "J11's two (wall loss, droplet diameter)",
            **evaluate(F5, {**exact, "U_wall": 0.3, "d0": 0.1}),
        }
    )
    rows.append(
        {
            "name": "droplet diameter and velocity together",
            **evaluate(F5, {**exact, "d0": PRIOR["d0"], "v_p": PRIOR["v_p"]}),
        }
    )
    rows.append(
        {
            "name": "the six balance inputs, droplets exact",
            **evaluate(F5, {"d0": exact["d0"], "v_p": exact["v_p"]}),
        }
    )
    rows.append({"name": "all admitted (the baseline)", **evaluate(F5)})
    out["nuisances"] = rows
    print(
        "\n1. what each input's uncertainty costs (five-point campaign): std of X_cr | n | margin at the limit (K)"
    )
    for r in rows:
        print(f"   {r['name']:<42} {r['X_cr']:.3f} | {r['n']:.3f} | {r['margin']:.2f}")

    # 2. remedies
    remedies = [
        ("baseline", F5, None),
        ("droplet size by laser diffraction (3 %)", F5, {"d0": 0.03}),
        ("droplet velocity known to 5 %", F5, {"v_p": 0.05}),
        ("air flow calibrated to 1 %", F5, {"G": 0.01}),
        ("wall loss known to 10 %", F5, {"U_wall": 0.10}),
        ("feed solids fraction to 0.002", F5, {"w_s": 0.002}),
        ("thermocouples to 0.2 K", info(FACTORIAL5, noise={"T_air": 0.2}), None),
        ("two thermocouples near the atomiser (0.3, 0.6 m)", info(FACTORIAL5, extra=True), None),
        ("outlet humidity to 0.0003", info(FACTORIAL5, noise={"Y_out": 3e-4}), None),
        ("four replicate lab assays (0.0015)", info(FACTORIAL5, noise={"X_out": 1.5e-3}), None),
        ("powder sampled at five heights", info(FACTORIAL5, tower=True), None),
        ("nine operating points (factorial)", info(FACTORIAL9), None),
        ("laser diffraction + velocity to 5 %", F5, {"d0": 0.03, "v_p": 0.05}),
        (
            "laser diffraction + thermocouples near the atomiser",
            info(FACTORIAL5, extra=True),
            {"d0": 0.03},
        ),
        (
            "laser diffraction + powder sampled at five heights",
            info(FACTORIAL5, tower=True),
            {"d0": 0.03},
        ),
        (
            "laser diffraction + velocity + tower samples",
            info(FACTORIAL5, tower=True),
            {"d0": 0.03, "v_p": 0.05},
        ),
    ]
    out["remedies"] = [{"name": n, **evaluate(F, o)} for n, F, o in remedies]
    print("\n2. remedies: std of X_cr | n | margin at the limit (K) | droplet diameter known to")
    for r in out["remedies"]:
        print(
            f"   {r['name']:<52} {r['X_cr']:.3f} | {r['n']:.3f} | {r['margin']:.2f} | {100 * r['nuisance_std'][1]:.1f} %"
        )

    # 3. which operating points
    cands = [Jp[p][rows_base] for p in WINDOW]
    sig8 = np.array([NOISE["T_air"]] * 6 + [NOISE["Y_out"], NOISE["X_out"]])
    camp = {}
    for name, target in (("margin", margin_target), ("X_cr", np.eye(len(PARAMS))[0])):
        g = greedy_design(
            cands, [sig8] * len(cands), 9, prior_std=prior_std(), target=target, repeat=True
        )
        camp[name] = {
            "points": [WINDOW[i] for i in g.chosen],
            "criterion": g.criterion[1:],
        }
    fact = {
        "factorial 5": evaluate(F5),
        "factorial 9": evaluate(info(FACTORIAL9)),
        "greedy 5 (for the margin)": evaluate(info(camp["margin"]["points"][:5])),
        "greedy 9 (for the margin)": evaluate(info(camp["margin"]["points"][:9])),
        "greedy 5 (for X_cr)": evaluate(info(camp["X_cr"]["points"][:5])),
        "base point five times": evaluate(info([FACTORIAL5[0]] * 5)),
    }
    out["campaign"] = {"greedy": camp, "compare": fact, "window": WINDOW}
    print("\n3. which operating points: std of X_cr | n | margin")
    for k, v in fact.items():
        print(f"   {k:<28} {v['X_cr']:.3f} | {v['n']:.3f} | {v['margin']:.2f}")
    print(
        "   greedy points for the margin:",
        [(round(T - 273.15), round(1e3 * F, 1)) for T, F in camp["margin"]["points"]],
    )
    print(
        "   greedy points for X_cr:      ",
        [(round(T - 273.15), round(1e3 * F, 1)) for T, F in camp["X_cr"]["points"][:5]],
    )

    # 4. validation
    designs = {
        "baseline": dict(points=FACTORIAL5, extra=(), tower=False, noise=None, overrides=None),
        "laser diffraction": dict(
            points=FACTORIAL5, extra=(), tower=False, noise=None, overrides={"d0": 0.03}
        ),
        "laser diffraction + velocity + tower samples": dict(
            points=FACTORIAL5, extra=(), tower=True, noise=None, overrides={"d0": 0.03, "v_p": 0.05}
        ),
    }
    jobs = [
        {
            "design": name,
            "seed": s,
            "ps": prior_std(cfg["overrides"]),
            **{k: cfg[k] for k in ("points", "extra", "tower", "noise")},
        }
        for name, cfg in designs.items()
        for s in range(args.fits)
    ]
    tic = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fits = list(pool.map(fit_campaign, jobs, chunksize=2))
    val = []
    print(f"\n4. validation: {args.fits} fitted campaigns per design ({time.time() - tic:.0f} s)")
    for name, cfg in designs.items():
        pred = evaluate(info(cfg["points"], tower=cfg["tower"]), cfg["overrides"])
        rows_ = [f for f in fits if f["design"] == name]
        th = np.array([f["theta"][:2] for f in rows_])
        me = np.array([f["margin_error"] for f in rows_])
        val.append(
            {
                "design": name,
                "predicted": {k: pred[k] for k in ("X_cr", "n", "margin")},
                "found": {
                    "X_cr": float(th[:, 0].std(ddof=1)),
                    "n": float(th[:, 1].std(ddof=1)),
                    "margin": float(np.sqrt(np.mean(me**2))),
                },
                "bias": {
                    "X_cr": float(th[:, 0].mean() - 1.0),
                    "n": float(th[:, 1].mean() - 1.5),
                    "margin": float(me.mean()),
                },
                "margin_coverage": float(np.mean(np.abs(me) <= Z90 * pred["margin"])),
                "margin_errors": me.tolist(),
                "time_s": float(np.median([f["time_s"] for f in rows_])),
            }
        )
        v = val[-1]
        print(
            f"   {name:<46} X_cr {v['found']['X_cr']:.3f} (pred {pred['X_cr']:.3f})  n {v['found']['n']:.3f} ({pred['n']:.3f})"
            f"  margin {v['found']['margin']:.2f} K ({pred['margin']:.2f}), coverage {v['margin_coverage']:.2f}, {v['time_s']:.1f} s"
        )
    out["validation"] = val
    (args.out / "j12.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved", args.out / "j12.json")


if __name__ == "__main__":
    main()
