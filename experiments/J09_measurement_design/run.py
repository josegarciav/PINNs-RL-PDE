"""J09: which measurement narrows the fouling estimate? Bayesian design on the evaporator.

J06 left the fouling interval at +-12 to 15 % once the steam record's 0.3 K
offset is admitted. Here every instrument error is admitted and the remedies
are priced in the same currency, the posterior standard deviation of the
fouling at the end of the window, before any of them is bought.

The unknowns are six Legendre modes of R_f(t) and six nuisances with the
accuracy each instrument is trusted to as its prior: the steam record's offset
``b``, the feed solids' offset, the feed flow's gain, the outlet density
meter's offset, the outlet flow meter's gain and the condensate meter's gain.
Sensitivities come from the method-of-lines reference by central differences;
the posterior covariance is ``(P0^-1 + J^T R^-1 J)^-1`` (``pinnrl.analysis``).

Parts:

1. ``nuisances``: starting from perfectly known instruments, admit each error
   alone, then all of them: which one costs the most?
2. ``remedies``: from the all-admitted baseline (online outlet solids and
   flow), one remedy at a time and the best combinations: calibrate an
   instrument, add a condensate (heat-duty) meter, excite the steam, take lab
   samples of the outlet solids.
3. ``lab``: when to take ``k`` lab samples (greedy, checked against exhaustive
   search for k = 3; against equally spaced and end-loaded schedules).
4. ``validation``: for three designs, 200 synthetic data sets with the
   nuisances drawn from their priors, fitted through the nonlinear reference
   (Gauss-Newton with the nominal Jacobian, checked against a full
   least-squares fit): does the predicted standard deviation match the errors,
   and does the 90 % interval cover?

Usage: python experiments/J09_measurement_design/run.py [--datasets 200] [--workers 8]
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numpy.polynomial import legendre

HERE = Path(__file__).parent
T_MAX = 1200.0
EVERY = 5.0
TIMES = np.arange(EVERY, T_MAX + 1e-9, EVERY)
NZ, NT = 101, 241
C_SCALE = 1e-4
MODES = 6
C_TRUE = np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])  # the ramp 0 -> 2e-4
Z90 = 1.6448536269514722

# noise of one record
NOISE = {"X": 0.002, "W": 0.001, "C": 1.2e-4, "lab": 5e-4}
# the nuisances and what each instrument is trusted to (1 sigma)
NUISANCES = ["b", "xin", "win", "ox", "ow", "gc"]
PRIOR = {"b": 0.3, "xin": 0.001, "win": 0.005, "ox": 0.002, "ow": 0.005, "gc": 0.02}
LABEL = {
    "b": "steam temperature offset (0.3 K)",
    "xin": "feed solids offset (0.001)",
    "win": "feed flow gain (0.5 %)",
    "ox": "outlet density meter offset (0.002)",
    "ow": "outlet flow meter gain (0.5 %)",
    "gc": "condensate meter gain (2 %)",
}
C_PRIOR = 3.0  # each fouling mode, in units of 1e-4: wide
KNOWN = 1e-6  # the prior of an instrument taken as exact, relative to its nominal prior
PARAMS = [f"c{k}" for k in range(MODES)] + NUISANCES
CORE = PARAMS[: MODES + 3]  # those that enter the solver: c, b, xin, win
STEP = {**{f"c{k}": 0.02 for k in range(MODES)}, "b": 0.02, "xin": 1e-4, "win": 1e-3}
LAB_EVERY = 60.0
LAB_IDX = [int(round(t / EVERY)) - 1 for t in np.arange(LAB_EVERY, T_MAX + 1e-9, LAB_EVERY)]
EXCITATIONS = [(0.0, 400.0), (1.0, 400.0), (2.0, 400.0), (4.0, 400.0), (2.0, 100.0), (2.0, 1200.0)]
BASE_EXC = (1.0, 400.0)  # J06's steam record


def forward(core: np.ndarray, exc: tuple[float, float]) -> np.ndarray:
    """Clean outlet solids, outlet flow and condensate flow, ``(3, K)``, for the core parameters."""
    from pinnrl.process import FallingFilmEvaporator

    c, b, xin, win = core[:MODES], core[MODES], core[MODES + 1], core[MODES + 2]
    amp, period = exc
    ev = FallingFilmEvaporator(
        T_s=lambda t: 355.15 + amp * math.sin(2 * math.pi * t / period) - b,
        R_f=lambda t: C_SCALE * float(legendre.legval(2.0 * t / T_MAX - 1.0, c)),
        X_in=0.1 + xin,
        W_in=0.1 * (1.0 + win),
    )
    ref = ev.reference(t_max=T_MAX, nz=NZ, nt=NT)
    cond = ev.perimeter / ev.latent_heat * np.trapezoid(ref.q[:, 1:], ref.z, axis=0)
    return np.stack([ref.X.u[-1, 1:], ref.W.u[-1, 1:], cond])


def nominal_core() -> np.ndarray:
    return np.concatenate([C_TRUE, np.zeros(3)])


def core_jacobian(exc):
    """``(clean (3, K), d clean / d core (3, K, 9))`` at the nominal point, central differences."""
    x0 = nominal_core()
    clean = forward(x0, exc)
    cols = []
    for j, name in enumerate(CORE):
        h = STEP[name]
        up, dn = x0.copy(), x0.copy()
        up[j] += h
        dn[j] -= h
        cols.append((forward(up, exc) - forward(dn, exc)) / (2 * h))
    return exc, clean, np.stack(cols, axis=-1)


def blocks(clean: np.ndarray, dcore: np.ndarray) -> dict[str, np.ndarray]:
    """The sensitivity of each kind of record to all 12 parameters, ``{kind: (K, 12)}``."""
    K = clean.shape[1]
    n_core = len(CORE)

    def pad(d, extra=None):
        J = np.zeros((K, len(PARAMS)))
        J[:, :n_core] = d
        for name, col in (extra or {}).items():
            J[:, PARAMS.index(name)] = col
        return J

    return {
        "X": pad(dcore[0], {"ox": np.ones(K)}),  # record = X + ox
        "W": pad(dcore[1], {"ow": clean[1]}),  # record = W (1 + ow)
        "C": pad(dcore[2], {"gc": clean[2]}),  # record = C (1 + gc)
        "lab": pad(dcore[0]),  # an unbiased assay of the outlet solids
    }


def prior_std(overrides: dict[str, float] | None = None) -> np.ndarray:
    p = {**PRIOR, **(overrides or {})}
    return np.array([C_PRIOR] * MODES + [p[n] for n in NUISANCES])


def targets() -> dict[str, np.ndarray]:
    """Linear targets on theta: the fouling at T, its mean, and its value at T as a ramp reads it."""
    end = np.zeros(len(PARAMS))
    end[:MODES] = C_SCALE * legendre.legvander(np.array([1.0]), MODES - 1)[0]
    mean = np.zeros(len(PARAMS))
    mean[0] = C_SCALE
    return {"end": end, "mean": mean}


def design_information(B: dict[str, np.ndarray], sensors, lab=()):
    from pinnrl.analysis import information

    F = np.zeros((len(PARAMS), len(PARAMS)))
    for s in sensors:
        F += information(B[s], NOISE[s])
    for i in lab:
        F += information(B["lab"][i], NOISE["lab"])
    return F


def evaluate(B, sensors, lab=(), overrides=None, modes=MODES):
    """Posterior std of the targets (relative to 2e-4 for ``end``) and the information gain."""
    from pinnrl.analysis import information_gain, posterior_covariance, target_std

    ps = prior_std(overrides)
    if modes < MODES:  # a ramp assumed: the higher modes are pinned at zero
        ps[modes:MODES] = 1e-9
    cov = posterior_covariance(design_information(B, sensors, lab), ps)
    tg = targets()
    return {
        "end_rel": target_std(cov, tg["end"]) / 2e-4,
        "mean_rel": target_std(cov, tg["mean"]) / 1e-4,
        "b_std": math.sqrt(cov[PARAMS.index("b"), PARAMS.index("b")]),
        "gain_nats": information_gain(np.diag(ps**2), cov),
        "cov": cov,
    }


# ------------------------------------------------------------------ validation
def fit_dataset(job):
    """One synthetic data set of a design, fitted by Gauss-Newton with the nominal Jacobian."""
    design, seed, J_w, sig, ps, exc, sensors, lab, full = (
        job[k] for k in ("design", "seed", "J_w", "sig", "ps", "exc", "sensors", "lab", "full")
    )
    rng = np.random.default_rng(seed)
    truth = np.concatenate(
        [C_TRUE, [rng.normal(0.0, ps[MODES + i]) for i in range(len(NUISANCES))]]
    )

    def records(theta):
        clean = forward(theta[: len(CORE)], exc)
        ox, ow, gc = theta[len(CORE) :]
        parts = {"X": clean[0] + ox, "W": clean[1] * (1 + ow), "C": clean[2] * (1 + gc)}
        out = [parts[s] for s in sensors]
        if lab:
            out.append(clean[0][list(lab)])
        return np.concatenate(out)

    y = records(truth) + sig * rng.standard_normal(sig.size)
    mean0 = np.concatenate([C_TRUE, np.zeros(len(NUISANCES))])  # the prior's centre
    P0 = np.diag(1.0 / ps**2)
    A = np.linalg.inv(J_w.T @ J_w + P0)
    theta = mean0.copy()
    for _ in range(8):
        r = (y - records(theta)) / sig
        step = A @ (J_w.T @ r - P0 @ (theta - mean0))
        theta = theta + step
        if np.max(np.abs(step) / ps) < 1e-4:
            break
    end = targets()["end"]
    out = {
        "design": design,
        "seed": seed,
        "end": float(end @ theta),
        "end_truth": float(end @ truth),
        "b_error": float(theta[MODES] - truth[MODES]),
    }
    if full:  # the same data by a full nonlinear least-squares fit, scaled by the priors
        from scipy.optimize import least_squares

        def resid(z):
            th = mean0 + z * ps
            return np.concatenate([(records(th) - y) / sig, z])

        res = least_squares(resid, np.zeros(ps.size), diff_step=0.02, xtol=1e-9)
        out["end_full"] = float(end @ (mean0 + res.x * ps))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--datasets", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    from pinnrl.analysis import greedy_design, information, target_std

    tic = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jac = {exc: (clean, d) for exc, clean, d in pool.map(core_jacobian, EXCITATIONS)}
    B = {exc: blocks(*jac[exc]) for exc in EXCITATIONS}
    Bb = B[BASE_EXC]
    print(f"sensitivities for {len(EXCITATIONS)} steam signals in {time.time() - tic:.0f} s")
    online = ("X", "W")
    out: dict = {
        "params": PARAMS,
        "prior": PRIOR,
        "noise": NOISE,
        "lab_times": TIMES[LAB_IDX].tolist(),
    }

    def row(name, B_, sensors, lab=(), overrides=None, **extra):
        e6 = evaluate(B_, sensors, lab, overrides)
        e2 = evaluate(B_, sensors, lab, overrides, modes=2)
        return {
            "name": name,
            "end_rel": e6["end_rel"],
            "mean_rel": e6["mean_rel"],
            "end_rel_ramp": e2["end_rel"],
            "b_std": e6["b_std"],
            "gain_nats": e6["gain_nats"],
            **extra,
        }

    # 1. which instrument error costs the most
    exact = {n: KNOWN * PRIOR[n] for n in NUISANCES}
    rows = [row("all instruments exact", Bb, online, overrides=exact)]
    for n in ["b", "xin", "win", "ox", "ow"]:
        rows.append(row(LABEL[n], Bb, online, overrides={**exact, n: PRIOR[n]}, nuisance=n))
    both = {**exact, "win": PRIOR["win"], "ow": PRIOR["ow"]}
    rows.append(row("feed and outlet flow gains together", Bb, online, overrides=both))
    rows.append(row("all but the steam offset", Bb, online, overrides={"b": exact["b"]}))
    rows.append(row("all admitted (the baseline)", Bb, online))
    out["nuisances"] = rows
    print("\n1. the cost of each instrument error (std of R_f(T), % of the truth; 6 modes | ramp)")
    for r in rows:
        print(f"   {r['name']:<40} {100 * r['end_rel']:6.2f} | {100 * r['end_rel_ramp']:5.2f}")

    # 3. lab samples: when (computed before the remedies, which use the best schedule)
    tg = targets()["end"]
    base_F = design_information(Bb, online)
    cands = [Bb["lab"][i] for i in LAB_IDX]
    lab_rows = []
    greedy = greedy_design(
        cands, NOISE["lab"], 8, prior_std=prior_std(), base_information=base_F, target=tg
    )

    def lab_std(picks):
        return evaluate(Bb, online, [LAB_IDX[i] for i in picks])["end_rel"]

    for k in range(1, 9):
        spaced = (
            [int(round(i)) for i in np.linspace(0, len(LAB_IDX) - 1, k + 2)[1:-1]]
            if k > 1
            else [len(LAB_IDX) // 2]
        )
        spaced_ends = (
            [int(round(i)) for i in np.linspace(0, len(LAB_IDX) - 1, k)]
            if k > 1
            else [len(LAB_IDX) - 1]
        )
        lab_rows.append(
            {
                "k": k,
                "greedy": greedy.criterion[k] / 2e-4,
                "greedy_times": sorted(TIMES[LAB_IDX[i]] for i in greedy.chosen[:k]),
                "interior": lab_std(spaced),
                "with_ends": lab_std(spaced_ends),
                "at_end": lab_std(list(range(len(LAB_IDX) - k, len(LAB_IDX)))),
            }
        )
    best3 = min(itertools.combinations(range(len(LAB_IDX)), 3), key=lab_std)
    out["lab"] = {
        "rows": lab_rows,
        "exhaustive3": lab_std(best3),
        "exhaustive3_times": [TIMES[LAB_IDX[i]] for i in best3],
        "none": greedy.criterion[0] / 2e-4,
    }
    # the same samples when the ramp is assumed, and replicated at the end
    rep = greedy_design(
        cands,
        NOISE["lab"],
        8,
        prior_std=prior_std(),
        base_information=base_F,
        target=tg,
        repeat=True,
    )
    out["lab"]["replicates"] = [c / 2e-4 for c in rep.criterion]
    out["lab"]["replicate_times"] = [TIMES[LAB_IDX[i]] for i in rep.chosen]
    print(
        "\n3. lab samples of the outlet solids (std of R_f(T), %): greedy | interior | with ends | all at the end"
    )
    print(f"   k=0  {100 * out['lab']['none']:.2f}")
    for r in lab_rows:
        print(
            f"   k={r['k']}  {100 * r['greedy']:.2f} | {100 * r['interior']:.2f} | {100 * r['with_ends']:.2f} | {100 * r['at_end']:.2f}"
            f"   greedy at {[int(t) for t in r['greedy_times']]}"
        )
    print(
        f"   exhaustive k=3: {100 * out['lab']['exhaustive3']:.2f} at {out['lab']['exhaustive3_times']}"
    )
    lab3 = [LAB_IDX[i] for i in greedy.chosen[:3]]

    # 2. remedies, one at a time and combined
    remedies = {
        "steam calibrated to 0.1 K": dict(overrides={"b": 0.1}),
        "steam calibrated to 0.03 K": dict(overrides={"b": 0.03}),
        "outlet density meter calibrated to 0.0005": dict(overrides={"ox": 0.0005}),
        "feed flow calibrated to 0.1 %": dict(overrides={"win": 0.001}),
        "outlet flow calibrated to 0.1 %": dict(overrides={"ow": 0.001}),
        "feed solids to 0.0003": dict(overrides={"xin": 0.0003}),
        "3 lab samples of the outlet solids": dict(lab=lab3),
        "condensate meter (gain unknown to 2 %)": dict(sensors=("X", "W", "C")),
        "condensate meter calibrated to 0.5 %": dict(
            sensors=("X", "W", "C"), overrides={"gc": 0.005}
        ),
        "steam swing 2 K (period 400 s)": dict(exc=(2.0, 400.0)),
        "steam swing 4 K (period 400 s)": dict(exc=(4.0, 400.0)),
        "steam swing 2 K, period 100 s": dict(exc=(2.0, 100.0)),
        "steam swing 2 K, period 1200 s": dict(exc=(2.0, 1200.0)),
        "no steam swing at all": dict(exc=(0.0, 400.0)),
    }

    def apply(specs):
        cfg = {"sensors": online, "lab": (), "overrides": {}, "exc": BASE_EXC}
        for s in specs:
            for k, v in remedies[s].items():
                cfg[k] = {**cfg[k], **v} if k == "overrides" else v
        return cfg

    def remedy_row(specs):
        cfg = apply(specs)
        return row(
            " + ".join(specs),
            B[cfg["exc"]],
            cfg["sensors"],
            cfg["lab"],
            cfg["overrides"],
            specs=list(specs),
        )

    rem_rows = [row("baseline", Bb, online, specs=[])] + [remedy_row([s]) for s in remedies]
    combos = [
        ["condensate meter (gain unknown to 2 %)", "steam swing 4 K (period 400 s)"],
        ["condensate meter (gain unknown to 2 %)", "steam swing 2 K, period 100 s"],
        ["condensate meter calibrated to 0.5 %", "steam calibrated to 0.1 K"],
        ["steam calibrated to 0.1 K", "3 lab samples of the outlet solids"],
        [
            "steam calibrated to 0.1 K",
            "feed flow calibrated to 0.1 %",
            "outlet flow calibrated to 0.1 %",
        ],
        [
            "steam calibrated to 0.03 K",
            "feed flow calibrated to 0.1 %",
            "outlet flow calibrated to 0.1 %",
            "3 lab samples of the outlet solids",
        ],
        [
            "condensate meter (gain unknown to 2 %)",
            "steam swing 4 K (period 400 s)",
            "3 lab samples of the outlet solids",
        ],
    ]
    rem_rows += [remedy_row(c) for c in combos]
    out["remedies"] = rem_rows
    print(
        "\n2. remedies (std of R_f(T), %; 6 modes | ramp assumed | std of the steam offset, K | gain, nats)"
    )
    for r in rem_rows:
        print(
            f"   {r['name'][:88]:<88} {100 * r['end_rel']:5.2f} | {100 * r['end_rel_ramp']:5.2f} | {r['b_std']:.3f} | {r['gain_nats']:.1f}"
        )

    # the steam swing: amplitude and period on a finer grid of amplitudes (the response is linear in it)
    out["excitation"] = [
        {
            "amp": a,
            "period": p,
            **{k: v for k, v in row("", B[(a, p)], online).items() if k != "name"},
            "with_condensate": evaluate(B[(a, p)], ("X", "W", "C"))["end_rel"],
        }
        for a, p in EXCITATIONS
    ]

    # 4. validation by simulation
    val_designs = {
        "baseline": [],
        "steam calibrated to 0.1 K": ["steam calibrated to 0.1 K"],
        "condensate meter + 4 K swing": combos[0],
    }
    jobs = []
    predicted = {}
    for name, specs in val_designs.items():
        cfg = apply(specs)
        Bd = B[cfg["exc"]]
        J = np.concatenate(
            [Bd[s] for s in cfg["sensors"]] + ([Bd["lab"][list(cfg["lab"])]] if cfg["lab"] else [])
        )
        sig = np.concatenate(
            [np.full(TIMES.size, NOISE[s]) for s in cfg["sensors"]]
            + ([np.full(len(cfg["lab"]), NOISE["lab"])] if cfg["lab"] else [])
        )
        ps = prior_std(cfg["overrides"])
        predicted[name] = evaluate(Bd, cfg["sensors"], cfg["lab"], cfg["overrides"])["end_rel"]
        assert (
            abs(
                target_std(np.linalg.inv(information(J, sig) + np.diag(1 / ps**2)), tg) / 2e-4
                - predicted[name]
            )
            < 1e-6
        )
        for seed in range(args.datasets):
            jobs.append(
                {
                    "design": name,
                    "seed": seed,
                    "J_w": J / sig[:, None],
                    "sig": sig,
                    "ps": ps,
                    "exc": cfg["exc"],
                    "sensors": cfg["sensors"],
                    "lab": tuple(cfg["lab"]),
                    "full": name == "baseline" and seed < 6,
                }
            )
    tic = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fits = list(pool.map(fit_dataset, jobs, chunksize=4))
    val = []
    for name in val_designs:
        rows_ = [f for f in fits if f["design"] == name]
        err = np.array([f["end"] - f["end_truth"] for f in rows_]) / 2e-4
        half = Z90 * predicted[name]
        entry = {
            "design": name,
            "predicted_std": predicted[name],
            "error_std": float(err.std(ddof=1)),
            "error_mean": float(err.mean()),
            "coverage": float(np.mean(np.abs(err) <= half)),
            "n": len(rows_),
            "errors": err.tolist(),
        }
        full = [abs(f["end_full"] - f["end"]) / 2e-4 for f in rows_ if "end_full" in f]
        if full:
            entry["full_fit_max_difference"] = float(max(full))
        val.append(entry)
    out["validation"] = val
    print(f"\n4. validation, {args.datasets} data sets per design ({time.time() - tic:.0f} s)")
    for v in val:
        extra = (
            f"  (full least squares differs by at most {100 * v['full_fit_max_difference']:.3f} %)"
            if "full_fit_max_difference" in v
            else ""
        )
        print(
            f"   {v['design']:<30} predicted {100 * v['predicted_std']:.2f} %  errors {100 * v['error_std']:.2f} % (mean {100 * v['error_mean']:+.2f})  coverage {v['coverage']:.2f}{extra}"
        )
    (args.out / "j09.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved", args.out / "j09.json")


if __name__ == "__main__":
    main()
