"""J08: a soft sensor for the stickiness margin of the powder leaving a spray dryer.

Powder sticks to walls and cakes when the particles are warmer than their
glass transition by more than a material limit (here 20 K). The margin
``T_p - T_g(X)`` cannot be measured online: the glass transition falls by 9 K
for each 0.01 of moisture, and the moisture comes from the lab hours later.
On ``pinnrl.process.SprayDryer`` (J11's tower and instruments) this experiment
compares four estimates of the outlet margin at an operating point:

* ``balance``: the powder moisture from the water balance of the air
  (inlet and outlet humidity, the two flows), the outlet air temperature for T_p;
* ``lab``: the lab assay of the powder with the outlet air temperature;
* ``twin``: the model with the curve's two parameters, the wall loss and the
  droplet size as identified on J11's five-point campaign, run forward at the
  new operating point (no records of that point);
* ``twin+online``: the same, updated with the point's own online records (six
  air thermocouples and the outlet humidity).

1. ``map``: the margin over the operating window, and where it crosses the limit.
2. ``sensors``: the standard deviation of each estimate at four operating points
   (linear-Gaussian, ``pinnrl.analysis``), and of the margin at 5 m, where no
   measurement exists.
3. ``validation``: the twin's margin from J11's 40 fitted campaigns against the truth.
4. ``decision``: at the limit, the probability of calling a sticky point safe.

Usage: python experiments/J08_stickiness/run.py     (needs experiments/J11_dryer_closure/results/j11.json)
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from math import erf, sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "J11_dryer_closure"))
import run as J  # noqa: E402

LIMIT = 20.0  # K above the glass transition at which the powder sticks
POINTS = {  # inlet air (K), solids flow (kg/s)
    "dry and safe": (453.15, 0.009),
    "base": (453.15, 0.010),
    "at the limit": (453.15, 0.011),
    "sticky": (438.15, 0.011),
}
Z_MID = 5.0


def dryer(point, theta):
    """The tower at an operating point for ``theta = [X_cr, n, U_wall rel, d0 rel]``."""
    dr = J.tower()
    drop = replace(dr.droplet, X_cr=theta[0], n=theta[1])
    drop = replace(drop, d0=drop.d0 * (1.0 + theta[3]))
    return replace(
        dr, droplet=drop, T_in=point[0], solids_flow=point[1], U_wall=dr.U_wall * (1 + theta[2])
    )


def margins(point, theta):
    """``[margin at the outlet, margin at 5 m, X_out, T_p,out, T_a,out]``."""
    dr = dryer(point, theta)
    ref = dr.reference()
    m = dr.stickiness_margin(ref)
    return np.array([m[-1], np.interp(Z_MID, ref.z, m), ref.X[-1], ref.T_p[-1], ref.T_a[-1]])


def online_records(point, theta):
    dr = dryer(point, theta)
    rec = dr.records(J.PROBES)
    return np.concatenate([rec["T_air"], rec["Y_out"]])


def main():
    from pinnrl.analysis import information, posterior_covariance, target_std

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    j11 = json.loads((HERE.parent / "J11_dryer_closure" / "results" / "j11.json").read_text())
    dr0 = J.tower()
    x0, h = J.nominal("cdc"), J.steps_for("cdc")
    out: dict = {"limit": LIMIT, "points": POINTS}

    # 1. the map
    T_grid = np.linspace(433.15, 473.15, 17)
    F_grid = np.linspace(0.008, 0.013, 21)
    grid = np.array([[margins((T, F), x0) for F in F_grid] for T in T_grid])
    out["map"] = {
        "T_in": T_grid.tolist(),
        "solids_flow": F_grid.tolist(),
        "margin_out": grid[:, :, 0].tolist(),
        "margin_mid": grid[:, :, 1].tolist(),
        "X_out": grid[:, :, 2].tolist(),
    }
    slope = float(
        (dr0.glass_transition(np.array(0.09)) - dr0.glass_transition(np.array(0.08))) / 0.01
    )
    out["dTg_dX"] = slope
    print(
        f"1. dT_g/dX near the outlet moisture: {slope:.0f} K per unit ({slope / 100:.1f} K per 0.01)"
    )

    # the campaign's posterior (J11: 5 points, plant instruments, nuisances uncertain)
    points5 = J.CAMPAIGNS["5 points"]
    Jc = J.jacobian(lambda th: J.campaign(points5, th, "cdc"), x0, h)
    F_campaign = information(Jc, J.sigma(len(points5)))
    prior = J.prior_for("cdc", False)
    sig_online = np.array([J.NOISE["T_air"]] * len(J.PROBES) + [J.NOISE["Y_out"]])

    rows = []
    print(
        "\n2. std of the estimated margin (K): balance | lab | twin | twin + online ; margin at 5 m: twin | twin + online"
    )
    for name, point in POINTS.items():
        truth = margins(point, x0)
        A = J.jacobian(lambda th: margins(point, th), x0, h)  # noqa: B023  (5, 4)
        Jo = J.jacobian(lambda th: online_records(point, th), x0, h)  # noqa: B023
        cov = posterior_covariance(F_campaign, prior)
        cov_on = posterior_covariance(F_campaign + information(Jo, sig_online), prior)
        dTg = float(
            (dr0.glass_transition(truth[2] + 1e-4) - dr0.glass_transition(truth[2] - 1e-4)) / 2e-4
        )
        G, Fs = dr0.air_flow, point[1]
        # the water balance: X_out = X_0 - G (Y_out - Y_in) / F_s, with both humidities measured
        sx_balance = G / Fs * np.sqrt(2.0) * J.NOISE["Y_out"]
        sT = J.NOISE["T_air"]
        row = {
            "point": name,
            "T_in": point[0],
            "solids_flow": point[1],
            "margin_out": truth[0],
            "margin_mid": truth[1],
            "X_out": truth[2],
            "Tp_minus_Ta_out": truth[3] - truth[4],
            "dTg_dX": dTg,
            "balance": float(np.hypot(dTg * sx_balance, sT)),
            "balance_X_std": float(sx_balance),
            "lab": float(np.hypot(dTg * J.NOISE["X_out"], sT)),
            "twin": target_std(cov, A[0]),
            "twin_online": target_std(cov_on, A[0]),
            "twin_mid": target_std(cov, A[1]),
            "twin_online_mid": target_std(cov_on, A[1]),
            "twin_X_std": target_std(cov, A[2]),
        }
        # 4. the decision: P(estimate says safe | truth exactly at the limit) is 0.5 for any
        # unbiased sensor; the useful number is the guard band for a 5 % miss rate.
        for key in ("balance", "lab", "twin", "twin_online"):
            row[f"guard_{key}"] = 1.6448536269514722 * row[key]
            row[f"p_wrong_{key}"] = 0.5 * (1 - erf(abs(truth[0] - LIMIT) / (row[key] * sqrt(2))))
        rows.append(row)
        print(
            f"   {name:<13} margin {truth[0]:5.1f} K (X_out {truth[2]:.3f})   {row['balance']:5.1f} | {row['lab']:.1f} |"
            f" {row['twin']:.1f} | {row['twin_online']:.1f} ;  5 m ({truth[1]:.0f} K): {row['twin_mid']:.1f} | {row['twin_online_mid']:.1f}"
        )
    out["sensors"] = rows

    # 3. validation with J11's fitted campaigns
    fits = [
        f
        for f in j11["fits"]
        if f["kind"] == "cdc" and f["campaign"] == "5 points" and not f["known"]
    ]
    val = []
    print(f"\n3. the twin's margin from {len(fits)} fitted campaigns (J11) against the truth")
    for name, point in POINTS.items():
        err = np.array(
            [
                margins(point, np.array(f["theta"]))[:2] - margins(point, np.array(f["truth"]))[:2]
                for f in fits
            ]
        )
        pred = next(r for r in rows if r["point"] == name)
        val.append(
            {
                "point": name,
                "rmse_out": float(np.sqrt(np.mean(err[:, 0] ** 2))),
                "bias_out": float(err[:, 0].mean()),
                "rmse_mid": float(np.sqrt(np.mean(err[:, 1] ** 2))),
                "predicted_out": pred["twin"],
                "predicted_mid": pred["twin_mid"],
                "errors_out": err[:, 0].tolist(),
            }
        )
        print(
            f"   {name:<13} outlet rmse {val[-1]['rmse_out']:.2f} K (predicted {pred['twin']:.2f}, bias {val[-1]['bias_out']:+.2f})"
            f"   5 m rmse {val[-1]['rmse_mid']:.2f} K (predicted {pred['twin_mid']:.2f})"
        )
    out["validation"] = val
    print(
        "\n4. guard band for a 5 % chance of calling a sticky point safe (K): balance | lab | twin | twin + online"
    )
    r = rows[1]
    print(
        f"   {r['guard_balance']:.0f} | {r['guard_lab']:.1f} | {r['guard_twin']:.1f} | {r['guard_twin_online']:.1f}"
    )
    (out_dir / "j08.json").write_text(json.dumps(out, indent=1, default=float))
    print("saved", out_dir / "j08.json")


if __name__ == "__main__":
    main()
