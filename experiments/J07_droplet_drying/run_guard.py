"""J07c: does a reference-solver misfit guard catch the bad closures of J07b?

For every closure J07b produced (``results/closure2.json``: the pointwise
start, pointwise start + PINN, two-stage, Legendre basis; 3 seeds) the droplet
ODE is solved with that closure by the reference solver and the reduced
chi-square of the seed's own mass-and-temperature record is computed
(``DryingDroplet.record_misfit``). The guard accepts a refinement when its
misfit is not above its start's (the pointwise estimate of the same record).
The plant's own curve gives the noise level of the statistic, and the PINN's
field errors (what the network itself reports) are listed next to it. No
training: the closures are read from J07b's results.

Usage: python experiments/J07_droplet_drying/run_guard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import run as J  # noqa: E402


def main():
    runs = json.loads((HERE / "results" / "closure2.json").read_text())
    dr = J.plant()
    sigma_m, sigma_T = J.NOISE_M_REL * dr.m0, J.NOISE_T
    out = []
    start = {}
    for seed in sorted({r["seed"] for r in runs}):
        times, m_rec, T_rec, _ = J.records(seed)
        truth = dr.record_misfit(times, m_rec, T_rec, sigma_m, sigma_T)
        out.append({"arm": "plant curve", "seed": seed, "closure_rel_l2": 0.0, **truth})
        for r in sorted(
            (r for r in runs if r["seed"] == seed), key=lambda r: r["arm"] != "pointwise_only"
        ):
            grid, values = np.array(r["X_grid"]), np.array(r["f_hat"])
            chi = dr.record_misfit(
                times,
                m_rec,
                T_rec,
                sigma_m,
                sigma_T,
                closure=lambda X, g=grid, v=values: np.interp(X, g, v),
            )
            if r["arm"] == "pointwise_only":
                start[seed] = chi["total"]
            out.append(
                {
                    "arm": r["arm"],
                    "seed": seed,
                    "closure_rel_l2": r["closure_rel_l2"],
                    "pinn_rel_l2_X": r.get("rel_l2_X"),
                    **chi,
                }
            )
    for row in out:
        row["start"] = start[row["seed"]]
        row["accepted"] = bool(row["total"] <= start[row["seed"]])
    (HERE / "results" / "guard.json").write_text(json.dumps(out, indent=1))
    print(
        f"{'arm':<16}{'seed':>4}{'f rel L2':>10}{'PINN X err':>12}{'chi2 mass':>11}{'chi2 T':>9}{'total':>9}{'start':>8}  verdict"
    )
    for row in out:
        x_err = "" if row.get("pinn_rel_l2_X") is None else f"{row['pinn_rel_l2_X']:.3f}"
        print(
            f"{row['arm']:<16}{row['seed']:>4}{row['closure_rel_l2']:>10.3f}{x_err:>12}{row['mass']:>11.2f}"
            f"{row['temperature']:>9.2f}{row['total']:>9.2f}{row['start']:>8.2f}  {'accept' if row['accepted'] else 'reject'}"
        )


if __name__ == "__main__":
    main()
