"""K01: Fisher information from the trained surrogate, checked against the exact layer.

For heat (alpha), the two-parameter KdV (alpha, beta) and Burgers (log nu), a
network is trained at the true parameters, its sensitivities du/dtheta at
observation points are computed by the implicit-function theorem
(pinnrl.analysis.surrogate_sensitivities), and the resulting Fisher matrix,
Cramér–Rao bound, correlation and condition number are compared with the ones
from the closed form (heat, KdV) or from finite differences of the Cole–Hopf
reference (Burgers).

Usage: python experiments/K01_fisher_information/run.py [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
MODEL = {"hidden_dims": [64, 64, 64, 64], "layer_norm": False}
TRAIN = dict(
    num_collocation_points=1000,
    batch_size=1000,
    num_boundary_points=512,
    num_initial_points=512,
    learning_rate=1e-3,
    scheduler_type="constant",
    early_stopping={"enabled": False},
)
WEIGHTS = {"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0}
SIGMA = 0.05


def summarize(summary, true_values):
    return {
        "rel_std": summary.relative_std(true_values),
        "correlation": summary.correlation.tolist(),
        "condition_number": summary.condition_number,
        "eigenvalues": summary.eigenvalues.tolist(),
        "eigenvectors": summary.eigenvectors.tolist(),
    }


def heat_case(seed):
    from pinnrl.analysis import closed_form_bound, fisher_information, surrogate_sensitivities
    from pinnrl.pdes.catalog import from_catalog

    torch.manual_seed(seed)
    pde = from_catalog("heat")
    trainer = pde.build_trainer("feedforward", model=MODEL, loss_weights=WEIGHTS, **TRAIN)
    t0 = time.time()
    trainer.train_steps(num_steps=5000, resample_every=50)
    l2 = pde.validate(trainer.model, 4000)["l2_error"]
    inverse = pde  # the trained problem itself; alpha is made trainable for the solve
    cells = []
    gen = torch.Generator().manual_seed(seed)
    layouts = {
        f"random_{n}": (torch.rand(n, 1, generator=gen), torch.rand(n, 1, generator=gen))
        for n in (20, 50, 200, 1000)
    }
    for n_s in (1, 2, 4, 8):
        m = 200 // n_s
        pos = torch.arange(1, n_s + 1, dtype=torch.float32) / (n_s + 1)
        times = (torch.arange(m, dtype=torch.float32) + 0.5) / m
        layouts[f"sensors_{n_s}"] = (pos.repeat_interleave(m)[:, None], times.repeat(n_s)[:, None])
    for name, (x, t) in layouts.items():
        exact = closed_form_bound(inverse, ["alpha"], x, t, SIGMA)
        sur = surrogate_sensitivities(inverse, trainer.model, ["alpha"], x, t)
        approx = fisher_information(sur.jacobian, SIGMA, ["alpha"])
        Je, Js = exact_sensitivities_pair(inverse, x, t, sur.jacobian)
        cells.append(
            {
                "layout": name,
                "exact": summarize(exact, {"alpha": 0.1}),
                "surrogate": summarize(approx, {"alpha": 0.1}),
                "jacobian_rel_error": float((Js - Je).norm() / Je.norm()),
                "cg_residual": sur.cg_residuals[0],
            }
        )
    return {
        "problem": "heat",
        "seed": seed,
        "l2_error": l2,
        "train_time_s": time.time() - t0,
        "cells": cells,
    }


def exact_sensitivities_pair(pde, x, t, jac_surrogate, names=None):
    from pinnrl.analysis import exact_sensitivities

    names = names or ["alpha"]
    Je = exact_sensitivities(pde, names, x, t).reshape(-1, len(names))
    Js = jac_surrogate.reshape(-1, len(names))
    return Je, Js


def kdv_case(seed):
    from pinnrl.analysis import closed_form_bound, fisher_information, surrogate_sensitivities
    from pinnrl.pdes.catalog import from_catalog

    torch.manual_seed(seed)
    pde = from_catalog("kdv_two_parameter")
    # Data-augmented forward training: the soliton collapses without data.
    pde.generate_synthetic_observations(n_points=200, noise_std=0.0, seed=seed)
    trainer = pde.build_trainer(
        "feedforward", model=MODEL, loss_weights=WEIGHTS, mode="data_augmented", **TRAIN
    )
    t0 = time.time()
    trainer.train_steps(num_steps=8000, resample_every=100)
    l2 = pde.validate(trainer.model, 4000)["l2_error"]
    true = {"alpha": 6.0, "beta": 1.0}
    inverse = pde
    cells = []
    gen = torch.Generator().manual_seed(seed)
    for n in (50, 200, 1000):
        x = torch.rand(n, 1, generator=gen) * 30 - 15
        t = torch.rand(n, 1, generator=gen) * 2
        exact = closed_form_bound(inverse, list(true), x, t, SIGMA)
        sur = surrogate_sensitivities(inverse, trainer.model, list(true), x, t)
        approx = fisher_information(sur.jacobian, SIGMA, list(true))
        Je, Js = exact_sensitivities_pair(inverse, x, t, sur.jacobian, list(true))
        cells.append(
            {
                "layout": f"random_{n}",
                "exact": summarize(exact, true),
                "surrogate": summarize(approx, true),
                "jacobian_rel_error": float((Js - Je).norm() / Je.norm()),
                "cg_residual": max(sur.cg_residuals),
            }
        )
    return {
        "problem": "kdv_two_parameter",
        "seed": seed,
        "l2_error": l2,
        "train_time_s": time.time() - t0,
        "cells": cells,
    }


def burgers_case(seed):
    from pinnrl.analysis import fisher_information, surrogate_sensitivities
    from pinnrl.pdes.custom import CustomPDE

    ref = np.load(HERE.parent / "F03_burgers_nu" / "results" / "reference.npz")
    x_grid, t_grid, u_grid, s_grid = ref["x"], ref["t"], ref["u"], ref["du_dnu"]
    nu = 0.01 / math.pi
    rng = np.random.default_rng(seed)
    ii = rng.integers(0, len(x_grid), 200)
    jj = rng.integers(1, len(t_grid), 200)
    obs = {
        "x": torch.tensor(x_grid[ii], dtype=torch.float32)[:, None],
        "t": torch.tensor(t_grid[jj], dtype=torch.float32)[:, None],
        "u": torch.tensor(u_grid[ii, jj], dtype=torch.float32)[:, None],
    }
    spec = dict(
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"log_nu": math.log(nu)},
        initial_condition="-sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        name="Burgers (log nu)",
    )
    torch.manual_seed(seed)
    pde = CustomPDE("u_t + u*u_x - exp(log_nu)*u_xx", observation_data=obs, **spec)
    trainer = pde.build_trainer(
        "feedforward", model=MODEL, loss_weights=WEIGHTS, mode="data_augmented", **TRAIN
    )
    t0 = time.time()
    trainer.train_steps(num_steps=10000, resample_every=100)
    xx, tt = np.meshgrid(x_grid[::8], t_grid[::2], indexing="ij")
    with torch.no_grad():
        pred = trainer.model(
            torch.tensor(np.stack([xx.ravel(), tt.ravel()], 1), dtype=torch.float32)
        ).numpy()[:, 0]
    l2 = float(np.linalg.norm(pred - u_grid[::8, ::2].ravel()) / np.linalg.norm(u_grid[::8, ::2]))
    inverse = pde
    cells = []
    for n in (20, 50, 200, 1000):
        ii = rng.integers(0, len(x_grid), n)
        jj = rng.integers(1, len(t_grid), n)
        x = torch.tensor(x_grid[ii], dtype=torch.float32)[:, None]
        t = torch.tensor(t_grid[jj], dtype=torch.float32)[:, None]
        # d u / d log_nu = nu * du/dnu, from the finite-difference reference.
        Je = torch.tensor(nu * s_grid[ii, jj], dtype=torch.float64)[:, None]
        exact = fisher_information(Je, SIGMA, ["log_nu"])
        sur = surrogate_sensitivities(inverse, trainer.model, ["log_nu"], x, t)
        approx = fisher_information(sur.jacobian, SIGMA, ["log_nu"])
        Js = sur.jacobian.reshape(-1, 1).double()
        cells.append(
            {
                "layout": f"random_{n}",
                "exact": {
                    "rel_std": {"log_nu": float(exact.std[0])},
                    "note": "std of log nu, i.e. relative std of nu",
                },
                "surrogate": {"rel_std": {"log_nu": float(approx.std[0])}},
                "jacobian_rel_error": float((Js - Je).norm() / Je.norm()),
                "cg_residual": sur.cg_residuals[0],
            }
        )
    return {
        "problem": "burgers_log_nu",
        "seed": seed,
        "l2_error": l2,
        "train_time_s": time.time() - t0,
        "cells": cells,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    records = []
    for seed in range(args.seeds):
        for case in (heat_case, kdv_case, burgers_case):
            rec = case(seed)
            records.append(rec)
            print(
                f"{rec['problem']} seed {seed}: l2 {rec['l2_error']:.3f} in {rec['train_time_s']:.0f}s",
                flush=True,
            )
            for c in rec["cells"]:
                print(
                    f"   {c['layout']:<12} J rel err {c['jacobian_rel_error']:.3f}  "
                    f"exact {c['exact']['rel_std']}  surrogate {c['surrogate']['rel_std']}  cg {c['cg_residual']:.2e}",
                    flush=True,
                )
            (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
