"""K06: does the learned solution respect the PDE's own stability?

Forward heat and wave problems are perturbed in their initial data, boundary
data and a parameter; the change of the trained field is compared with the
change of the exact solution (amplification ratio ||delta u_PINN|| /
||delta u_exact|| and the relative error of the PINN's response).

Heat, u_t = alpha u_xx, u(x,0) = sin(pi x), Dirichlet 0:
  ic:    u(x,0) -> (1 + eps) sin(pi x)            exact: (1 + eps) u
  alpha: alpha -> (1 + eps) alpha                 exact: closed form at the new alpha
  bc:    u(0,t) -> eps                             exact: eps (1 - x) + series
Wave, u_tt = c^2 u_xx, u(x,0) = sin(pi x), u_t(x,0) = 0:
  ic:    amplitude (1 + eps)                       exact: (1 + eps) u
  c:     c -> (1 + eps) c                          exact: closed form

Usage: python experiments/K06_problem_disturbance/run.py [--seeds 3] [--steps 3000] [--workers 4]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
EPS = [0.0, 0.01, 0.05, 0.2]
CASES = [("heat", "ic"), ("heat", "alpha"), ("heat", "bc"), ("wave", "ic"), ("wave", "c")]


def heat_bc_exact(eps, terms=60):
    """u with u(0,t) = eps, u(1,t) = 0, u(x,0) = sin(pi x): eps(1-x) + transient series."""
    import torch

    def exact(x, t):
        u = eps * (1 - x)
        for k in range(1, terms + 1):
            ck = (1.0 if k == 1 else 0.0) - 2 * eps / (k * math.pi)
            u = u + ck * torch.sin(k * math.pi * x) * torch.exp(-0.1 * (k * math.pi) ** 2 * t)
        return u

    return exact


def build(problem, kind, eps):
    from pinnrl.pdes.catalog import from_catalog

    if problem == "heat":
        if kind == "ic":
            return from_catalog(
                "heat",
                initial_condition=f"{1 + eps}*sin(pi*x)",
                exact_solution=f"{1 + eps}*exp(-alpha*pi^2*t)*sin(pi*x)",
            )
        if kind == "alpha":
            return from_catalog("heat", parameters={"alpha": 0.1 * (1 + eps)})
        if kind == "bc":
            return from_catalog(
                "heat",
                boundary_conditions={"dirichlet": {"left": eps, "right": 0.0}},
                exact_solution=heat_bc_exact(eps),
            )
    if problem == "wave":
        if kind == "ic":
            return from_catalog(
                "wave",
                initial_condition=f"{1 + eps}*sin(pi*x)",
                exact_solution=f"{1 + eps}*sin(pi*x)*cos(pi*c*t)",
            )
        if kind == "c":
            base = from_catalog("wave")
            return from_catalog("wave", parameters={"c": base.parameters["c"] * (1 + eps)})
    raise ValueError((problem, kind))


def run_cell(job):
    problem, kind, eps, seed, steps = job
    import torch

    from pinnrl.analysis import stability_ratio

    torch.set_num_threads(2)
    pde = build(problem, kind, eps)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        num_collocation_points=1000,
        batch_size=1000,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    torch.manual_seed(seed)
    trainer.model.apply(_reset)
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=100)
    model = trainer.model.eval()
    gen = torch.Generator().manual_seed(123)
    x = torch.rand(4000, 1, generator=gen)
    t = (
        torch.rand(4000, 1, generator=gen) * (pde.time_domain[1] - pde.time_domain[0])
        + pde.time_domain[0]
    )
    with torch.no_grad():
        u = model(torch.cat([x, t], 1))[:, :1]
    exact = pde.exact_solution(x, t)[:, :1]
    sr = stability_ratio(pde, model, n_points=4096)
    return {
        "problem": problem,
        "kind": kind,
        "eps": eps,
        "seed": seed,
        "steps": steps,
        "l2_error": float((u - exact).norm() / exact.norm()),
        "residual": sr.residual,
        "stability_ratio": sr.ratio,
        "u": u.squeeze(1).tolist(),
        "exact": exact.squeeze(1).tolist(),
        "train_time_s": time.time() - t0,
    }


def _reset(module):
    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    jobs = [(p, k, e, s, args.steps) for s in range(args.seeds) for p, k in CASES for e in EPS]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['problem']:<5} {rec['kind']:<6} eps {rec['eps']:<5} seed {rec['seed']}: "
                f"l2 {rec['l2_error']:.4f} residual {rec['residual']:.2e} C {rec['stability_ratio']:.2f} "
                f"{rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records))
    # Responses: field change between eps and eps = 0 for the same (problem, kind, seed).
    import numpy as np

    rows = []
    for rec in records:
        if rec["eps"] == 0.0:
            continue
        base = next(
            r
            for r in records
            if r["problem"] == rec["problem"]
            and r["kind"] == rec["kind"]
            and r["seed"] == rec["seed"]
            and r["eps"] == 0.0
        )
        du = np.array(rec["u"]) - np.array(base["u"])
        de = np.array(rec["exact"]) - np.array(base["exact"])
        rows.append(
            {
                "problem": rec["problem"],
                "kind": rec["kind"],
                "eps": rec["eps"],
                "seed": rec["seed"],
                "exact_response": float(np.linalg.norm(de) / np.linalg.norm(base["exact"])),
                "pinn_response": float(np.linalg.norm(du) / np.linalg.norm(base["u"])),
                "amplification": float(np.linalg.norm(du) / np.linalg.norm(de)),
                "response_error": float(np.linalg.norm(du - de) / np.linalg.norm(de)),
                "l2_error_base": base["l2_error"],
                "l2_error": rec["l2_error"],
            }
        )
    with open(args.out / "responses.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (args.out / "responses.json").write_text(json.dumps(rows, indent=1))
    print("saved", args.out / "responses.csv")


if __name__ == "__main__":
    main()
