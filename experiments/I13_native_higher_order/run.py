"""I13: the compiled kernel on equations with mixed, third and fourth derivatives.

Before I13 ``pinnrl.native`` carried first and pure second derivatives only, and KdV
(``u_xxx``), the beam (``w_xxxx``), Cahn-Hilliard and any mixed term fell back to autograd. The
kernel now carries general Taylor coefficients. This script checks the residual, loss and weight
gradient of each equation against autograd on one batch, then times the trainer's step with
autograd and with the kernel, plain network and with layer normalization, 1 thread, three seeds,
the best of five rounds of 200 steps.

Usage: python experiments/I13_native_higher_order/run.py [--steps 200] [--rounds 5] [--seeds 3]
Takes about 20 minutes. Timings want an idle machine; the two arms alternate and keep their
best round, which softens a busy neighbour but does not remove it.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import time
from pathlib import Path

import torch

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)
HERE = Path(__file__).parent

WIDTH, DEPTH = 64, 4
N_COL, N_BC, N_IC = 4096, 512, 512

# key -> (label, the derivatives in the residual)
PROBLEMS = {
    "heat": ("heat: u_t, u_xx (I11's problem)", dict(key="heat")),
    "kdv_soliton": ("KdV: u_t, u_x, u_xxx", dict(key="kdv_soliton")),
    "bbm": ("BBM: u_t, u_x, u_xxt (mixed third)", dict(custom="bbm")),
    "beam": ("beam: w_tt, w_xxxx", dict(key="beam")),
    "cahn_hilliard": ("Cahn-Hilliard: u_t, u_x, u_xx, u_xxxx", dict(key="cahn_hilliard")),
}


def problem(name):
    from pinnrl import CustomPDE
    from pinnrl.pdes.catalog import from_catalog

    spec = PROBLEMS[name][1]
    if "key" in spec:
        return from_catalog(spec["key"])
    # Benjamin-Bona-Mahony: u_t + u_x + u u_x - u_xxt = 0, solitary wave c/(cosh^2)
    return CustomPDE(
        "u_t + u_x + u*u_x - u_xxt",
        domain={"x": (-15.0, 15.0), "t": (0.0, 2.0)},
        parameters={"c": 1.5},
        initial_condition="3*(c - 1)/cosh(sqrt(1 - 1/c)/2*x)^2",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="3*(c - 1)/cosh(sqrt(1 - 1/c)/2*(x - c*t))^2",
    )


def make(name, mode, seed, layer_norm):
    torch.manual_seed(seed)
    pde = problem(name)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [WIDTH] * DEPTH, "layer_norm": layer_norm},
        derivatives=mode,
        num_collocation_points=N_COL,
        batch_size=N_COL,
        num_boundary_points=N_BC,
        num_initial_points=N_IC,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    return pde, trainer


def agreement(name, layer_norm):
    """Relative differences of the residual, the loss and the weight gradient on one batch."""
    (pde_a, tr_a), (pde_n, tr_n) = (make(name, m, 0, layer_norm) for m in ("autograd", "native"))
    assert pde_n.native_active(tr_n.model), name
    x, t = pde_a._sample_uniform(N_COL)
    r_a, r_n = pde_a.compute_residual(tr_a.model, x, t), pde_n.compute_residual(tr_n.model, x, t)
    torch.manual_seed(1)
    loss_a = pde_a.compute_loss(tr_a.model, x, t)["total"]
    torch.manual_seed(1)
    loss_n = pde_n.compute_loss(tr_n.model, x, t)["total"]
    loss_a.backward()
    loss_n.backward()
    grad_a = torch.cat([p.grad.flatten() for p in tr_a.model.parameters()])
    grad_n = torch.cat([p.grad.flatten() for p in tr_n.model.parameters()])
    return {
        "residual_rel": ((r_a - r_n).norm() / r_a.norm()).item(),
        "loss_rel": abs(loss_a.item() - loss_n.item()) / abs(loss_a.item()),
        "gradient_rel": ((grad_a - grad_n).norm() / grad_a.norm()).item(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=200, help="steps per round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--problems", nargs="*", default=list(PROBLEMS), choices=list(PROBLEMS))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    rows = []
    for name in args.problems:
        for layer_norm in (False, True):
            agree = agreement(name, layer_norm)
            print(
                f"{name:<14} {'layer norm' if layer_norm else 'plain     '} agreement:",
                {k: f"{v:.1e}" for k, v in agree.items()},
                flush=True,
            )
            for seed in range(args.seeds):
                # the two arms alternate in rounds and keep their best round, so a busy neighbour
                # on the machine inflates both alike and the minimum stays close to the idle time
                arms = {mode: make(name, mode, seed, layer_norm) for mode in ("autograd", "native")}
                best = dict.fromkeys(arms, float("inf"))
                for _, trainer in arms.values():
                    trainer.train_steps(num_steps=20, resample_every=1)  # warm up (compile, pools)
                for _ in range(args.rounds):
                    for mode, (_, trainer) in arms.items():
                        tic = time.perf_counter()
                        trainer.train_steps(num_steps=args.steps, resample_every=1)
                        best[mode] = min(best[mode], 1e3 * (time.perf_counter() - tic) / args.steps)
                for mode, (pde, trainer) in arms.items():
                    rows.append(
                        {
                            "problem": name,
                            "label": PROBLEMS[name][0],
                            "layer_norm": layer_norm,
                            "mode": mode,
                            "seed": seed,
                            "threads": args.threads,
                            "steps": args.steps,
                            "rounds": args.rounds,
                            "native_active": pde.native_active(trainer.model),
                            "ms_per_step": best[mode],
                            **({f"agree_{k}": v for k, v in agree.items()} if seed == 0 else {}),
                        }
                    )
                print(
                    f"  seed={seed}  autograd {best['autograd']:.2f}  native {best['native']:.2f} ms/step",
                    flush=True,
                )
    meta = {"machine": platform.machine(), "torch": torch.__version__, "threads": args.threads}
    path = args.out / "steps.json"
    if path.exists():  # a rerun of some problems replaces their rows and keeps the others
        kept = [
            r for r in json.loads(path.read_text())["rows"] if r["problem"] not in args.problems
        ]
        rows = kept + rows
    path.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
    print("saved", args.out / "steps.json")


if __name__ == "__main__":
    main()
