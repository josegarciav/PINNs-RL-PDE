"""K03: is PINN training ill-conditioned by the differential operator?

Eight catalogue problems spanning operator orders (first-order convection,
second-order heat / wave / Klein–Gordon / Poisson, nonlinear Burgers and
Allen–Cahn, the fourth-order beam) x two architectures (tanh MLP, Fourier
features). After 3,000 Adam steps: the extreme eigenvalues of the physics-loss
Hessian (Lanczos), the tangent-kernel spectra of the residual, boundary and
initial terms on 96 points each, and the relative L2 error against the closed
form where one exists.

Usage: python experiments/K03_training_conditioning/run.py [--seeds 2] [--steps 3000]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
PROBLEMS = [
    "convection",
    "heat",
    "wave",
    "klein_gordon",
    "poisson",
    "burgers_travelling_wave",
    "allen_cahn",
    "beam",
]
ARCHS = ["feedforward", "fourier"]
MODEL = {"hidden_dims": [64, 64, 64, 64], "layer_norm": False}
N_KERNEL = 96


def kernel_points(pde, n, gen):
    """Interior, boundary and initial points for the tangent kernels."""
    d = pde.dimension
    lo = torch.tensor([a for a, _ in pde.domain[:d]], dtype=torch.float32)
    hi = torch.tensor([b for _, b in pde.domain[:d]], dtype=torch.float32)
    t0, t1 = pde.time_domain
    x = lo + (hi - lo) * torch.rand(n, d, generator=gen)
    t = t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)
    xb = x.clone()
    side = torch.rand(n, generator=gen) < 0.5
    xb[:, 0] = torch.where(side, lo[0], hi[0])
    x0 = lo + (hi - lo) * torch.rand(n, d, generator=gen)
    return x, t, xb, t.clone(), x0, torch.full((n, 1), float(t0))


def run_case(key, arch, seed, steps):
    from pinnrl.analysis import loss_hessian_spectrum, tangent_kernel_spectrum
    from pinnrl.pdes.catalog import from_catalog

    torch.manual_seed(seed)
    pde = from_catalog(key)
    trainer = pde.build_trainer(
        arch,
        model=MODEL if arch == "feedforward" else {"hidden_dims": [64, 64, 64, 64]},
        num_collocation_points=1000,
        batch_size=1000,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    t0 = time.time()
    out = trainer.train_steps(num_steps=steps, resample_every=100)
    model = trainer.model
    model.eval()
    train_time = time.time() - t0
    l2 = pde.validate(model, 4000)["l2_error"] if pde.has_exact_solution else None

    # Loss Hessian of the physics loss on a fixed collocation sample.
    gen = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        xc, tc = pde.generate_collocation_points(1000)
    params = [p for p in model.parameters() if p.requires_grad]

    def physics_loss():
        losses = pde.compute_loss(model, xc, tc)
        return losses["residual"] + 10.0 * losses["boundary"] + 10.0 * losses["initial"]

    t1 = time.time()
    hess = loss_hessian_spectrum(physics_loss, params, iterations=60, seed=seed)
    hess_time = time.time() - t1

    # Tangent kernels of the three terms.
    x, t, xb, tb, x0, t00 = kernel_points(pde, N_KERNEL, gen)
    kernels = {}
    kernels["residual"] = tangent_kernel_spectrum(model, lambda m: pde.compute_residual(m, x, t))
    if pde.has_exact_solution:
        target_b = pde.exact_solution(xb, tb).detach()
        kernels["boundary"] = tangent_kernel_spectrum(
            model, lambda m: m(torch.cat([xb, tb], 1)) - target_b
        )
    else:
        kernels["boundary"] = tangent_kernel_spectrum(model, lambda m: m(torch.cat([xb, tb], 1)))
    if pde.initial_condition is not None and pde.time_dependent:
        kernels["initial"] = tangent_kernel_spectrum(model, lambda m: m(torch.cat([x0, t00], 1)))
    record = {
        "problem": key,
        "arch": arch,
        "seed": seed,
        "steps": steps,
        "final_loss": float(out["loss"][-1]),
        "l2_error": l2,
        "train_time_s": train_time,
        "hessian": {
            "largest": hess.largest,
            "smallest": hess.smallest,
            "condition_number": hess.condition_number(),
            "ritz": hess.eigenvalues.tolist(),
            "time_s": hess_time,
        },
        "kernels": {
            name: {
                "trace": k.trace,
                "largest": float(k.eigenvalues[0]),
                "condition_number": k.condition_number(1e-8 * float(k.eigenvalues[0])),
                "eigenvalues": k.eigenvalues.tolist(),
            }
            for name, k in kernels.items()
        },
    }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    records = []
    t0 = time.time()
    for seed in range(args.seeds):
        for key in PROBLEMS:
            for arch in ARCHS:
                rec = run_case(key, arch, seed, args.steps)
                records.append(rec)
                h, k = rec["hessian"], rec["kernels"]
                ratio = (
                    k["residual"]["trace"] / k["boundary"]["trace"]
                    if k["boundary"]["trace"]
                    else float("nan")
                )
                print(
                    f"{key:<24} {arch:<11} seed {seed}: l2 {rec['l2_error'] if rec['l2_error'] is None else round(rec['l2_error'], 3)} "
                    f"loss {rec['final_loss']:.2e}  H: max {h['largest']:.3g} min {h['smallest']:.3g} kappa {h['condition_number']:.3g}  "
                    f"NTK: tr(res)/tr(bc) {ratio:.3g} kappa(res) {k['residual']['condition_number']:.3g}  "
                    f"({rec['train_time_s']:.0f}s + {h['time_s']:.0f}s, elapsed {time.time() - t0:.0f}s)",
                    flush=True,
                )
                (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
