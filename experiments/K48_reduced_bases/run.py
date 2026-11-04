"""K48: reduced bases for field unknowns. Which modes of alpha(x) do the data leave free?

The F09 fields (a centred bump, a two-layer medium) on the heat equation with
200 observations, written in a cosine basis alpha(x) = sum_k c_k cos(k pi x),
k = 0..n-1.

Part A, Fisher (seconds): the sensitivity of the observations to each mode
coefficient, by central differences of the method-of-lines reference around
the truth, gives F = J^T J / sigma^2: its eigenvalues (the spectrum of what
the data determine), the Cramér-Rao bound per mode, and the eigenvectors of
the null space (the mode combinations the data leave free). Also the best
n-mode projection of each truth, so the basis error is known.

Part B, swarm (minutes): a cloud particle swarm over the first 4 coefficients
with the reference solver as the forward model, sampling the likelihood-ratio
region of the data misfit at 90 %; the spread of the samples along each Fisher
eigenvector against the Fisher prediction (K46 on a field).

Part C, training (10,000 steps per run): alpha(x) identified as a 6-mode
cosine basis against the F09 network, 3 seeds, noise 0.01; the recovered
coefficients against the truth's projection and the per-mode bound.

Usage: python experiments/K48_reduced_bases/run.py [--seeds 3] [--steps 10000]
       [--workers 3] [--skip-training]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
MODES = 8
TRAIN_MODES = 6
SWARM_MODES = 4
N_OBS = 200
NOISE = 0.01
TRUTHS = {
    "bump": (
        "0.05 + 0.1*exp(-((x - 0.5)/0.15)^2)",
        lambda x: 0.05 + 0.1 * np.exp(-(((x - 0.5) / 0.15) ** 2)),
    ),
    "layers": (
        "0.02 + 0.18*sigmoid((x - 0.5)/0.05)",
        lambda x: 0.02 + 0.18 / (1 + np.exp(-(x - 0.5) / 0.05)),
    ),
}


def cosine_modes(x, n):
    k = np.arange(n)
    return np.cos(np.outer(x, k) * np.pi)  # (len(x), n)


def observation_points(seed: int):
    """The F09 observation layout: 200 uniform points, seeded."""
    import torch

    gen = torch.Generator().manual_seed(seed)
    return torch.rand(N_OBS, 1, generator=gen).numpy().ravel(), torch.rand(
        N_OBS, 1, generator=gen
    ).numpy().ravel()


def solve(alpha_fn):
    from pinnrl.benchmarks import heat_variable_reference

    return heat_variable_reference(lambda x, t: alpha_fn(x), nx=101, nt=101, refine=2)


def _perturbed(truth, k: int, eps: float):
    """``truth + eps cos(k pi x)`` as a function of ``x``."""

    def alpha(x):
        return truth(x) + eps * np.cos(k * np.pi * x)

    return alpha


def mode_jacobian(truth, xo, to, n_modes: int, eps: float = 0.002):
    """``d u(x_o, t_o) / d c_k`` by central differences of the reference, ``(N_OBS, n_modes)``."""
    J = np.zeros((len(xo), n_modes))
    for k in range(n_modes):
        plus = solve(_perturbed(truth, k, eps)).interpolate(xo, to)
        minus = solve(_perturbed(truth, k, -eps)).interpolate(xo, to)
        J[:, k] = (plus - minus) / (2 * eps)
    return J


def _basis_alpha(c):
    """The positive field of coefficients ``c`` in the swarm's cosine basis."""

    def alpha(x):
        return np.maximum(cosine_modes(x, SWARM_MODES) @ c, 1e-3)

    return alpha


def _misfit_for(data, xo, to):
    def misfit(C):
        vals = []
        for c in np.atleast_2d(C):
            vals.append(
                float(np.sum((solve(_basis_alpha(c)).interpolate(xo, to) - data) ** 2) / NOISE**2)
            )
        return np.asarray(vals)

    return misfit


def fisher_part(seed: int = 0) -> dict:
    xo, to = observation_points(seed)
    out = {}
    for name, (_, truth) in TRUTHS.items():
        ref = solve(truth)
        u0 = ref.interpolate(xo, to)
        J = mode_jacobian(truth, xo, to, MODES)
        F = J.T @ J / NOISE**2
        evals, evecs = np.linalg.eigh(F)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        cov = np.linalg.pinv(F)
        cr = np.sqrt(np.diag(cov))
        grid = np.linspace(0, 1, 201)
        Phi = cosine_modes(grid, MODES)
        coef_truth = np.linalg.lstsq(Phi, truth(grid), rcond=None)[0]
        proj_err = [
            float(
                np.linalg.norm(
                    Phi[:, :n] @ np.linalg.lstsq(Phi[:, :n], truth(grid), rcond=None)[0]
                    - truth(grid)
                )
                / np.linalg.norm(truth(grid))
            )
            for n in range(1, MODES + 1)
        ]
        out[name] = {
            "eigenvalues": evals.tolist(),
            "eigenvectors": evecs.T.tolist(),  # row i = eigenvector i
            "cr_std_per_mode": cr.tolist(),
            "truth_coefficients": coef_truth.tolist(),
            "cr_rel_to_truth_coefficient": (cr / np.maximum(np.abs(coef_truth), 1e-6)).tolist(),
            "projection_error_vs_modes": proj_err,
            "sensitivity_norm_per_mode": np.linalg.norm(J, axis=0).tolist(),
            "u0_norm": float(np.linalg.norm(u0)),
            "condition_number": float(evals[0] / max(evals[-1], 1e-300)),
        }
        print(f"{name}: eigenvalues {np.array2string(evals, precision=1)}", flush=True)
        print(
            f"{name}: CR / |truth coef| {np.array2string(cr / np.maximum(np.abs(coef_truth), 1e-6), precision=2)}",
            flush=True,
        )
        print(
            f"{name}: projection error vs n {np.array2string(np.array(proj_err), precision=3)}",
            flush=True,
        )
    return out


def swarm_part(seed: int = 0) -> dict:
    from pinnrl.uq import sample_equivalence_region

    xo, to = observation_points(seed)
    rng = np.random.default_rng(seed)
    out = {}
    for name, (_, truth) in TRUTHS.items():
        data = solve(truth).interpolate(xo, to) + NOISE * rng.standard_normal(N_OBS)
        grid = np.linspace(0, 1, 201)
        Phi = cosine_modes(grid, SWARM_MODES)
        c_truth = np.linalg.lstsq(Phi, truth(grid), rcond=None)[0]

        misfit = _misfit_for(data, xo, to)
        bounds = [(0.02, 0.25)] + [(-0.12, 0.12)] * (SWARM_MODES - 1)
        t0 = time.time()
        sample = sample_equivalence_region(
            misfit, bounds, sigma=1.0, n_particles=24, n_iterations=60, seed=seed
        )
        acc = sample.positions[sample.accepted]
        # Fisher at the truth in the swarm's basis
        J = mode_jacobian(truth, xo, to, SWARM_MODES)
        F = J.T @ J / NOISE**2
        evals, evecs = np.linalg.eigh(F)
        order = np.argsort(evals)[::-1]
        evals, evecs = evals[order], evecs[:, order]
        # The truth is not in the truncated basis, so the region sits around the best
        # 4-mode fit, not around the truth's projection: centre the comparison there.
        centre = sample.best
        proj = (acc - centre) @ evecs if len(acc) else np.zeros((0, SWARM_MODES))
        misfit_truth = float(misfit(c_truth[None, :])[0])
        from scipy.stats import chi2

        radius = np.sqrt(chi2.ppf(0.9, SWARM_MODES))
        out[name] = {
            "n_evaluations": int(sample.positions.shape[0]),
            "n_accepted": int(acc.shape[0]),
            "best": sample.best.tolist(),
            "best_misfit": float(sample.best_misfit),
            "misfit_at_truth_projection": misfit_truth,
            "truth_coefficients": c_truth.tolist(),
            "best_minus_truth_along_eigenvectors": ((centre - c_truth) @ evecs).tolist(),
            "eigenvalues": evals.tolist(),
            "swarm_halfwidth_along_eigenvectors": (
                np.percentile(np.abs(proj), 90, axis=0).tolist() if len(acc) else []
            ),
            "fisher_halfwidth_along_eigenvectors": (radius / np.sqrt(evals)).tolist(),
            "time_s": time.time() - t0,
        }
        print(
            f"{name}: swarm {out[name]['n_accepted']}/{out[name]['n_evaluations']} accepted in {out[name]['time_s']:.0f}s; "
            f"swarm/Fisher half-widths {np.array2string(np.array(out[name]['swarm_halfwidth_along_eigenvectors']) / np.array(out[name]['fisher_halfwidth_along_eigenvectors']), precision=2)}",
            flush=True,
        )
    return out


def run_cell(job):
    name, model, seed, steps = job
    import numpy as np
    import torch

    from pinnrl import CustomPDE

    torch.set_num_threads(2)
    text, truth = TRUTHS[name]
    ref = solve(truth)
    spec = {"inputs": ["x"], "positive": True, "initial": 0.1, "truth": text}
    if model == "basis":
        spec["basis"] = {"type": "cosine", "modes": TRAIN_MODES}
    else:
        spec["hidden"] = [32, 32]
    pde = CustomPDE(
        "u_t - alpha_x*u_x - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=ref.as_exact_solution(),
        unknown_functions={"alpha": spec},
        name=f"heat alpha(x) {name}",
    )
    pde.generate_synthetic_observations(n_points=N_OBS, noise_std=NOISE, seed=seed)
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
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
    )
    torch.manual_seed(seed)
    trainer.model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    t0 = time.time()
    trainer.train_steps(num_steps=steps, resample_every=50)
    fn = pde.unknown_functions["alpha"]
    grid = torch.linspace(0, 1, 101).reshape(-1, 1)
    with torch.no_grad():
        profile = fn({"x": grid}).ravel().numpy()
    truth_grid = truth(grid.ravel().numpy())
    rec = {
        "truth": name,
        "model": model,
        "seed": seed,
        "steps": steps,
        "alpha_rel_l2": float(np.linalg.norm(profile - truth_grid) / np.linalg.norm(truth_grid)),
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "train_time_s": time.time() - t0,
        "profile": profile.tolist(),
    }
    if model == "basis":
        rec["coefficients"] = fn.coefficients.detach().numpy().tolist()  # pre-softplus coefficients
    return rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-swarm", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    (args.out / "fisher.json").write_text(json.dumps(fisher_part(), indent=1))
    if not args.skip_swarm:
        (args.out / "swarm.json").write_text(json.dumps(swarm_part(), indent=1))
    if args.skip_training:
        return
    jobs = [
        (name, model, seed, args.steps)
        for seed in range(args.seeds)
        for name in TRUTHS
        for model in ("basis", "network")
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_cell, jobs), 1):
            records.append(rec)
            print(
                f"[{i}/{len(jobs)}] {rec['truth']:<7} {rec['model']:<8} seed={rec['seed']} alpha_err={rec['alpha_rel_l2']:.3f} "
                f"l2={rec['l2_error']:.3f} {rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
