"""K45: can one network u(x, t; theta) trained over the prior range replace a training run per theta?

Part A, accuracy and cost. Three problems with a closed form, each with the
parameters as extra network inputs (``CustomPDE(parametric=...)``):

* heat, alpha in [0.02, 0.5] (the K46 prior);
* KdV soliton, (alpha, beta) in [2, 12] x [0.3, 3], data-augmented with 2,000
  noise-free observations over the box (the soliton collapses to zero without data);
* Burgers travelling wave, log nu in [log 0.05, log 0.5], data-augmented likewise.

Each surrogate (tanh MLP 4 x 64, 4,096 points, Adam 4,000 steps then L-BFGS
100 steps) is scored at five parameter values across the range against the
closed form, next to a dedicated network trained at each value with the same
budget; and the cost of one forward evaluation of 10,000 points is timed.
2 seeds.

Part B, the swarm on the surrogate (K46, second pass). For 10 noisy data sets
of heat (noise 0.05) and KdV (noise 0.01), the likelihood-ratio region sampled
by the swarm with the surrogate's misfit (``pde.surrogate_misfit``) against
the same swarm on the closed-form misfit: best fit, 90 % interval, coverage.

Usage: python experiments/K45_parametric_surrogate/run.py [--seeds 2] [--datasets 10] [--workers 4]
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
ADAM_STEPS, LBFGS_STEPS = 4000, 100
POOL = 4096

PROBLEMS = {
    "heat": dict(
        catalogue="heat",
        parametric={"alpha": (0.02, 0.5)},
        overrides={},
        test_values=[[0.03], [0.1], [0.2], [0.35], [0.48]],
        data=False,
        noise=0.05,
        swarm_bounds=[(0.02, 0.5)],
        truth=[0.1],
    ),
    "kdv": dict(
        catalogue="kdv_two_parameter",
        parametric={"alpha": (2.0, 12.0), "beta": (0.3, 3.0)},
        overrides={},
        test_values=[[3.0, 0.5], [6.0, 1.0], [9.0, 2.0], [4.0, 2.5], [11.0, 0.4]],
        data=True,
        noise=0.01,
        swarm_bounds=[(2.0, 12.0), (0.3, 3.0)],
        truth=[6.0, 1.0],
    ),
    "burgers": dict(
        catalogue="burgers_travelling_wave",
        parametric={"log_nu": (math.log(0.05), math.log(0.5))},
        overrides=dict(
            equations="u_t + u*u_x - exp(log_nu)*u_xx",
            initial_condition="c*(1 - tanh(c*x/(2*exp(log_nu))))",
            exact_solution="c*(1 - tanh(c*(x - c*t)/(2*exp(log_nu))))",
            parameters={"c": 1.0},
        ),
        test_values=[
            [math.log(0.06)],
            [math.log(0.1)],
            [math.log(0.2)],
            [math.log(0.35)],
            [math.log(0.48)],
        ],
        data=True,
        noise=None,
        swarm_bounds=None,
        truth=None,
    ),
}


def build(problem: str, theta=None):
    """The parametric problem, or the fixed-parameter problem at ``theta``."""
    from pinnrl.pdes.catalog import from_catalog

    spec = PROBLEMS[problem]
    names = list(spec["parametric"])
    overrides = dict(spec["overrides"])
    base_params = dict(overrides.pop("parameters", {}))
    if theta is None:
        return from_catalog(
            spec["catalogue"], parameters=base_params, parametric=spec["parametric"], **overrides
        )
    params = {**base_params, **dict(zip(names, [float(v) for v in theta], strict=True))}
    return from_catalog(spec["catalogue"], parameters=params, **overrides)


def trainer_for(pde, problem: str, seed: int):
    import torch

    spec = PROBLEMS[problem]
    extra = {}
    if spec["data"]:
        pde.generate_synthetic_observations(n_points=2000, noise_std=0.0, seed=seed)
        extra = {
            "mode": "data_augmented",
            "loss_weights": {"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
        }
    torch.manual_seed(seed)
    return pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [64, 64, 64, 64], "layer_norm": False},
        num_collocation_points=POOL,
        batch_size=POOL,
        num_boundary_points=512,
        num_initial_points=512,
        learning_rate=1e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        **extra,
    )


def train(trainer):
    trainer.train_steps(num_steps=ADAM_STEPS, resample_every=50)
    trainer._switch_to_lbfgs()
    for g in trainer.optimizer.param_groups:
        g["lr"] = 1.0
    trainer.train_steps(num_steps=LBFGS_STEPS, resample_every=50)


class _AtTheta:
    """The parametric surrogate evaluated at a fixed parameter value, as a model of (x, t)."""

    def __init__(self, model, theta, d):
        import torch

        self.model, self.theta, self.d = model, torch.tensor(theta, dtype=torch.float32), d

    def __call__(self, inp):
        import torch

        th = self.theta.reshape(1, -1).expand(inp.shape[0], -1)
        return self.model(torch.cat([inp[:, : self.d], th, inp[:, self.d :]], dim=1))

    def eval(self):
        return self

    def train(self):
        return self


def error_at(model_fn, problem: str, theta) -> float:
    """Relative L2 error against the closed form at ``theta`` on 4,000 random points."""
    import torch

    pde = build(problem, theta)
    x, t = pde._sample_uniform(4000)
    with torch.no_grad():
        u = model_fn(torch.cat([x, t], dim=1))
        exact = pde.exact_solution(x, t)
    return float(torch.linalg.norm(u - exact) / torch.linalg.norm(exact))


def run_surrogate(job):
    problem, seed = job
    import torch

    torch.set_num_threads(2)
    spec = PROBLEMS[problem]
    pde = build(problem)
    trainer = trainer_for(pde, problem, seed)
    t0 = time.time()
    train(trainer)
    train_time = time.time() - t0
    d = pde._n_physical
    errors = [error_at(_AtTheta(trainer.model, th, d), problem, th) for th in spec["test_values"]]
    xt = torch.rand(10000, pde.config.input_dim)
    t1 = time.time()
    with torch.no_grad():
        for _ in range(10):
            trainer.model(xt)
    eval_ms = 100 * (time.time() - t1)
    torch.save(trainer.model.state_dict(), HERE / "results" / f"surrogate_{problem}_{seed}.pt")
    return {
        "part": "surrogate",
        "problem": problem,
        "seed": seed,
        "train_time_s": train_time,
        "box_rel_l2": pde.validate(trainer.model, 4000)["l2_error"],
        "errors_at_test_values": errors,
        "eval_ms_per_10k_points": eval_ms,
    }


def run_dedicated(job):
    problem, k, seed = job
    import torch

    torch.set_num_threads(2)
    spec = PROBLEMS[problem]
    theta = spec["test_values"][k]
    pde = build(problem, theta)
    trainer = trainer_for(pde, problem, seed)
    t0 = time.time()
    train(trainer)
    return {
        "part": "dedicated",
        "problem": problem,
        "seed": seed,
        "k": k,
        "theta": theta,
        "train_time_s": time.time() - t0,
        "error": error_at(trainer.model, problem, theta),
    }


def swarm_part(problem: str, seed: int, surrogate_path: Path) -> dict:
    import torch

    from pinnrl.uq import sample_equivalence_region

    spec = PROBLEMS[problem]
    truth = np.array(spec["truth"])
    pde = build(problem)
    trainer = trainer_for(pde, problem, 0)  # a trainer only to build the matching network
    trainer.model.load_state_dict(torch.load(surrogate_path))
    model = trainer.model.eval()
    data = build(problem, truth).generate_synthetic_observations(
        n_points=200, noise_std=spec["noise"], seed=100 + seed
    )
    x, t, y = data["x"], data["t"], data["u"]
    xn, tn, yn = x.numpy().T, t.numpy().T, y.numpy().T

    if problem == "heat":

        def exact_misfit(theta):
            return (
                (np.exp(-theta[:, 0:1] * math.pi**2 * tn) * np.sin(math.pi * xn) - yn) ** 2
            ).sum(axis=1)

    else:

        def exact_misfit(theta):
            a, b = theta[:, 0:1], theta[:, 1:2]
            return ((3 / a / np.cosh(0.5 * np.sqrt(1 / b) * (xn - tn)) ** 2 - yn) ** 2).sum(axis=1)

    def surrogate_misfit(theta):
        return pde.surrogate_misfit(model, x, t, y, theta).numpy()

    out = {}
    for name, misfit in (("closed_form", exact_misfit), ("surrogate", surrogate_misfit)):
        t0 = time.time()
        s = sample_equivalence_region(
            misfit,
            spec["swarm_bounds"],
            None,
            sigma=spec["noise"],
            n_particles=30,
            n_iterations=100,
            seed=seed,
        )
        iv = s.interval(0.9)
        out[name] = {
            "best": s.best.tolist(),
            "best_misfit": float(s.best_misfit),
            "interval_lo": iv[0].tolist(),
            "interval_hi": iv[1].tolist(),
            "width": (iv[1] - iv[0]).tolist(),
            "truth_in_interval": bool(np.all((iv[0] <= truth) & (truth <= iv[1]))),
            "n_accepted": int(s.accepted.sum()),
            "time_s": time.time() - t0,
        }
    out.update({"part": "swarm", "problem": problem, "seed": seed, "truth": truth.tolist()})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--datasets", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--skip-dedicated", action="store_true")
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    jobs = [(p, s) for s in range(args.seeds) for p in PROBLEMS]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for rec in pool.map(run_surrogate, jobs):
            records.append(rec)
            print(
                f"surrogate {rec['problem']:<8} seed={rec['seed']} box={rec['box_rel_l2']:.4f} at-values={np.round(rec['errors_at_test_values'], 4)} "
                f"{rec['train_time_s']:.0f}s eval {rec['eval_ms_per_10k_points']:.1f} ms/10k (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
        if not args.skip_dedicated:
            jobs = [
                (p, k, s)
                for s in range(args.seeds)
                for p in PROBLEMS
                for k in range(len(PROBLEMS[p]["test_values"]))
            ]
            for rec in pool.map(run_dedicated, jobs):
                records.append(rec)
                print(
                    f"dedicated {rec['problem']:<8} seed={rec['seed']} theta={np.round(rec['theta'], 3)} error={rec['error']:.4f} "
                    f"{rec['train_time_s']:.0f}s (elapsed {time.time() - t0:.0f}s)",
                    flush=True,
                )
    for problem in ("heat", "kdv"):
        for seed in range(args.datasets):
            rec = swarm_part(problem, seed, args.out / f"surrogate_{problem}_0.pt")
            records.append(rec)
            c, s = rec["closed_form"], rec["surrogate"]
            print(
                f"swarm {problem:<6} data set {seed}: best {np.round(c['best'], 3)} vs {np.round(s['best'], 3)}; width {np.round(c['width'], 4)} vs "
                f"{np.round(s['width'], 4)}; covered {c['truth_in_interval']}/{s['truth_in_interval']}; {c['time_s']:.1f}s vs {s['time_s']:.1f}s",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
