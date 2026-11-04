"""D01 / D02: Adam vs L-BFGS vs Adam -> L-BFGS at matched wall-clock time, and the switch point.

Part A (D01, D02). Heat (closed form), Burgers nu = 0.01/pi (Cole-Hopf
reference) and Allen-Cahn (method-of-lines reference), the sampling study's
setups, with a tanh MLP 4 x 64 and 1,000 collocation points refreshed every 50
optimizer steps. Arms: Adam at 1e-3; L-BFGS (full batch, strong Wolfe, 20
inner iterations per step, lr 1); Adam then L-BFGS, switching at a fraction
r in {0.3, 0.5, 0.7, 0.9} of the time budget. Every arm gets the same wall-clock
budget (120 s on two threads) and is scored against the reference at
15, 30, 60 and 120 s. 3 seeds.

Part B (the floor). Heat alpha identified from F01's cell (200 observations,
noise 0.01, alpha started at 0.5): Adam for 5,000 steps vs Adam for 3,500
steps then L-BFGS for 75 steps (about the same number of residual evaluations)
vs L-BFGS alone for 250 steps. 5 seeds. Does the L-BFGS finish lower the 2 %
floor?

Usage: python experiments/D01_optimizer_baseline/run.py [--seeds 3] [--budget 120] [--workers 4]
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)

HERE = Path(__file__).parent
PROBLEMS = ["heat", "burgers", "allen_cahn"]
RATIOS = [0.3, 0.5, 0.7, 0.9]
CHECKPOINTS = [15.0, 30.0, 60.0, 120.0]


def make_trainer(pde, optimizer: str, learning_rate: float, device, extra: dict | None = None):
    """The study's tanh MLP 4 x 64 and a trainer with the requested optimizer."""
    import yaml

    from pinnrl.config import DEFAULT_CONFIG_PATH, Config, ModelConfig, TrainingConfig
    from pinnrl.neural_networks import PINNModel
    from pinnrl.training.trainer import PDETrainer

    with open(DEFAULT_CONFIG_PATH) as f:
        defaults = yaml.safe_load(f)
    training = copy.deepcopy(defaults["training"])
    training.update(
        {
            "optimizer": optimizer,
            "batch_size": 1000,
            "num_collocation_points": 1000,
            "num_boundary_points": 512,
            "num_initial_points": 512,
            "scheduler_type": "constant",
            "early_stopping": {"enabled": False},
            "gradient_clipping": 0.0,
            "loss_weights": {"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 10.0},
        }
    )
    training.setdefault("optimizer_config", {})["learning_rate"] = learning_rate
    training["optimizer_config"]["weight_decay"] = 0.0
    training.update(extra or {})
    config = Config.__new__(Config)
    config.device = device
    config.training = TrainingConfig.from_dict(training)
    config.model = ModelConfig.from_architecture(
        "feedforward",
        {"hidden_dims": [64, 64, 64, 64], "layer_norm": False, "activation": "tanh"},
        input_dim=2,
        output_dim=1,
    )
    config.rl = None
    pde.config.training = config.training
    model = PINNModel(config=config, device=device).to(device)
    return PDETrainer(model=model, pde=pde, config=config, device=device)


def run_forward(job):
    problem, arm, ratio, seed, budget = job
    import torch

    from pinnrl.benchmarks.sampling_study import _build_pde, _relative_l2, build_reference

    torch.set_num_threads(2)
    torch.manual_seed(seed)
    device = torch.device("cpu")
    pde = _build_pde(problem, device)
    reference = build_reference(problem)
    optimizer = "lbfgs" if arm == "lbfgs" else "adam"
    trainer = make_trainer(pde, optimizer, 1.0 if arm == "lbfgs" else 1e-3, device)
    scores = []

    def record(step, clock, model):
        scores.append({"t": clock, "step": step, "rel_l2": _relative_l2(model, reference, device)})

    if arm == "adam_lbfgs":
        switch = ratio * budget
        first = [c for c in CHECKPOINTS if c <= switch]
        out = trainer.train_steps(
            time_budget_s=switch, resample_every=50, checkpoints=first, on_checkpoint=record
        )
        trainer._switch_to_lbfgs()
        for g in trainer.optimizer.param_groups:
            g["lr"] = 1.0
        rest = [c - switch for c in CHECKPOINTS if c > switch]
        t_offset = out["train_time_s"]

        def record_rest(step, clock, model):
            record(out["steps"] + step, t_offset + clock, model)

        out2 = trainer.train_steps(
            time_budget_s=budget - switch,
            resample_every=50,
            checkpoints=rest,
            on_checkpoint=record_rest,
        )
        steps, clock = out["steps"] + out2["steps"], t_offset + out2["train_time_s"]
    else:
        out = trainer.train_steps(
            time_budget_s=budget, resample_every=50, checkpoints=CHECKPOINTS, on_checkpoint=record
        )
        steps, clock = out["steps"], out["train_time_s"]
    return {
        "part": "forward",
        "problem": problem,
        "arm": arm,
        "ratio": ratio,
        "seed": seed,
        "budget_s": budget,
        "steps": steps,
        "train_time_s": clock,
        "final_rel_l2": _relative_l2(trainer.model, reference, device),
        "scores": scores,
    }


def run_floor(job):
    arm, seed = job
    import torch

    from pinnrl.pdes.catalog import from_catalog

    torch.set_num_threads(2)
    device = torch.device("cpu")
    pde = from_catalog(
        "heat",
        parameters={"alpha": 0.1},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": 0.5},
    )
    pde.generate_synthetic_observations(n_points=200, noise_std=0.01, seed=seed)
    optimizer = "lbfgs" if arm == "lbfgs" else "adam"
    trainer = make_trainer(
        pde, optimizer, 1.0 if arm == "lbfgs" else 1e-3, device, {"mode": "inverse"}
    )
    torch.manual_seed(seed)
    trainer.model.apply(lambda m: m.reset_parameters() if hasattr(m, "reset_parameters") else None)
    t0 = time.time()
    if arm == "adam":
        trainer.train_steps(num_steps=5000, resample_every=50)
    elif arm == "lbfgs":
        trainer.train_steps(num_steps=250, resample_every=50)
    else:
        trainer.train_steps(num_steps=3500, resample_every=50)
        trainer._switch_to_lbfgs()
        for g in trainer.optimizer.param_groups:
            g["lr"] = 1.0
        trainer.train_steps(num_steps=75, resample_every=50)
    alpha_hat = pde.get_trainable_parameter_values()["alpha"]
    return {
        "part": "floor",
        "arm": arm,
        "seed": seed,
        "alpha_hat": alpha_hat,
        "rel_error": abs(alpha_hat - 0.1) / 0.1,
        "l2_error": pde.validate(trainer.model, 4000)["l2_error"],
        "train_time_s": time.time() - t0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--floor-seeds", type=int, default=5)
    parser.add_argument("--budget", type=float, default=120.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for seed in range(args.seeds):
        for problem in PROBLEMS:
            jobs.append((problem, "adam", None, seed, args.budget))
            jobs.append((problem, "lbfgs", None, seed, args.budget))
            jobs += [(problem, "adam_lbfgs", r, seed, args.budget) for r in RATIOS]
    floor_jobs = [
        (arm, seed) for seed in range(args.floor_seeds) for arm in ("adam", "adam_lbfgs", "lbfgs")
    ]
    records = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, rec in enumerate(pool.map(run_floor, floor_jobs), 1):
            records.append(rec)
            print(
                f"[floor {i}/{len(floor_jobs)}] {rec['arm']:<10} seed={rec['seed']} alpha={rec['alpha_hat']:.4f} "
                f"rel_err={rec['rel_error']:.4f} l2={rec['l2_error']:.4f} {rec['train_time_s']:.0f}s",
                flush=True,
            )
        for i, rec in enumerate(pool.map(run_forward, jobs), 1):
            records.append(rec)
            curve = " ".join(f"{s['t']:.0f}s:{s['rel_l2']:.3f}" for s in rec["scores"])
            print(
                f"[{i}/{len(jobs)}] {rec['problem']:<10} {rec['arm']:<10} r={rec['ratio']} seed={rec['seed']} "
                f"steps={rec['steps']} {curve} final={rec['final_rel_l2']:.3f} (elapsed {time.time() - t0:.0f}s)",
                flush=True,
            )
    (args.out / "runs.json").write_text(json.dumps(records, indent=1))
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
