"""I15: one value pass for the boundary and initial points, and the registry's classes through the kernel.

Two measurements, both on the trainer's step of I11 (4 x 64 tanh network, 4,096 + 512 + 512 points,
Adam, 1 thread), each the best of rounds with the arms alternating:

1. The catalogue's heat problem with ``derivatives="native"``, with the boundary faces and the
   initial points evaluated in one kernel call against one call per set (the step before I15).
2. The registry's classes (``HeatEquation``, ``KdVEquation``, ``BurgersEquation``,
   ``PendulumEquation``), which did not use the kernel at all, with autograd and with it.

Usage: python experiments/I15_native_value_pass/run.py [--steps 200] [--rounds 5] [--seeds 3]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import torch
import yaml

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)
HERE = Path(__file__).parent
WIDTH, DEPTH = 64, 4
TRAINING = dict(
    num_collocation_points=4096,
    batch_size=4096,
    num_boundary_points=512,
    num_initial_points=512,
    learning_rate=1e-3,
    weight_decay=0.0,
    gradient_clipping=0.0,
    scheduler_type="constant",
    early_stopping={"enabled": False},
    loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
)


def catalogue_trainer(mode, seed, layer_norm, one_pass=True):
    from pinnrl.pdes.catalog import from_catalog

    torch.manual_seed(seed)
    pde = from_catalog("heat")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [WIDTH] * DEPTH, "layer_norm": layer_norm},
        derivatives=mode,
        **TRAINING,
    )
    if not one_pass:
        pde._batched_values = lambda model: {}  # the step before I15
    return pde, trainer


def registry_trainer(key, mode, seed, layer_norm):
    from pinnrl.config import DEFAULT_CONFIG_PATH
    from pinnrl.training.train import build_config_dict, build_model, create_pde
    from pinnrl.training.trainer import PDETrainer

    with open(DEFAULT_CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    cfg = build_config_dict(base, key, "feedforward")
    cfg["architectures"]["feedforward"].update(
        {"hidden_dims": [WIDTH] * DEPTH, "layer_norm": layer_norm, "dropout": 0.0}
    )
    cfg["training"].update({k: v for k, v in TRAINING.items() if k != "early_stopping"})
    cfg["training"]["early_stopping"] = {"enabled": False}
    cfg["model"]["derivatives"] = mode
    device = torch.device("cpu")
    torch.manual_seed(seed)
    pde = create_pde(cfg, device)
    model, config_obj = build_model(cfg, device)
    return pde, PDETrainer(model, pde, config_obj, device=device)


def best_of_rounds(arms, steps, rounds):
    """ms per step of each arm, the arms alternating, the best round kept."""
    best = dict.fromkeys(arms, float("inf"))
    for trainer in arms.values():
        trainer.train_steps(num_steps=20, resample_every=1)
    for _ in range(rounds):
        for name, trainer in arms.items():
            tic = time.perf_counter()
            trainer.train_steps(num_steps=steps, resample_every=1)
            best[name] = min(best[name], 1e3 * (time.perf_counter() - tic) / steps)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    rows = []
    for layer_norm in (False, True):
        for seed in range(args.seeds):
            arms = {
                "separate passes": catalogue_trainer("native", seed, layer_norm, one_pass=False)[1],
                "one pass": catalogue_trainer("native", seed, layer_norm, one_pass=True)[1],
            }
            best = best_of_rounds(arms, args.steps, args.rounds)
            rows.append(
                {
                    "part": "value pass",
                    "problem": "heat (catalogue)",
                    "layer_norm": layer_norm,
                    "seed": seed,
                    **best,
                }
            )
            print(
                f"catalogue heat {'LN' if layer_norm else 'plain'} seed={seed}: "
                + ", ".join(f"{k} {v:.2f}" for k, v in best.items()),
                flush=True,
            )
    for key in ("heat", "kdv", "burgers", "pendulum"):
        for layer_norm in (False, True):
            for seed in range(args.seeds):
                pdes = {}
                arms = {}
                for mode in ("autograd", "native"):
                    pdes[mode], arms[mode] = registry_trainer(key, mode, seed, layer_norm)
                assert pdes["native"].native_active(arms["native"].model), key
                if seed == 0:  # the two arms agree on one batch
                    x, t = pdes["autograd"]._sample_uniform(256)
                    r_a = pdes["autograd"].compute_residual(arms["autograd"].model, x, t)
                    r_n = pdes["native"].compute_residual(arms["autograd"].model, x, t)
                    agree = ((r_a - r_n).norm() / r_a.norm()).item()
                best = best_of_rounds(arms, args.steps, args.rounds)
                rows.append(
                    {
                        "part": "registry",
                        "problem": key,
                        "layer_norm": layer_norm,
                        "seed": seed,
                        "residual_rel": agree,
                        **best,
                    }
                )
                print(
                    f"registry {key:<8} {'LN' if layer_norm else 'plain'} seed={seed}: autograd {best['autograd']:.2f}  native {best['native']:.2f} ms/step  (x{best['autograd'] / best['native']:.2f}, residual agree {agree:.1e})",
                    flush=True,
                )
    (args.out / "steps.json").write_text(json.dumps(rows, indent=1))
    print("saved", args.out / "steps.json")


if __name__ == "__main__":
    main()
