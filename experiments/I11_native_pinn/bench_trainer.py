"""I11: the package's trainer on the same heat problem, with autograd and with the compiled kernel.

``pde.build_trainer(derivatives="native")`` takes the residual's derivatives, and the values at
the initial and boundary points, from ``pinnrl.native``. Timings per step at 1 and 4 threads,
and where the time of a step goes.

Usage: python experiments/I11_native_pinn/bench_trainer.py [--steps 2000] [--seeds 3]
"""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import os
import pstats
import sys
import time
from pathlib import Path

import torch

os.environ.setdefault("TQDM_DISABLE", "1")
logging.disable(logging.WARNING)
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import bench_python as B  # noqa: E402

STAGES = {  # function name fragments -> stage of the step
    "kernel forward": ("native/__init__.py", "forward"),
    "kernel backward": ("native/__init__.py", "backward"),
    "autograd engine": ("run_backward",),
    "tanh": ("torch.tanh",),
    "linear": ("_nn.linear",),
    "sampling": ("_sample_uniform",),
    "optimizer": ("_single_tensor_adam",),
}


def make(mode, seed, layer_norm=False):
    from pinnrl.pdes.catalog import from_catalog

    torch.manual_seed(seed)
    pde = from_catalog("heat")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [B.WIDTH] * B.DEPTH, "layer_norm": layer_norm},
        derivatives=mode,
        num_collocation_points=B.N_COL,
        batch_size=B.N_COL,
        num_boundary_points=B.N_BC,
        num_initial_points=B.N_IC,
        learning_rate=B.LR,
        weight_decay=0.0,
        gradient_clipping=0.0,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": B.W_BC, "initial": B.W_IC},
    )
    return pde, trainer


def stage_times(mode, steps=300):
    """Own time of the functions that dominate a step, ms per step (cProfile)."""
    _, trainer = make(mode, 0)
    trainer.train_steps(num_steps=20, resample_every=1)
    prof = cProfile.Profile()
    prof.enable()
    trainer.train_steps(num_steps=steps, resample_every=1)
    prof.disable()
    stats = pstats.Stats(prof).stats
    out = dict.fromkeys(STAGES, 0.0)
    total = 0.0
    for (filename, _line, name), (_cc, _nc, tottime, _ct, _callers) in stats.items():
        total += tottime
        label = f"{filename}:{name}"
        for stage, fragments in STAGES.items():
            if all(f in label for f in fragments):
                out[stage] += tottime
    out["everything else"] = total - sum(out.values())
    return {k: 1e3 * v / steps for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for threads in (1, 4):
        torch.set_num_threads(threads)
        for mode, layer_norm in (
            ("autograd", False),
            ("native", False),
            ("autograd", True),
            ("native", True),
        ):
            for seed in range(args.seeds):
                pde, trainer = make(mode, seed, layer_norm)
                active = pde.native_active(trainer.model)
                tic = time.perf_counter()
                trainer.train_steps(num_steps=args.steps, resample_every=1)
                ms = 1e3 * (time.perf_counter() - tic) / args.steps
                rows.append(
                    {
                        "impl": f"pinnrl-{mode}" + (" + layer norm" if layer_norm else ""),
                        "threads": threads,
                        "seed": seed,
                        "steps": args.steps,
                        "native_active": active,
                        "ms_per_step": ms,
                        "rel_l2": B.rel_l2(trainer.model),
                    }
                )
                print(
                    f"pinnrl {mode:<8} {'layer norm' if layer_norm else 'plain     '} threads={threads} seed={seed}  {ms:.2f} ms/step  rel L2 {rows[-1]['rel_l2']:.2e}",
                    flush=True,
                )
    torch.set_num_threads(1)
    for mode in ("autograd", "native"):
        st = stage_times(mode)
        rows.append({"impl": f"stages pinnrl-{mode}", "threads": 1, "stages_ms": st})
        print(f"stages {mode}:", {k: round(v, 2) for k, v in st.items()}, flush=True)
    (args.out / "trainer.json").write_text(json.dumps(rows, indent=1))
    print("saved", args.out / "trainer.json")


if __name__ == "__main__":
    main()
