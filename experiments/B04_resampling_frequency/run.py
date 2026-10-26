"""B04: how often should collocation points be resampled?

Uniform and RAD with resampling every 1, 10, 100 and 1,000 steps, and a fixed
set never resampled, on Burgers and Allen–Cahn, 5 seeds, 10,000 steps.
"uniform" resamples every step by construction; the other intervals use
"uniform_periodic" and "rad" with resample_every set accordingly.

Usage: python experiments/B04_resampling_frequency/run.py [--seeds 5] [--steps 10000] [--workers 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
INTERVALS = [1, 10, 100, 1000, 10_000]


def main():
    from pinnrl.benchmarks.sampling_study import StudySettings, run_sampling_study

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    records = []
    t0 = time.time()
    for interval in INTERVALS:
        n_checkpoints = max(1, min(10, args.steps // interval))
        settings = StudySettings(
            num_steps=args.steps, n_checkpoints=n_checkpoints, resample_every=interval
        )
        strategies = ["uniform", "rad"] if interval == 1 else ["uniform_periodic", "rad"]
        if interval == 1:
            settings = StudySettings(num_steps=args.steps, n_checkpoints=10, resample_every=1)
        runs = run_sampling_study(
            ["burgers", "allen_cahn"], strategies, seeds, settings, workers=args.workers
        )
        for r in runs:
            d = dataclasses.asdict(r)
            d["resample_every"] = interval
            records.append(d)
        (args.out / "runs.json").write_text(json.dumps(records, indent=1))
        for p in ("burgers", "allen_cahn"):
            for st in strategies:
                vals = sorted(r.rel_l2 for r in runs if r.pde == p and r.strategy == st)
                print(
                    f"every {interval:>5} {p:<10} {st:<16}: median rel L2 {vals[len(vals) // 2]:.4f}",
                    flush=True,
                )
        print(f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
