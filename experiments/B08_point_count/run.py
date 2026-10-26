"""B08: error vs number of collocation points for uniform and RAD sampling.

N in {250, 500, 1000, 2000, 5000} points per step on Burgers and Allen–Cahn,
3 seeds, 5,000 steps, resampled every 100 steps; log–log efficiency curves.

Usage: python experiments/B08_point_count/run.py [--seeds 3] [--steps 5000] [--workers 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
COUNTS = [250, 500, 1000, 2000, 5000]


def main():
    from pinnrl.benchmarks.sampling_study import StudySettings, run_sampling_study

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    records = []
    t0 = time.time()
    for n in COUNTS:
        settings = StudySettings(
            num_steps=args.steps, n_checkpoints=10, resample_every=100, n_collocation=n
        )
        runs = run_sampling_study(
            ["burgers", "allen_cahn"],
            ["uniform_periodic", "rad"],
            seeds,
            settings,
            workers=args.workers,
        )
        for r in runs:
            d = dataclasses.asdict(r)
            d["n_collocation"] = n
            records.append(d)
        (args.out / "runs.json").write_text(json.dumps(records, indent=1))
        for p in ("burgers", "allen_cahn"):
            for st in ("uniform_periodic", "rad"):
                vals = sorted(r.rel_l2 for r in runs if r.pde == p and r.strategy == st)
                print(
                    f"N {n:>5} {p:<10} {st:<16}: median rel L2 {vals[len(vals) // 2]:.4f}",
                    flush=True,
                )
        print(f"(elapsed {time.time() - t0:.0f}s)", flush=True)
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
