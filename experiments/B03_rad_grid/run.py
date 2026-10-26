"""B03: which RAD hyperparameters work, and are they PDE-dependent?

RAD with k in {0.5, 1, 2, 4} x c in {0, 0.5, 1, 2} (the grid of Wu et al., 2023)
on Burgers and Allen–Cahn, 3 seeds, 10,000 steps, resampled every 100 steps,
plus the uniform-every-100-steps control (k = 0 is uniform by construction).

Usage: python experiments/B03_rad_grid/run.py [--seeds 3] [--steps 10000] [--workers 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
KS = [0.5, 1.0, 2.0, 4.0]
CS = [0.0, 0.5, 1.0, 2.0]


def main():
    from pinnrl.benchmarks.sampling_study import StudySettings, run_sampling_study

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = list(range(args.seeds))
    records = []
    t0 = time.time()

    def record(runs, k, c):
        for r in runs:
            d = dataclasses.asdict(r)
            d["rad_k"], d["rad_c"] = k, c
            records.append(d)
        (args.out / "runs.json").write_text(json.dumps(records, indent=1))

    control = run_sampling_study(
        ["burgers", "allen_cahn"],
        ["uniform_periodic"],
        seeds,
        StudySettings(num_steps=args.steps, n_checkpoints=10, resample_every=100),
        workers=args.workers,
    )
    record(control, 0.0, 0.0)
    print(f"control done (elapsed {time.time() - t0:.0f}s)", flush=True)
    for k in KS:
        for c in CS:
            runs = run_sampling_study(
                ["burgers", "allen_cahn"],
                ["rad"],
                seeds,
                StudySettings(
                    num_steps=args.steps, n_checkpoints=10, resample_every=100, rad_k=k, rad_c=c
                ),
                workers=args.workers,
            )
            record(runs, k, c)
            med = {
                p: sorted(r.rel_l2 for r in runs if r.pde == p)[len(seeds) // 2]
                for p in ("burgers", "allen_cahn")
            }
            print(f"k {k} c {c}: median rel L2 {med} (elapsed {time.time() - t0:.0f}s)", flush=True)
    print("saved", args.out / "runs.json")


if __name__ == "__main__":
    main()
