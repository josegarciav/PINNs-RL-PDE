"""B02: does the B01 sampler ranking hold on other 1-D equations?

The B01 protocol (uniform every step vs uniform every 100 steps vs RAD k=1, c=1)
on four catalogue equations with a closed form: heat, convection, wave and the
KdV soliton, 5 seeds, 10,000 Adam steps, 1,000 points.

Usage: python experiments/B02_more_pdes/run.py [--seeds 5] [--steps 10000] [--workers 4]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

HERE = Path(__file__).parent
PDES = ["heat", "convection", "wave", "kdv_soliton"]
STRATEGIES = ["uniform", "uniform_periodic", "rad"]


def main():
    from pinnrl.benchmarks.sampling_study import StudySettings, run_sampling_study, summarize

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    settings = StudySettings(num_steps=args.steps, n_checkpoints=20, resample_every=100)
    t0 = time.time()
    runs = run_sampling_study(
        PDES,
        STRATEGIES,
        list(range(args.seeds)),
        settings,
        workers=args.workers,
        on_result=lambda r: print(
            f"{r.pde:<12} {r.strategy:<16} seed {r.seed}: rel L2 {r.rel_l2:.4f} ({r.train_time_s:.0f}s, elapsed {time.time() - t0:.0f}s)",
            flush=True,
        ),
    )
    (args.out / "runs.json").write_text(json.dumps([dataclasses.asdict(r) for r in runs], indent=1))
    rows = [dataclasses.asdict(s) for s in summarize(runs)]
    (args.out / "summary.json").write_text(json.dumps(rows, indent=1))
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
