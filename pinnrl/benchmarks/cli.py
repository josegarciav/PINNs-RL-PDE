"""``pinnrl-benchmark`` CLI: classical FDM baseline + equal-compute sampling study.

Two subcommands:

* ``pinnrl-benchmark fdm --pde {heat,wave}`` — solve with the bundled NumPy
  FDM solver and print L2 / max error against the analytical solution
  (``--csv PATH`` writes the row to a file).
* ``pinnrl-benchmark sampling`` — the multi-seed sampling study of
  :mod:`pinnrl.benchmarks.sampling_study` on Burgers and Allen-Cahn at equal
  training steps; prints the summary at equal steps and at equal residual
  evaluations, and writes ``runs.csv``, ``summary.csv``, ``curves.json`` and
  ``settings.json`` to ``--out``.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, Sequence

from pinnrl.benchmarks.fdm import FDMResult, solve_heat_1d, solve_wave_1d
from pinnrl.benchmarks.sampling_study import (
    STUDY_PDES,
    STUDY_STRATEGIES,
    StudySettings,
    run_sampling_study,
    write_study_outputs,
)


def _print_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Emit a fixed-width, header-aligned table to stdout."""
    rows = list(rows)
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    fmt = "  ".join("{:<" + str(w) + "}" for w in widths)
    print(fmt.format(*headers))
    print(fmt.format(*["-" * w for w in widths]))
    for row in rows:
        print(fmt.format(*row))


def _write_csv(path: Path, headers: Sequence[str], rows: Iterable[Sequence[str]]) -> None:
    """Write the rendered rows to ``path`` for downstream comparison."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f"\nWrote {path}", file=sys.stderr)


def _fdm_command(args: argparse.Namespace) -> int:
    if args.pde == "heat":
        result: FDMResult = solve_heat_1d(
            alpha=args.alpha,
            frequency=args.frequency,
            t_max=args.t_max,
            nx=args.nx,
            nt=args.nt,
        )
        params = f"α={args.alpha}, f={args.frequency}"
    else:
        result = solve_wave_1d(
            c=args.c,
            t_max=args.t_max,
            nx=args.nx,
            nt=args.nt,
        )
        params = f"c={args.c}"

    headers = ["pde", "params", "nx", "nt", "l2_error", "max_error", "wall_time_s"]
    row = [
        args.pde,
        params,
        str(args.nx),
        str(args.nt),
        f"{result.l2_error:.6e}",
        f"{result.max_error:.6e}",
        f"{result.wall_time_s:.4f}",
    ]
    _print_table(headers, [row])
    if args.csv:
        _write_csv(Path(args.csv), headers, [row])
    return 0


def _sampling_command(args: argparse.Namespace) -> int:
    settings = StudySettings(
        num_steps=args.steps,
        n_checkpoints=args.checkpoints,
        n_collocation=args.n_collocation,
        resample_every=args.resample_every,
    )

    def report(run) -> None:
        print(
            f"{run.pde:<10} {run.strategy:<9} seed={run.seed:<3} "
            f"rel_l2={run.rel_l2:.4e} time={run.train_time_s:.0f}s evals={run.residual_evals}",
            file=sys.stderr,
        )

    runs = run_sampling_study(
        pdes=args.pdes,
        strategies=args.strategies,
        seeds=range(args.seeds),
        settings=settings,
        workers=args.workers,
        on_result=report,
    )
    summaries = write_study_outputs(Path(args.out), runs, settings)
    headers = ["budget", "pde", "strategy", "n", "median_rel_l2", "iqr", "time_s", "wins", "p"]
    rows = [
        [
            s.budget,
            s.pde,
            s.strategy,
            str(s.n),
            f"{s.median_rel_l2:.3e}",
            f"[{s.q25_rel_l2:.2e}, {s.q75_rel_l2:.2e}]",
            f"{s.median_time_s:.0f}",
            "-" if s.strategy == "uniform" else f"{s.wins_vs_uniform}/{s.n}",
            "-" if s.strategy == "uniform" else f"{s.p_vs_uniform:.3f}",
        ]
        for s in summaries
    ]
    _print_table(headers, rows)
    print(f"\nWrote {args.out}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree for ``pinnrl-benchmark``."""
    parser = argparse.ArgumentParser(
        prog="pinnrl-benchmark",
        description=(
            "Run pinnrl reference benchmarks: FDM baselines for heat / wave, "
            "and the collocation-sampling study (uniform, stratified, RAD, RL bandit)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fdm = sub.add_parser("fdm", help="Run the bundled FDM solver and report error vs analytical.")
    fdm.add_argument("--pde", choices=("heat", "wave"), required=True)
    fdm.add_argument("--alpha", type=float, default=0.1, help="Heat diffusivity (heat only).")
    fdm.add_argument(
        "--frequency", type=float, default=1.0, help="Initial-condition frequency (heat only)."
    )
    fdm.add_argument("--c", type=float, default=1.0, help="Wave speed (wave only).")
    fdm.add_argument("--t-max", type=float, default=1.0)
    fdm.add_argument("--nx", type=int, default=101)
    fdm.add_argument("--nt", type=int, default=4001)
    fdm.add_argument("--csv", default=None, help="Optional path to write CSV output.")
    fdm.set_defaults(func=_fdm_command)

    smp = sub.add_parser(
        "sampling",
        help="Multi-seed comparison of collocation samplers at equal training budget.",
    )
    smp.add_argument("--pdes", nargs="+", choices=STUDY_PDES, default=list(STUDY_PDES))
    smp.add_argument(
        "--strategies", nargs="+", choices=STUDY_STRATEGIES, default=list(STUDY_STRATEGIES)
    )
    smp.add_argument("--seeds", type=int, default=10, help="Number of seeds (0..N-1).")
    smp.add_argument("--steps", type=int, default=15_000, help="Optimizer steps per run.")
    smp.add_argument("--checkpoints", type=int, default=50, help="Evaluations along each run.")
    smp.add_argument("--n-collocation", type=int, default=1000)
    smp.add_argument("--resample-every", type=int, default=100)
    smp.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    smp.add_argument("--out", default="runs/sampling_study", help="Output directory.")
    smp.set_defaults(func=_sampling_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and dispatch to the requested subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
