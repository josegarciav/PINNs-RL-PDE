"""``pinnrl-benchmark`` CLI: classical FDM baseline + sampling-strategy comparison.

Two subcommands:

* ``pinnrl-benchmark fdm --pde {heat,wave}`` — solve with the bundled NumPy
  FDM solver and print L2 / max error against the analytical solution.
* ``pinnrl-benchmark sampling --pde {heat,wave} --strategies ...`` — train
  a small PINN under each requested sampling strategy and print one row per
  strategy with L2 error, final loss, and wall time.

Both subcommands accept ``--csv PATH`` to dump a machine-readable copy of
the same rows alongside the human-readable table.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

from pinnrl.benchmarks.fdm import FDMResult, solve_heat_1d, solve_wave_1d
from pinnrl.benchmarks.sampling import (
    SUPPORTED_STRATEGIES,
    SamplingResult,
    run_sampling_benchmark,
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
    strategies: List[str] = args.strategies or list(SUPPORTED_STRATEGIES)
    unknown = [s for s in strategies if s not in SUPPORTED_STRATEGIES]
    if unknown:
        print(
            f"Unknown strategies: {unknown}. Supported: {SUPPORTED_STRATEGIES}",
            file=sys.stderr,
        )
        return 2

    results: List[SamplingResult] = run_sampling_benchmark(
        pde_name=args.pde,
        strategies=strategies,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
    )

    headers = ["pde", "strategy", "epochs", "l2_error", "max_error", "final_loss", "wall_time_s"]
    rows = [
        [
            args.pde,
            r.strategy,
            str(r.epochs),
            f"{r.l2_error:.6e}",
            f"{r.max_error:.6e}",
            f"{r.final_loss:.6e}",
            f"{r.wall_time_s:.4f}",
        ]
        for r in results
    ]
    _print_table(headers, rows)
    if args.csv:
        _write_csv(Path(args.csv), headers, rows)
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse tree for ``pinnrl-benchmark``."""
    parser = argparse.ArgumentParser(
        prog="pinnrl-benchmark",
        description=(
            "Run pinnrl reference benchmarks: FDM baselines for heat / wave, "
            "and PINN sampling-strategy comparisons (uniform vs stratified vs "
            "residual-based vs RL-adaptive)."
        ),
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    fdm = sub.add_parser("fdm", help="Run the bundled FDM solver and report error vs analytical.")
    fdm.add_argument("--pde", choices=("heat", "wave"), required=True)
    fdm.add_argument("--alpha", type=float, default=0.1, help="Heat diffusivity (heat only).")
    fdm.add_argument("--frequency", type=float, default=1.0, help="Initial-condition frequency (heat only).")
    fdm.add_argument("--c", type=float, default=1.0, help="Wave speed (wave only).")
    fdm.add_argument("--t-max", type=float, default=1.0)
    fdm.add_argument("--nx", type=int, default=101)
    fdm.add_argument("--nt", type=int, default=4001)
    fdm.add_argument("--csv", default=None, help="Optional path to write CSV output.")
    fdm.set_defaults(func=_fdm_command)

    smp = sub.add_parser(
        "sampling",
        help="Train a small PINN under each strategy and report L2 errors.",
    )
    smp.add_argument("--pde", choices=("heat", "wave"), required=True)
    smp.add_argument(
        "--strategies",
        nargs="+",
        choices=SUPPORTED_STRATEGIES,
        default=None,
        help=f"Subset of {SUPPORTED_STRATEGIES}. Defaults to all four.",
    )
    smp.add_argument("--epochs", type=int, default=200)
    smp.add_argument("--batch-size", type=int, default=256)
    smp.add_argument("--lr", type=float, default=5e-3)
    smp.add_argument("--seed", type=int, default=0)
    smp.add_argument("--csv", default=None, help="Optional path to write CSV output.")
    smp.set_defaults(func=_sampling_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse argv and dispatch to the requested subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
