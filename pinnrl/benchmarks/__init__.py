"""Benchmarks: classical solvers and sampling-strategy harness for pinnrl.

This subpackage provides three things the ``pinnrl-benchmark`` CLI exposes:

* :mod:`pinnrl.benchmarks.fdm` — minimal NumPy finite-difference solvers for
  the 1-D heat and wave equations, used as classical baselines for PINN
  accuracy comparisons.
* :mod:`pinnrl.benchmarks.sampling` — runs a short PINN training under a
  chosen collocation sampling strategy (``uniform`` / ``stratified`` /
  ``residual_based`` / ``adaptive``) and reports the L2 error against the
  analytical solution.
* :mod:`pinnrl.benchmarks.cli` — argparse entry point bound to the
  ``pinnrl-benchmark`` console script.
"""

from pinnrl.benchmarks.fdm import (
    FDMResult,
    solve_heat_1d,
    solve_wave_1d,
)
from pinnrl.benchmarks.sampling import (
    SamplingResult,
    run_sampling_benchmark,
)

__all__ = [
    "FDMResult",
    "SamplingResult",
    "solve_heat_1d",
    "solve_wave_1d",
    "run_sampling_benchmark",
]
