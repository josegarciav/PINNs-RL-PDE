"""Benchmarks: classical solvers and sampling-strategy harness for pinnrl.

This subpackage provides what the ``pinnrl-benchmark`` CLI exposes:

* :mod:`pinnrl.benchmarks.fdm` — minimal NumPy finite-difference solvers for
  the 1-D heat and wave equations, used as classical baselines for PINN
  accuracy comparisons.
* :mod:`pinnrl.benchmarks.reference` — high-accuracy numerical reference
  solutions (Burgers via Cole-Hopf, Allen-Cahn via method of lines).
* :mod:`pinnrl.benchmarks.sampling_study` — multi-seed comparison of
  collocation samplers (uniform, stratified, periodic uniform, RAD, RL bandit)
  at equal training steps and at equal residual evaluations.
* :mod:`pinnrl.benchmarks.cli` — argparse entry point bound to the
  ``pinnrl-benchmark`` console script.
"""

from pinnrl.benchmarks.fdm import (
    FDMResult,
    solve_heat_1d,
    solve_wave_1d,
)
from pinnrl.benchmarks.reference import (
    ReferenceSolution,
    allen_cahn_reference,
    burgers_reference,
)
from pinnrl.benchmarks.sampling_study import (
    STUDY_PDES,
    STUDY_STRATEGIES,
    StudyRun,
    StudySettings,
    run_sampling_study,
)

__all__ = [
    "FDMResult",
    "ReferenceSolution",
    "STUDY_PDES",
    "STUDY_STRATEGIES",
    "StudyRun",
    "StudySettings",
    "allen_cahn_reference",
    "burgers_reference",
    "run_sampling_study",
    "solve_heat_1d",
    "solve_wave_1d",
]
