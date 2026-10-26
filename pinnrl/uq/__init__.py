"""Uncertainty quantification for identified physics.

Three ways to put an interval on a parameter a PINN identified, and the
metrics that say whether the interval is honest:

- :func:`laplace_posterior`: a Gaussian from the surrogate's Fisher
  information at the estimate (linear; valid where the surrogate resolves
  the solution).
- :func:`sample_equivalence_region`: the set of parameters whose misfit is
  within the noise tolerance, sampled by an exploratory particle swarm
  (nonlinear; needs a cheap forward map).
- Deep ensembles, which are a training loop over seeds; :func:`ensemble_summary`
  turns their estimates into an interval and a dispersion diagnostic.
- :mod:`pinnrl.uq.metrics`: coverage, width and the CRPS.

See the theory pages on the Bayesian view and on equivalence regions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from pinnrl.uq.laplace import LaplacePosterior, laplace_posterior, likelihood_weights
from pinnrl.uq.metrics import (
    coverage,
    crps_gaussian,
    crps_samples,
    gaussian_interval,
    mean_width,
    sample_interval,
)
from pinnrl.uq.swarm import (
    SwarmSample,
    chi2_quantile,
    misfit_tolerance,
    sample_equivalence_region,
    second_order_stable,
)


@dataclass
class EnsembleSummary:
    """Point estimates of an ensemble and what they imply.

    Attributes:
        estimates: ``(n_members, p)``.
        mean: Per parameter.
        std: Per parameter (the ensemble spread).
        dispersion: Median distance of the members to their median, relative
            to the mean absolute value of the parameters (a collapse diagnostic).
    """

    estimates: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    dispersion: float

    def interval(self, level: float = 0.9, gaussian: bool = False) -> np.ndarray:
        """Central interval per parameter, ``(2, p)``: empirical, or Gaussian from mean and std."""
        if gaussian:
            from pinnrl.uq.metrics import _normal_quantile

            z = _normal_quantile(0.5 + level / 2)
            return np.stack([self.mean - z * self.std, self.mean + z * self.std])
        return np.percentile(self.estimates, [50 * (1 - level), 50 * (1 + level)], axis=0)


def ensemble_summary(estimates: Sequence[Sequence[float]]) -> EnsembleSummary:
    """Summarize the estimates of an ensemble of independently trained members."""
    e = np.atleast_2d(np.asarray(estimates, dtype=float))
    med = np.median(e, axis=0)
    scale = float(np.mean(np.abs(med))) or 1.0
    return EnsembleSummary(
        estimates=e,
        mean=e.mean(axis=0),
        std=e.std(axis=0, ddof=1) if e.shape[0] > 1 else np.zeros(e.shape[1]),
        dispersion=float(np.median(np.linalg.norm(e - med, axis=1))) / scale,
    )


__all__ = [
    "EnsembleSummary",
    "LaplacePosterior",
    "SwarmSample",
    "chi2_quantile",
    "coverage",
    "crps_gaussian",
    "crps_samples",
    "ensemble_summary",
    "gaussian_interval",
    "laplace_posterior",
    "likelihood_weights",
    "mean_width",
    "misfit_tolerance",
    "sample_equivalence_region",
    "sample_interval",
    "second_order_stable",
]
