"""Calibration metrics for intervals and samples: coverage, width and the CRPS."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def coverage(lower: Sequence[float], upper: Sequence[float], truth: Sequence[float]) -> float:
    """Fraction of intervals ``[lower, upper]`` that contain the truth."""
    lo, hi, t = (np.asarray(v, dtype=float) for v in (lower, upper, truth))
    return float(np.mean((lo <= t) & (t <= hi)))


def mean_width(lower: Sequence[float], upper: Sequence[float]) -> float:
    """Mean width of the intervals."""
    return float(np.mean(np.asarray(upper, dtype=float) - np.asarray(lower, dtype=float)))


def gaussian_interval(mean: float, std: float, level: float = 0.9) -> tuple[float, float]:
    """Central interval of a Gaussian at the given level."""
    z = _normal_quantile(0.5 + level / 2)
    return mean - z * std, mean + z * std


def sample_interval(samples: Sequence[float], level: float = 0.9) -> tuple[float, float]:
    """Central empirical interval of a set of samples at the given level."""
    s = np.asarray(samples, dtype=float)
    lo, hi = np.percentile(s, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


def crps_gaussian(mean: float, std: float, truth: float) -> float:
    """Continuous ranked probability score of a Gaussian forecast (Gneiting & Raftery, 2007).

    ``CRPS = sigma [ z (2 Phi(z) - 1) + 2 phi(z) - 1/sqrt(pi) ]`` with ``z = (y - mu) / sigma``.
    Lower is better; it rewards both calibration and sharpness.
    """
    if std <= 0:
        return abs(truth - mean)
    z = (truth - mean) / std
    phi = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    Phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return std * (z * (2 * Phi - 1) + 2 * phi - 1 / math.sqrt(math.pi))


def crps_samples(samples: Sequence[float], truth: float) -> float:
    """CRPS of an empirical forecast: ``E|X - y| - E|X - X'| / 2``."""
    s = np.asarray(samples, dtype=float)
    term1 = float(np.mean(np.abs(s - truth)))
    term2 = float(np.mean(np.abs(s[:, None] - s[None, :])))
    return term1 - 0.5 * term2


def _normal_quantile(p: float) -> float:
    """Inverse of the standard normal CDF by bisection on erf (no SciPy dependency)."""
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


__all__ = [
    "coverage",
    "crps_gaussian",
    "crps_samples",
    "gaussian_interval",
    "mean_width",
    "sample_interval",
]
