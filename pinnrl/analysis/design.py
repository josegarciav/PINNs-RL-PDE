"""Bayesian measurement design in the linear-Gaussian approximation.

Around a nominal operating point, a set of measurements ``y = J theta + noise``
with Gaussian noise and a Gaussian prior on ``theta`` has the posterior
covariance ``(P0^-1 + J^T R^-1 J)^-1``, which does not depend on the records.
The value of a measurement can therefore be computed before it is taken: how
far it shrinks the variance of the quantity one cares about (a *target*
``a^T theta``, such as the fouling at the end of the window), or how much
information it carries about all the parameters (the log-determinant).

``theta`` holds everything the records depend on and that is not known
exactly: the unknowns of interest and the nuisances (instrument offsets and
gains) with the accuracy they are trusted to as their prior. A better
instrument is then a tighter prior, a new sensor a new block of rows, a lab
sample one more row, and all of them are compared in the same currency.

:func:`posterior_covariance` is the formula, :func:`target_std` and
:func:`information_gain` the two criteria, and :func:`greedy_design` picks
``k`` candidates (rows or groups of rows) one at a time, each the one that
improves the criterion most given those already chosen. The sensitivities
``J`` come from a reference solver by finite differences, or from
:func:`~pinnrl.pdes.sensors.sensor_jacobian` for a problem with a closed form.
Experiment J09 applies it to the evaporator.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np


def _whiten(jacobian: np.ndarray, noise: float | np.ndarray) -> np.ndarray:
    J = np.atleast_2d(np.asarray(jacobian, dtype=float))
    s = np.asarray(noise, dtype=float)
    if s.ndim == 0:
        s = np.full(J.shape[0], float(s))
    if s.shape != (J.shape[0],):
        raise ValueError(f"noise must be a scalar or have one entry per row, got {s.shape}")
    if np.any(s <= 0):
        raise ValueError("noise must be positive")
    return J / s[:, None]


def information(jacobian: np.ndarray, noise: float | np.ndarray) -> np.ndarray:
    """The Fisher information ``J^T R^-1 J`` of measurements with independent Gaussian noise.

    Args:
        jacobian: Sensitivities of the records to the parameters, ``(m, p)``.
        noise: Noise standard deviation, a scalar or one per record ``(m,)``.
    """
    Jw = _whiten(jacobian, noise)
    return Jw.T @ Jw


def _prior_precision(prior_std: Sequence[float] | np.ndarray | None, p: int) -> np.ndarray:
    if prior_std is None:
        return np.zeros((p, p))
    s = np.asarray(prior_std, dtype=float)
    if s.shape != (p,):
        raise ValueError(f"prior_std needs one entry per parameter ({p}), got {s.shape}")
    if np.any(s <= 0):
        raise ValueError("prior_std must be positive (use inf for a flat prior)")
    return np.diag(1.0 / s**2)  # inf -> 0: a flat prior


def posterior_covariance(
    information_matrix: np.ndarray, prior_std: Sequence[float] | np.ndarray | None = None
) -> np.ndarray:
    """``(P0^-1 + F)^-1`` for the information ``F`` and independent Gaussian priors.

    Args:
        information_matrix: ``F``, ``(p, p)``; the sum of :func:`information`
            over the measurement blocks of a design.
        prior_std: Prior standard deviation of each parameter, ``inf`` for a
            flat prior; ``None`` for all flat.

    Raises:
        numpy.linalg.LinAlgError: When the design and the prior together leave
            a direction of parameter space undetermined.
    """
    F = np.asarray(information_matrix, dtype=float)
    A = F + _prior_precision(prior_std, F.shape[0])
    scale = np.sqrt(np.clip(np.diag(A), 1e-300, None))  # the parameters differ by many decades
    An = A / np.outer(scale, scale)
    if np.linalg.cond(An) > 1e13:
        raise np.linalg.LinAlgError(
            "the design and the prior leave a direction of the parameters undetermined"
        )
    return np.linalg.inv(An) / np.outer(scale, scale)


def target_std(covariance: np.ndarray, target: np.ndarray) -> float:
    """The standard deviation of ``a^T theta``; for a matrix ``A``, ``sqrt(trace(A C A^T))``."""
    A = np.atleast_2d(np.asarray(target, dtype=float))
    return float(np.sqrt(np.trace(A @ covariance @ A.T)))


def information_gain(prior_covariance: np.ndarray, covariance: np.ndarray) -> float:
    """The expected information gain ``0.5 log(det P0 / det P)`` in nats.

    For a linear-Gaussian model this is the mutual information between the
    parameters and the records, the expected Kullback–Leibler divergence from
    prior to posterior. Both covariances must be proper (finite priors).
    """
    _, a = np.linalg.slogdet(np.asarray(prior_covariance, dtype=float))
    _, b = np.linalg.slogdet(np.asarray(covariance, dtype=float))
    return float(0.5 * (a - b))


@dataclass
class DesignResult:
    """The outcome of :func:`greedy_design`.

    ``chosen`` are the indices of the picked candidates in pick order;
    ``criterion`` the criterion's value before any pick and after each one
    (``k + 1`` entries: the target's standard deviation, or minus the
    log-determinant of the posterior precision for ``"d"``); ``covariance``
    the posterior covariance of the final design.
    """

    chosen: list[int] = field(default_factory=list)
    criterion: list[float] = field(default_factory=list)
    covariance: np.ndarray | None = None


def greedy_design(
    candidates: Sequence[np.ndarray],
    noise: Sequence[float | np.ndarray] | float,
    k: int,
    *,
    prior_std: Sequence[float] | np.ndarray | None = None,
    base_information: np.ndarray | None = None,
    target: np.ndarray | None = None,
    repeat: bool = False,
) -> DesignResult:
    """Pick ``k`` of the candidate measurements, each the best given those already picked.

    Args:
        candidates: One sensitivity block per candidate, ``(m_i, p)`` or
            ``(p,)`` for a single record: a lab sample at one time, a sensor
            with its whole record, a probe position.
        noise: Noise standard deviation of each candidate (a scalar or one per
            row of its block), or one scalar for all.
        k: How many to pick.
        prior_std: Prior standard deviations of the parameters (``inf`` flat).
        base_information: The information of the measurements that are taken
            anyway, ``(p, p)``.
        target: ``a`` (or a matrix ``A``): minimise the variance of
            ``a^T theta`` (c-optimality). Without it, maximise the
            log-determinant of the posterior precision (D-optimality).
        repeat: Whether a candidate may be picked more than once (replicates).

    Returns:
        :class:`DesignResult`. Greedy selection is not guaranteed optimal; for
        the D criterion it is within ``1 - 1/e`` of the best information gain
        (the gain is submodular), and on the problems of J09 it matches
        exhaustive search.

    Raises:
        ValueError: For ``k`` larger than the number of candidates without ``repeat``.
    """
    blocks = [np.atleast_2d(np.asarray(c, dtype=float)) for c in candidates]
    if not blocks:
        raise ValueError("greedy_design needs at least one candidate")
    p = blocks[0].shape[1]
    noises: Sequence[Any] = (
        [noise] * len(blocks)
        if np.isscalar(noise) or isinstance(noise, np.ndarray)
        else cast("Sequence[Any]", noise)
    )
    if len(noises) != len(blocks):
        raise ValueError("noise needs one entry per candidate")
    infos = [information(b, s) for b, s in zip(blocks, noises, strict=True)]
    if k > len(blocks) and not repeat:
        raise ValueError(f"cannot pick {k} of {len(blocks)} candidates without repeat")
    F = np.zeros((p, p)) if base_information is None else np.array(base_information, dtype=float)
    prior = _prior_precision(prior_std, p)

    def score(Fi: np.ndarray) -> float:
        if target is None:
            sign, logdet = np.linalg.slogdet(Fi + prior)
            return float(-logdet) if sign > 0 else float("inf")
        try:
            return target_std(posterior_covariance(Fi + prior), target)
        except np.linalg.LinAlgError:
            return float("inf")

    result = DesignResult(criterion=[score(F)])
    available = set(range(len(blocks)))
    for _ in range(int(k)):
        scores = {i: score(F + infos[i]) for i in sorted(available)}
        best = min(scores, key=lambda i: (scores[i], i))
        result.chosen.append(best)
        result.criterion.append(scores[best])
        F = F + infos[best]
        if not repeat:
            available.discard(best)
    try:
        result.covariance = posterior_covariance(F + prior)
    except np.linalg.LinAlgError:
        result.covariance = None
    return result
