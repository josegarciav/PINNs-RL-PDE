"""Sampling the region of equivalence with an exploratory particle swarm.

The region of equivalence of an inverse problem is the set of parameters whose
misfit is within the noise tolerance; its samples give intervals, percentiles
and CDFs by counting (Pallero et al., 2018). A particle swarm used in its
exploratory form samples it while it optimizes: every evaluated position with
misfit below the tolerance is kept.

Each particle follows the generalized PSO of Fernández-Martínez & García-Gonzalo
(2008, 2009), the discretization of a damped mass–spring system with a
stochastic centre of attraction. The "cloud" design gives every particle its
own inertia, accelerations and time step drawn inside the second-order
stability region of the trajectories, so nothing is tuned by hand, and the
swarm's dispersion (median distance to its centre of gravity relative to the
first iteration) is monitored so that a collapse is detected.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass
class SwarmSample:
    """The outcome of a swarm run.

    Attributes:
        positions: Every evaluated position, ``(n_iter * n_particles, p)``.
        misfits: Their misfits.
        accepted: Mask of the positions with ``misfit <= tol``.
        best: The lowest-misfit position.
        best_misfit: Its misfit.
        dispersion: Per iteration, the median distance of the particles to
            their centre of gravity relative to iteration 0 (1.0 at the start;
            a value near 0 means the swarm collapsed).
        tolerance: The tolerance used.
    """

    positions: np.ndarray
    misfits: np.ndarray
    accepted: np.ndarray
    best: np.ndarray
    best_misfit: float
    dispersion: np.ndarray
    tolerance: float

    @property
    def samples(self) -> np.ndarray:
        """The accepted positions, the sample of the region of equivalence."""
        return self.positions[self.accepted]

    def interval(self, level: float = 0.9) -> np.ndarray:
        """Central empirical interval per parameter from the accepted samples, ``(2, p)``."""
        s = self.samples
        return np.percentile(s, [50 * (1 - level), 50 * (1 + level)], axis=0)

    def ranges(self) -> np.ndarray:
        """Min and max per parameter over the accepted samples, ``(2, p)``."""
        s = self.samples
        return np.stack([s.min(axis=0), s.max(axis=0)])


def second_order_stable(omega: float, phi: float) -> bool:
    """Whether a PSO parameter pair is inside the second-order (variance) stability region.

    For the classical PSO with uniformly distributed accelerations the mean
    trajectory is stable for ``|omega| < 1`` and ``0 < phi < 2 (omega + 1)``, and
    the variance for ``phi < 12 (1 - omega^2) / (7 - 5 omega)`` (Poli, 2009;
    García-Gonzalo & Fernández-Martínez, 2014, for general distributions).
    ``phi`` is the mean total acceleration ``(a_g + a_l) / 2``.
    """
    return abs(omega) < 1 and 0 < phi < 12 * (1 - omega**2) / (7 - 5 * omega)


def sample_equivalence_region(
    misfit: Callable[[np.ndarray], np.ndarray],
    bounds: Sequence[tuple[float, float]],
    tolerance: float | None = None,
    *,
    sigma: float | None = None,
    level: float = 0.9,
    n_particles: int = 40,
    n_iterations: int = 100,
    seed: int = 0,
    exploration: float = 0.9,
    stop_on_collapse: float = 0.02,
) -> SwarmSample:
    """Sample ``{theta : misfit(theta) <= tolerance}`` with a cloud particle swarm.

    Args:
        misfit: Vectorized misfit, ``(n, p) -> (n,)``. For Gaussian noise use
            the sum of squared residuals over ``sigma^2`` and a tolerance of
            about ``N (1 + epsilon)``; see :func:`misfit_tolerance`.
        bounds: ``(low, high)`` per parameter: the prior range, which the
            swarm explores and particles are reflected into.
        tolerance: Accept positions with misfit at or below it. When ``None``,
            the likelihood-ratio region is used instead: positions within
            ``sigma^2 chi2_p(level)`` of the best misfit found, which for
            Gaussian noise is the asymptotic ``level`` confidence region and
            is directly comparable with the Fisher ellipse at that level.
        sigma: Noise standard deviation, needed when ``tolerance`` is ``None``
            and the misfit is a plain sum of squares (use ``1`` if the misfit
            is already divided by ``sigma^2``).
        level: Confidence level of the likelihood-ratio region.
        n_particles: Swarm size.
        n_iterations: Iterations; every evaluation is a candidate sample.
        seed: Seed of the initial positions and of the per-particle parameters.
        exploration: Fraction of the way to the second-order stability border
            at which each particle's acceleration is drawn (close to 1 is the
            exploratory setting the papers recommend).
        stop_on_collapse: Stop when the dispersion falls below this fraction
            of its initial value.
    """
    rng = np.random.default_rng(seed)
    lo = np.asarray([b[0] for b in bounds], dtype=float)
    hi = np.asarray([b[1] for b in bounds], dtype=float)
    p = lo.size
    x = lo + (hi - lo) * rng.random((n_particles, p))
    v = np.zeros_like(x)
    # Cloud parameters: inertia in [0.5, 0.95], total acceleration a fraction of
    # the way to the variance-stability border, time step in [0.8, 1.2].
    omega = rng.uniform(0.5, 0.95, n_particles)
    phi_max = 12 * (1 - omega**2) / (7 - 5 * omega)
    phi = exploration * phi_max * rng.uniform(0.8, 1.0, n_particles)
    dt = rng.uniform(0.8, 1.2, n_particles)

    f = np.asarray(misfit(x), dtype=float)
    local_best, local_f = x.copy(), f.copy()
    g_idx = int(np.argmin(f))
    global_best, global_f = x[g_idx].copy(), float(f[g_idx])
    positions, misfits = [x.copy()], [f.copy()]
    centre = np.median(x, axis=0)
    d0 = float(np.median(np.linalg.norm(x - centre, axis=1))) or 1.0
    dispersion = [1.0]
    for _ in range(1, n_iterations):
        r1, r2 = rng.random((n_particles, p)), rng.random((n_particles, p))
        a_g = phi[:, None] * r1  # each particle's accelerations sum to phi on average
        a_l = phi[:, None] * r2
        # Generalized PSO with time step dt: damping (1 - omega) dt, stiffness phi dt^2.
        v = (1 - (1 - omega[:, None]) * dt[:, None]) * v + dt[:, None] * (
            a_g * (global_best - x) + a_l * (local_best - x)
        )
        x = x + dt[:, None] * v
        # Reflect into the bounds.
        below, above = x < lo, x > hi
        x = np.where(below, 2 * lo - x, x)
        x = np.where(above, 2 * hi - x, x)
        x = np.clip(x, lo, hi)
        v = np.where(below | above, -v, v)
        f = np.asarray(misfit(x), dtype=float)
        improved = f < local_f
        local_best[improved], local_f[improved] = x[improved], f[improved]
        i = int(np.argmin(local_f))
        if local_f[i] < global_f:
            global_best, global_f = local_best[i].copy(), float(local_f[i])
        positions.append(x.copy())
        misfits.append(f.copy())
        centre = np.median(x, axis=0)
        dispersion.append(float(np.median(np.linalg.norm(x - centre, axis=1))) / d0)
        if dispersion[-1] < stop_on_collapse:
            break
    P = np.concatenate(positions)
    M = np.concatenate(misfits)
    if tolerance is None:
        if sigma is None:
            raise ValueError("give a tolerance, or sigma for the likelihood-ratio region")
        tolerance = global_f + sigma**2 * chi2_quantile(p, level)
    return SwarmSample(
        positions=P,
        misfits=M,
        accepted=M <= tolerance,
        best=global_best,
        best_misfit=global_f,
        dispersion=np.asarray(dispersion),
        tolerance=tolerance,
    )


_CHI2 = {  # quantiles of chi-square with p degrees of freedom
    0.9: {1: 2.706, 2: 4.605, 3: 6.251, 4: 7.779, 5: 9.236},
    0.95: {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070},
    0.99: {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086},
}


def chi2_quantile(p: int, level: float = 0.9) -> float:
    """Quantile of the chi-square distribution with ``p`` degrees of freedom (``p <= 5``)."""
    try:
        return _CHI2[level][p]
    except KeyError as exc:
        raise ValueError(f"chi2 quantile tabulated for levels {list(_CHI2)} and p <= 5") from exc


def misfit_tolerance(n_observations: int, sigma: float, epsilon: float = 0.1) -> float:
    """Tolerance on the squared misfit ``sum (y - u)^2`` for Gaussian noise.

    The expected squared misfit of the true parameters is ``N sigma^2``; the
    region of equivalence is taken as the set within ``(1 + epsilon)`` of it.
    """
    return (1 + epsilon) * n_observations * sigma**2


__all__ = [
    "SwarmSample",
    "chi2_quantile",
    "misfit_tolerance",
    "sample_equivalence_region",
    "second_order_stable",
]
