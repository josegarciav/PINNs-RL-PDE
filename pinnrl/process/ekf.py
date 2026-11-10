"""An extended Kalman filter and smoother for process models (the classical baseline, J05).

The filter takes a continuous-time model ``ds/dt = f(t, s)`` and a measurement
``y = h(t, s)``, linearizes both by central finite differences, propagates the
state between records with ``solve_ivp`` and the covariance with the matrix
exponential of the Jacobian, and keeps what a Rauch–Tung–Striebel smoother
needs for the batch estimate. Parameters to estimate are appended to the state
as random walks (:func:`augment_random_walk`) or, for a parameter that drifts,
as a level and a rate (:func:`augment_local_linear_trend`), which is the model
that also predicts when the parameter reaches a threshold
(:func:`time_to_threshold`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.stats import norm

Dynamics = Callable[[float, np.ndarray], np.ndarray]
Measurement = Callable[[float, np.ndarray], np.ndarray]


def finite_difference_jacobian(
    fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, rel: float = 1e-6, floor: float = 1e-6
) -> np.ndarray:
    """Central finite differences of ``fn`` at ``x``, one column per state."""
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(fn(x), dtype=float).ravel()
    J = np.zeros((y0.size, x.size))
    for j in range(x.size):
        h = rel * max(abs(x[j]), floor)
        xp, xm = x.copy(), x.copy()
        xp[j] += h
        xm[j] -= h
        J[:, j] = (
            np.asarray(fn(xp), dtype=float).ravel() - np.asarray(fn(xm), dtype=float).ravel()
        ) / (2 * h)
    return J


def augment_random_walk(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    n_state: int,
    n_params: int,
) -> Dynamics:
    """``f(t, s, theta)`` with ``theta`` appended to the state as constants (random walks through ``Q``)."""

    def f_aug(t: float, x: np.ndarray) -> np.ndarray:
        s, theta = x[:n_state], x[n_state:]
        return np.concatenate([np.asarray(f(t, s, theta), dtype=float).ravel(), np.zeros(n_params)])

    return f_aug


def augment_local_linear_trend(
    f: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
    n_state: int,
    n_params: int,
) -> Dynamics:
    """``f(t, s, theta)`` with ``theta`` and its rate appended: ``x = [s, theta, rate]``.

    ``d theta/dt = rate`` and ``d rate/dt = 0``, the rate a random walk through
    ``Q``. A random-walk parameter lags a steady drift or, with enough process
    noise not to lag, reports a band wider than its error; the trend model
    follows a ramp without lag and carries the rate a prognosis needs (J06).
    """

    def f_aug(t: float, x: np.ndarray) -> np.ndarray:
        s = x[:n_state]
        theta = x[n_state : n_state + n_params]
        rate = x[n_state + n_params :]
        return np.concatenate(
            [np.asarray(f(t, s, theta), dtype=float).ravel(), rate, np.zeros(n_params)]
        )

    return f_aug


@dataclass
class ThresholdTime:
    """When a drifting quantity reaches a threshold, from its estimated level and rate.

    Attributes:
        median: Median time to the threshold (``inf`` when most draws never reach it).
        lower: Lower end of the central interval.
        upper: Upper end (``inf`` when more than ``(1 - level) / 2`` of the draws never reach it).
        probability: Fraction of the draws that reach the threshold (a rate of the right sign).
        level: The interval's nominal level.
    """

    median: float
    lower: float
    upper: float
    probability: float
    level: float


def time_to_threshold(
    value: float,
    rate: float,
    covariance: np.ndarray,
    threshold: float,
    level: float = 0.9,
    rising: bool | None = None,
    n_samples: int = 4000,
    seed: int = 0,
) -> ThresholdTime:
    """Time until ``value + rate * tau`` reaches ``threshold``, with its uncertainty.

    ``(value, rate)`` is Gaussian with the given ``2 x 2`` covariance (a filter's
    level and rate with their block of the covariance, or a fitted line's);
    the time ``tau = (threshold - value) / rate`` is a ratio of Gaussians, so
    its interval comes from draws rather than a delta-method standard
    deviation: draws whose rate points away from the threshold never reach it
    (``inf``), and draws already past it give zero. The rate is taken as
    constant from now on, which is the trend model's own prediction.
    ``rising`` says from which side the threshold is approached (a fouling
    resistance rises to its limit, a heat-transfer coefficient falls to its
    own); by default it is the side ``value`` is on.
    """
    cov = np.asarray(covariance, dtype=float).reshape(2, 2)
    std = np.sqrt(np.maximum(np.diag(cov), 0.0))
    safe = np.where(std > 0, std, 1.0)
    corr = cov / np.outer(safe, safe)  # sample in units of the standard deviations
    rho = float(np.clip(corr[0, 1], -1.0, 1.0))
    z = np.random.default_rng(seed).standard_normal((n_samples, 2))
    v = value + std[0] * z[:, 0]
    r = rate + std[1] * (rho * z[:, 0] + np.sqrt(1.0 - rho**2) * z[:, 1])
    if rising is None:
        rising = threshold >= value
    direction = 1.0 if rising else -1.0
    gap = threshold - v
    reaches = direction * r > 0
    tau = np.where(reaches, gap / np.where(reaches, r, 1.0), np.inf)
    tau = np.where(direction * gap <= 0, 0.0, tau)  # draws already past the threshold
    # Order statistics, not interpolation: the draws that never arrive are infinite.
    lo, med, hi = np.quantile(tau, [(1 - level) / 2, 0.5, (1 + level) / 2], method="inverted_cdf")
    return ThresholdTime(
        median=float(med),
        lower=float(lo),
        upper=float(hi),
        probability=float(np.isfinite(tau).mean()),
        level=level,
    )


@dataclass
class FilterResult:
    """States and covariances of a filter run (and of the smoother when ``smoothed`` is set).

    Attributes:
        t: Record times ``(K,)``.
        filtered: Posterior means after each update ``(K, n)``.
        filtered_cov: Posterior covariances ``(K, n, n)``.
        smoothed: RTS-smoothed means ``(K, n)`` or ``None``.
        smoothed_cov: Smoothed covariances or ``None``.
        innovations: ``y - h(x_pred)`` per record ``(K, m)``.
        time_s: Wall-clock time of the run.
    """

    t: np.ndarray
    filtered: np.ndarray
    filtered_cov: np.ndarray
    smoothed: np.ndarray | None = None
    smoothed_cov: np.ndarray | None = None
    innovations: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    time_s: float = 0.0

    def std(self, smoothed: bool = False) -> np.ndarray:
        """Marginal standard deviations ``(K, n)``."""
        cov = self.smoothed_cov if smoothed else self.filtered_cov
        if cov is None:
            raise ValueError("no smoothed covariances; run with smooth=True")
        return np.sqrt(np.maximum(np.einsum("kii->ki", cov), 0.0))

    def interval(self, level: float = 0.9, smoothed: bool = False) -> np.ndarray:
        """Central Gaussian intervals of every state at every record, ``(2, K, n)``."""
        mean = self.smoothed if smoothed else self.filtered
        if mean is None:
            raise ValueError("no smoothed states; run with smooth=True")
        half = float(norm.ppf(0.5 + level / 2)) * self.std(smoothed=smoothed)
        return np.stack([mean - half, mean + half])


class ExtendedKalmanFilter:
    """EKF for ``ds/dt = f(t, s)``, ``y_k = h(t_k, s(t_k)) + noise``.

    Args:
        f: Continuous-time dynamics ``f(t, s)``.
        h: Measurement function ``h(t, s)`` returning ``(m,)``.
        Q: Process-noise intensity per unit time, ``(n, n)`` or a diagonal ``(n,)``;
            the discrete covariance over a step ``dt`` is ``Q dt``.
        R: Measurement covariance ``(m, m)`` or a diagonal ``(m,)``.
        method: ``solve_ivp`` method for the state propagation.
    """

    def __init__(
        self,
        f: Dynamics,
        h: Measurement,
        Q: np.ndarray,
        R: np.ndarray,
        method: str = "LSODA",
        rtol: float = 1e-8,
        atol: float = 1e-10,
    ):
        self.f = f
        self.h = h
        self.Q = np.diag(Q) if np.ndim(Q) == 1 else np.asarray(Q, dtype=float)
        self.R = np.diag(R) if np.ndim(R) == 1 else np.asarray(R, dtype=float)
        self.method = method
        self.rtol = rtol
        self.atol = atol

    def propagate(self, t0: float, t1: float, x: np.ndarray) -> np.ndarray:
        """The state at ``t1`` from ``x`` at ``t0``."""
        if t1 <= t0:
            return x.copy()
        sol = solve_ivp(self.f, (t0, t1), x, method=self.method, rtol=self.rtol, atol=self.atol)
        if not sol.success:
            raise RuntimeError(f"state propagation failed: {sol.message}")
        return sol.y[:, -1]

    def step(
        self,
        t_prev: float,
        t: float,
        x: np.ndarray,
        P: np.ndarray,
        y: np.ndarray,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """One prediction to ``t`` and update with the record ``y``.

        Returns ``(x, P, x_pred, P_pred, F, innovation)``: the posterior, the
        prediction it came from, the transition matrix of the step and
        ``y - h(x_pred)``.
        """
        n = x.size
        dt = t - t_prev
        # linearize at the current estimate, then propagate mean and covariance
        J = finite_difference_jacobian(lambda s, _t=t_prev: self.f(_t, s), x)  # type: ignore[misc]
        F = expm(J * dt) if dt > 0 else np.eye(n)
        x_pred = self.propagate(t_prev, t, x)
        P_pred = F @ P @ F.T + self.Q * dt
        H = finite_difference_jacobian(lambda s, _t=t: self.h(_t, s), x_pred)  # type: ignore[misc]
        y_pred = np.asarray(self.h(t, x_pred), dtype=float).ravel()
        S = H @ P_pred @ H.T + self.R
        K_gain = np.linalg.solve(S, H @ P_pred).T
        x_new = x_pred + K_gain @ (y - y_pred)
        if lower is not None or upper is not None:
            x_new = np.clip(
                x_new,
                lower if lower is not None else -np.inf,
                upper if upper is not None else np.inf,
            )
        IKH = np.eye(n) - K_gain @ H
        P_new = IKH @ P_pred @ IKH.T + K_gain @ self.R @ K_gain.T
        return x_new, 0.5 * (P_new + P_new.T), x_pred, P_pred, F, y - y_pred

    def run(
        self,
        times: np.ndarray,
        records: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        t_start: float = 0.0,
        smooth: bool = False,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> FilterResult:
        """Filter the records ``(K, m)`` at ``times`` from the prior ``(x0, P0)`` at ``t_start``.

        ``lower``/``upper`` clip the posterior mean (bounds on a physical state,
        a fouling resistance that cannot be negative).
        """
        import time as _time

        tic = _time.perf_counter()
        times = np.asarray(times, dtype=float)
        records = np.asarray(records, dtype=float).reshape(times.size, -1)
        x, P = np.asarray(x0, dtype=float).copy(), np.asarray(P0, dtype=float).copy()
        t_prev = t_start
        xs, Ps, x_preds, P_preds, Fs, innov = [], [], [], [], [], []
        for t, y in zip(times, records, strict=True):
            x, P, x_pred, P_pred, F, innovation = self.step(t_prev, t, x, P, y, lower, upper)
            xs.append(x.copy())
            Ps.append(P.copy())
            x_preds.append(x_pred)
            P_preds.append(P_pred)
            Fs.append(F)
            innov.append(innovation)
            t_prev = t
        result = FilterResult(
            t=times,
            filtered=np.array(xs),
            filtered_cov=np.array(Ps),
            innovations=np.array(innov),
        )
        if smooth:
            xs_s, Ps_s = list(xs), list(Ps)
            for k in range(len(times) - 2, -1, -1):
                C = np.linalg.solve(P_preds[k + 1].T, (Ps[k] @ Fs[k + 1].T).T).T
                xs_s[k] = xs[k] + C @ (xs_s[k + 1] - x_preds[k + 1])
                Ps_s[k] = Ps[k] + C @ (Ps_s[k + 1] - P_preds[k + 1]) @ C.T
            result.smoothed = np.array(xs_s)
            result.smoothed_cov = np.array(Ps_s)
        result.time_s = _time.perf_counter() - tic
        return result
