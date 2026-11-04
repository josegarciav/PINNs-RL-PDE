"""An extended Kalman filter and smoother for process models (the classical baseline, J05).

The filter takes a continuous-time model ``ds/dt = f(t, s)`` and a measurement
``y = h(t, s)``, linearizes both by central finite differences, propagates the
state between records with ``solve_ivp`` and the covariance with the matrix
exponential of the Jacobian, and keeps what a Rauch–Tung–Striebel smoother
needs for the batch estimate. Parameters to estimate are appended to the state
as random walks (:func:`augment_random_walk`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

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
        n = x0.size
        x, P = np.asarray(x0, dtype=float).copy(), np.asarray(P0, dtype=float).copy()
        t_prev = t_start
        xs, Ps, x_preds, P_preds, Fs, innov = [], [], [], [], [], []
        for t, y in zip(times, records, strict=True):
            dt = t - t_prev
            # linearize at the current estimate, then propagate mean and covariance
            J = finite_difference_jacobian(lambda s, _t=t_prev: self.f(_t, s), x)
            F = expm(J * dt) if dt > 0 else np.eye(n)
            x_pred = self.propagate(t_prev, t, x)
            P_pred = F @ P @ F.T + self.Q * dt
            H = finite_difference_jacobian(lambda s, _t=t: self.h(_t, s), x_pred)
            y_pred = np.asarray(self.h(t, x_pred), dtype=float).ravel()
            S = H @ P_pred @ H.T + self.R
            K_gain = np.linalg.solve(S, H @ P_pred).T
            x = x_pred + K_gain @ (y - y_pred)
            if lower is not None or upper is not None:
                x = np.clip(
                    x,
                    lower if lower is not None else -np.inf,
                    upper if upper is not None else np.inf,
                )
            IKH = np.eye(n) - K_gain @ H
            P = IKH @ P_pred @ IKH.T + K_gain @ self.R @ K_gain.T
            P = 0.5 * (P + P.T)
            xs.append(x.copy())
            Ps.append(P.copy())
            x_preds.append(x_pred)
            P_preds.append(P_pred)
            Fs.append(F)
            innov.append(y - y_pred)
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
