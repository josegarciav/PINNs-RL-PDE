"""Moving-horizon estimation for process models (the windowed counterpart of the Kalman filter).

At each record the estimator re-solves a small batch problem over the last
``horizon`` records: it looks for the state at the start of the window and
the disturbances inside it that best explain the window's records, weighted
by the measurement noise ``R``, the process noise ``Q`` and an arrival cost
that summarizes everything before the window. For a linear model without
bounds this is the Kalman filter, for any horizon; what the window buys is
what the filter cannot do: the model is re-linearized along the whole window
at every iteration instead of once per record, and bounds on the states and
on the disturbances (a fouling resistance that only grows) are constraints of
the optimization rather than a clip after the update.

The window problem is a bounded nonlinear least squares in the start state
and one disturbance per record for every state with process noise; the
states without process noise follow the model exactly, so a parameter
appended as a random walk costs one unknown per record, not a state vector.
The Jacobian uses the filter's linearization (the matrix exponential of the
finite-difference Jacobian over each sampling interval), chained along the
window, and each Gauss–Newton step is a bounded linear least squares, so the
set of active bounds is found exactly at every iteration. The arrival cost is the filtering update: an extended Kalman filter
run alongside gives the mean and covariance at the window's start.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
from scipy.linalg import expm
from scipy.optimize import lsq_linear

from pinnrl.process.ekf import ExtendedKalmanFilter, FilterResult, finite_difference_jacobian

Dynamics = Callable[[float, np.ndarray], np.ndarray]
Measurement = Callable[[float, np.ndarray], np.ndarray]


class MovingHorizonEstimator:
    """MHE for ``ds/dt = f(t, s)``, ``y_k = h(t_k, s(t_k)) + noise``.

    Args:
        f: Continuous-time dynamics ``f(t, s)`` (augmented with the parameters
            to estimate, as for the filter).
        h: Measurement function ``h(t, s)`` returning ``(m,)``.
        Q: Process-noise intensity per unit time, a diagonal ``(n,)``. A state
            with a zero entry follows the model exactly inside the window; one
            with a positive entry gets a disturbance per record.
        R: Measurement covariance ``(m, m)`` or a diagonal ``(m,)``.
        horizon: Number of records in the window.
        method: ``solve_ivp`` method for the state propagation.
        max_iterations: Cap on the Gauss–Newton iterations of one window solve.
        step_tolerance: A solve stops when its step, in units of the prior and
            disturbance standard deviations, is below this.
    """

    def __init__(
        self,
        f: Dynamics,
        h: Measurement,
        Q: np.ndarray,
        R: np.ndarray,
        horizon: int,
        method: str = "LSODA",
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_iterations: int = 10,
        step_tolerance: float = 1e-3,
    ):
        Q = np.asarray(Q, dtype=float)
        if Q.ndim != 1:
            raise ValueError("MovingHorizonEstimator takes a diagonal Q, shape (n,)")
        if horizon < 1:
            raise ValueError("horizon must be at least one record")
        self.filter = ExtendedKalmanFilter(f, h, Q, R, method=method, rtol=rtol, atol=atol)
        self.f, self.h = f, h
        self.noisy = np.flatnonzero(Q > 0)
        self.q = Q[self.noisy]
        self.horizon = int(horizon)
        self.max_iterations = max_iterations
        self.step_tolerance = step_tolerance
        self._R_whiten = np.linalg.inv(np.linalg.cholesky(self.filter.R))

    # ------------------------------------------------------------- one window
    def _linearize(self, v, t_arrival, times, records, prior_mean, prior_whiten):
        """Residuals and Jacobian of the window problem at ``v = [z, w_1..w_N]``.

        Also returns the states at the records and the sensitivity of the last
        one to ``v`` (for the covariance of the estimate).
        """
        n, p, N = prior_mean.size, self.noisy.size, times.size
        m = records.shape[1]
        z, w = v[:n], v[n:].reshape(N, p)
        x = z.copy()
        Phi = np.eye(n)  # d x_j / d z
        D = np.zeros((N, n, p))  # d x_j / d w_i
        G = np.zeros((n, p))
        G[self.noisy, np.arange(p)] = 1.0
        r = np.zeros(N * m + n + N * p)
        J = np.zeros((r.size, v.size))
        states = np.zeros((N, n))
        t_prev = t_arrival
        for j in range(N):
            dt = times[j] - t_prev
            A = finite_difference_jacobian(lambda s, _t=t_prev: self.f(_t, s), x)
            F = expm(A * dt) if dt > 0 else np.eye(n)
            x = self.filter.propagate(t_prev, times[j], x)
            x[self.noisy] += w[j]
            Phi = F @ Phi
            D[:j] = F @ D[:j]
            D[j] = G
            H = finite_difference_jacobian(lambda s, _t=times[j]: self.h(_t, s), x)
            rows = slice(j * m, (j + 1) * m)
            r[rows] = self._R_whiten @ (np.asarray(self.h(times[j], x), dtype=float) - records[j])
            WH = self._R_whiten @ H
            J[rows, :n] = WH @ Phi
            J[rows, n : n + (j + 1) * p] = np.concatenate([WH @ D[i] for i in range(j + 1)], axis=1)
            states[j] = x
            t_prev = times[j]
        base = N * m
        r[base : base + n] = prior_whiten @ (z - prior_mean)
        J[base : base + n, :n] = prior_whiten
        base += n
        steps = np.diff(np.concatenate([[t_arrival], times]))
        noise_std = np.sqrt(self.q[None, :] * np.maximum(steps, 1e-300)[:, None]).ravel()
        r[base:] = w.ravel() / noise_std
        J[base:, n:] = np.diag(1.0 / noise_std)
        end = np.concatenate([Phi, D.transpose(1, 0, 2).reshape(n, N * p)], axis=1)
        return r, J, states, end, noise_std

    def _solve(self, t_arrival, times, records, mean, cov, guess, bounds):
        std = np.sqrt(np.maximum(np.diag(cov), 1e-300))
        # Whiten the prior through its correlation matrix: the variances span many decades.
        lam, V = np.linalg.eigh(cov / np.outer(std, std))
        prior_whiten = (V / np.sqrt(np.maximum(lam, 1e-12))).T / std[None, :]
        lower, upper = bounds
        bounded = bool(np.isfinite(lower).any() or np.isfinite(upper).any())
        steps = np.diff(np.concatenate([[t_arrival], times]))
        scale = np.concatenate(
            [std, np.sqrt(self.q[None, :] * np.maximum(steps, 1e-300)[:, None]).ravel()]
        )

        def evaluate(v):
            return self._linearize(v, t_arrival, times, records, mean, prior_whiten)

        # Gauss-Newton, each step a bounded linear least squares (the active set is found
        # exactly, which a warm start lying on the bounds needs), halved while the cost rises.
        v = np.clip(guess, lower, upper)
        current = evaluate(v)
        cost = float(current[0] @ current[0])
        evaluations = 1
        for _ in range(self.max_iterations):
            r, J = current[0], current[1]
            Js = J * scale[None, :]
            if bounded:
                step = lsq_linear(
                    Js, -r, bounds=((lower - v) / scale, (upper - v) / scale), method="bvls"
                ).x
            else:
                step = np.linalg.lstsq(Js, -r, rcond=None)[0]
            if float(np.max(np.abs(step))) < self.step_tolerance:
                break  # a step below the tolerance (in prior standard deviations): converged
            alpha, accepted = 1.0, False
            for _halving in range(5):
                trial = np.clip(v + alpha * step * scale, lower, upper)
                candidate = evaluate(trial)
                evaluations += 1
                trial_cost = float(candidate[0] @ candidate[0])
                if trial_cost <= cost:
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted:
                break
            improved = cost - trial_cost
            v, current, cost = trial, candidate, trial_cost
            if improved < 1e-9 * max(cost, 1e-300):
                break
        r, J, states, end, _ = current
        Js = J * scale[None, :]
        cov_v = np.linalg.pinv(Js.T @ Js) * np.outer(scale, scale)
        P_end = end @ cov_v @ end.T
        y_pred = np.asarray(self.h(times[-1], states[-1]), dtype=float).ravel()
        return v, states, 0.5 * (P_end + P_end.T), records[-1] - y_pred, evaluations

    # ------------------------------------------------------------------ run
    def run(
        self,
        times: np.ndarray,
        records: np.ndarray,
        x0: np.ndarray,
        P0: np.ndarray,
        t_start: float = 0.0,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
        noise_lower: np.ndarray | float | None = None,
        noise_upper: np.ndarray | float | None = None,
        stride: int = 1,
    ) -> FilterResult:
        """Estimate the state at every ``stride``-th record from the window ending there.

        ``lower``/``upper`` bound the state at the start of each window (and
        clip the arrival filter's mean); ``noise_lower``/``noise_upper`` bound
        the disturbance of each state with process noise per record, in the
        state's units (``noise_lower=0`` for a parameter that only grows).
        Until ``horizon`` records have arrived the window grows from the prior.

        Returns:
            A :class:`FilterResult` whose ``t`` holds the solved records,
            ``filtered`` the estimates at the end of each window and
            ``filtered_cov`` their Gauss–Newton covariances.
        """
        tic = time.perf_counter()
        times = np.asarray(times, dtype=float)
        records = np.asarray(records, dtype=float).reshape(times.size, -1)
        x0, P0 = np.asarray(x0, dtype=float), np.asarray(P0, dtype=float)
        n, p, K = x0.size, self.noisy.size, times.size
        lo = np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        hi = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float)
        w_lo = np.broadcast_to(-np.inf if noise_lower is None else noise_lower, (p,))
        w_hi = np.broadcast_to(np.inf if noise_upper is None else noise_upper, (p,))
        # arrival[i]: the filter's mean and covariance before record i (after record i - 1)
        arrival = [(t_start, x0.copy(), P0.copy())]
        solved_t, xs, Ps, innov = [], [], [], []
        previous = None  # (first record index, start state, states, disturbances) of the last solve
        self.evaluations = 0
        for k in range(K):
            if (k + 1) % stride != 0 and k != K - 1:
                continue
            a = max(0, k + 1 - self.horizon)
            while len(arrival) <= a:
                i = len(arrival) - 1
                t_prev, x, P = arrival[-1]
                x, P, *_ = self.filter.step(t_prev, times[i], x, P, records[i], lower, upper)
                arrival.append((times[i], x, P))
            t_a, mean, cov = arrival[a]
            N = k - a + 1
            z, w = mean.copy(), np.zeros((N, p))
            if previous is not None:  # warm start from the last solution, shifted
                a_prev, z_prev, states_prev, w_prev = previous
                shift = a - a_prev
                if shift == 0:
                    z = z_prev
                elif shift <= states_prev.shape[0]:
                    z = states_prev[shift - 1]
                keep = w_prev[shift:]
                w[: keep.shape[0]] = keep[:N]
            bounds = (
                np.concatenate([lo, np.tile(w_lo, N)]),
                np.concatenate([hi, np.tile(w_hi, N)]),
            )
            v, states, P_end, innovation, nfev = self._solve(
                t_a,
                times[a : k + 1],
                records[a : k + 1],
                mean,
                cov,
                np.concatenate([z, w.ravel()]),
                bounds,
            )
            self.evaluations += nfev
            previous = (a, v[:n], states, v[n:].reshape(N, p))
            solved_t.append(times[k])
            xs.append(states[-1])
            Ps.append(P_end)
            innov.append(innovation)
        return FilterResult(
            t=np.array(solved_t),
            filtered=np.array(xs),
            filtered_cov=np.array(Ps),
            innovations=np.array(innov),
            time_s=time.perf_counter() - tic,
        )
