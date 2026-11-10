"""Moving-horizon estimation (``pinnrl.process.mhe``)."""

import numpy as np
import pytest

from pinnrl.process import (
    ExtendedKalmanFilter,
    FallingFilmEvaporator,
    MovingHorizonEstimator,
    augment_random_walk,
)


def _linear_problem(seed=0, K=40):
    """A damped oscillator observed in position, with process noise on the velocity."""
    A = np.array([[0.0, 1.0], [-1.0, -0.3]])
    rng = np.random.default_rng(seed)
    times = 0.2 * np.arange(1, K + 1)
    from scipy.linalg import expm

    F = expm(A * 0.2)
    x, ys = np.array([1.0, 0.0]), []
    for _ in times:
        x = F @ x + np.array([0.0, 0.05 * np.sqrt(0.2) * rng.standard_normal()])
        ys.append(x[0] + 0.1 * rng.standard_normal())
    return A, times, np.array(ys)[:, None]


@pytest.mark.parametrize("horizon", [1, 5, 40])
def test_linear_unconstrained_mhe_is_the_kalman_filter_for_any_horizon(horizon):
    A, times, y = _linear_problem()
    f = lambda t, x: A @ x  # noqa: E731
    h = lambda t, x: x[:1]  # noqa: E731
    Q, R = np.array([0.0, 0.05**2]), np.array([0.1**2])
    x0, P0 = np.array([0.5, 0.5]), np.diag([1.0, 1.0])
    kf = ExtendedKalmanFilter(f, h, Q, R).run(times, y, x0, P0)
    mhe = MovingHorizonEstimator(f, h, Q, R, horizon=horizon).run(times, y, x0, P0)
    np.testing.assert_allclose(mhe.t, times)
    np.testing.assert_allclose(mhe.filtered, kf.filtered, atol=2e-5)
    np.testing.assert_allclose(mhe.filtered_cov, kf.filtered_cov, atol=2e-6, rtol=2e-3)


def test_bounds_on_the_disturbance_keep_a_parameter_monotone():
    """A parameter that only grows, observed with noise: the bounded estimate never decreases."""
    rng = np.random.default_rng(3)
    times = np.arange(1.0, 61.0)
    truth = 0.02 * times
    y = (truth + 0.3 * rng.standard_normal(times.size))[:, None]
    f = lambda t, x: np.zeros(1)  # noqa: E731
    h = lambda t, x: x  # noqa: E731
    Q, R = np.array([0.03**2]), np.array([0.3**2])
    x0, P0 = np.array([0.0]), np.array([[1.0]])
    free = MovingHorizonEstimator(f, h, Q, R, horizon=10).run(times, y, x0, P0)
    grow = MovingHorizonEstimator(f, h, Q, R, horizon=10).run(
        times, y, x0, P0, lower=np.array([0.0]), noise_lower=0.0
    )
    assert np.any(np.diff(free.filtered[:, 0]) < -1e-3)  # the free estimate follows the noise down
    # Inside every window the bounded trajectory is non-decreasing, so its end is at least its start.
    assert np.all(grow.filtered[:, 0] >= 0.0)
    # Warm starts sit on the bounds; the solve must still follow the ramp to its end.
    assert abs(grow.filtered[-1, 0] - truth[-1]) < 0.3
    assert np.mean(np.abs(grow.filtered[20:, 0] - truth[20:])) < np.mean(
        np.abs(free.filtered[20:, 0] - truth[20:])
    )


def test_stride_solves_a_subset_of_the_records_with_the_same_estimates():
    A, times, y = _linear_problem(seed=1)
    f = lambda t, x: A @ x  # noqa: E731
    h = lambda t, x: x[:1]  # noqa: E731
    Q, R = np.array([0.0, 0.05**2]), np.array([0.1**2])
    x0, P0 = np.array([0.5, 0.5]), np.diag([1.0, 1.0])
    every = MovingHorizonEstimator(f, h, Q, R, horizon=8).run(times, y, x0, P0)
    fourth = MovingHorizonEstimator(f, h, Q, R, horizon=8).run(times, y, x0, P0, stride=4)
    np.testing.assert_allclose(fourth.t, times[3::4])
    np.testing.assert_allclose(fourth.filtered, every.filtered[3::4], atol=1e-5)


def test_mhe_tracks_the_fouling_resistance_like_the_filter():
    """The evaporator with R_f a random walk: the windowed estimate agrees with the EKF."""
    nz = 6
    plant = FallingFilmEvaporator(R_f=lambda t: 1e-4 + 1e-4 * t / 300.0)
    ref = plant.reference(t_max=300.0, nz=nz, nt=61)
    times = np.arange(5.0, 300.0 + 1e-9, 5.0)
    out = ref.outlet()
    rng = np.random.default_rng(0)
    records = np.stack(
        [
            np.interp(times, out["t"], out["X"]) + 0.002 * rng.standard_normal(times.size),
            np.interp(times, out["t"], out["W"]) + 0.001 * rng.standard_normal(times.size),
        ],
        axis=1,
    )
    f, h, s0 = FallingFilmEvaporator(R_f=1e-4).fouling_state_space(nz)
    n = s0.size
    f_aug = augment_random_walk(f, n, 1)
    h_aug = lambda t, x: h(t, x[:n])  # noqa: E731
    Q = np.concatenate([np.zeros(n), [(1e-6) ** 2]])
    R = np.array([0.002**2, 0.001**2])
    x0 = np.concatenate([s0, [1e-4]])
    P0 = np.diag(np.concatenate([np.full(n, 1e-6), [(1e-4) ** 2]]))
    lower = np.r_[np.full(n, -np.inf), 0.0]
    kf = ExtendedKalmanFilter(f_aug, h_aug, Q, R).run(times, records, x0, P0, lower=lower)
    mhe = MovingHorizonEstimator(f_aug, h_aug, Q, R, horizon=6).run(
        times, records, x0, P0, lower=lower, stride=6
    )
    rf_kf, rf_mhe = kf.filtered[5::6, -1], mhe.filtered[:, -1]
    assert np.max(np.abs(rf_mhe - rf_kf)) < 0.15e-4
    assert abs(rf_mhe[-1] - 2e-4) < 0.4e-4
    np.testing.assert_allclose(mhe.std()[-1, -1], kf.std()[-1, -1], rtol=0.3)
