"""The extended Kalman filter and smoother of ``pinnrl.process.ekf``."""

import numpy as np

from pinnrl.process import ExtendedKalmanFilter, FallingFilmEvaporator, augment_random_walk


def test_linear_scalar_filter_matches_the_closed_form_kalman_recursion():
    """A constant observed with noise: the EKF must reproduce the scalar Kalman filter."""
    rng = np.random.default_rng(0)
    truth = 2.0
    times = np.arange(1.0, 21.0)
    y = truth + 0.5 * rng.standard_normal(times.size)
    ekf = ExtendedKalmanFilter(
        lambda t, x: np.zeros(1), lambda t, x: x, Q=np.array([0.0]), R=np.array([0.25])
    )
    res = ekf.run(times, y[:, None], np.array([0.0]), np.array([[4.0]]), smooth=True)
    x, P = 0.0, 4.0
    for k, yk in enumerate(y):
        K = P / (P + 0.25)
        x = x + K * (yk - x)
        P = (1 - K) * P
        np.testing.assert_allclose(res.filtered[k, 0], x, rtol=1e-8)
        np.testing.assert_allclose(res.filtered_cov[k, 0, 0], P, rtol=1e-8)
    # With no process noise the smoother is the final estimate everywhere.
    np.testing.assert_allclose(res.smoothed[:, 0], res.filtered[-1, 0], rtol=1e-6)


def test_fouling_filter_tracks_a_constant_resistance_from_clean_outlet_records():
    """The evaporator filter on its own grid recovers a constant R_f from noise-free records."""
    nz = 11
    plant = FallingFilmEvaporator(R_f=1.5e-4)
    ref = plant.reference(t_max=600.0, nz=nz, nt=121)
    times = np.arange(5.0, 600.0 + 1e-9, 5.0)
    out = ref.outlet()
    records = np.stack(
        [np.interp(times, out["t"], out["X"]), np.interp(times, out["t"], out["W"])], axis=1
    )
    model = FallingFilmEvaporator(R_f=0.0)
    f, h, s0 = model.fouling_state_space(nz)
    n = s0.size
    ekf = ExtendedKalmanFilter(
        augment_random_walk(f, n, 1),
        lambda t, x: h(t, x[:n]),
        Q=np.concatenate([np.full(n, 1e-14), [1e-14]]),
        R=np.array([1e-8, 1e-8]),
    )
    x0 = np.concatenate([s0, [0.5e-4]])
    P0 = np.diag(np.concatenate([np.full(n, 1e-8), [1e-8]]))
    res = ekf.run(times, records, x0, P0, smooth=True, lower=np.r_[np.full(n, -np.inf), 0.0])
    rf = res.filtered[:, -1]
    assert abs(rf[-1] - 1.5e-4) / 1.5e-4 < 0.02
    assert abs(res.smoothed[-1, -1] - 1.5e-4) / 1.5e-4 < 0.02
    assert res.std()[-1, -1] < res.std()[0, -1]


def test_local_linear_trend_follows_a_ramp_that_a_random_walk_lags():
    """A parameter drifting at a constant rate, observed directly with noise."""
    from pinnrl.process import augment_local_linear_trend

    rng = np.random.default_rng(1)
    times = np.arange(1.0, 201.0)
    slope = 0.05
    y = slope * times + 0.5 * rng.standard_normal(times.size)

    def f(t, s, theta):  # no physical state: the parameter is the measurement
        return np.zeros(0)

    R = np.array([0.25])
    trend = ExtendedKalmanFilter(
        augment_local_linear_trend(f, 0, 1), lambda t, x: x[:1], Q=np.array([0.0, 1e-8]), R=R
    )
    res = trend.run(times, y[:, None], np.array([0.0, 0.0]), np.diag([1.0, 1.0]))
    walk = ExtendedKalmanFilter(
        augment_random_walk(f, 0, 1), lambda t, x: x, Q=np.array([1e-4]), R=R
    )
    lagging = walk.run(times, y[:, None], np.array([0.0]), np.array([[1.0]]))
    truth = slope * times[-1]
    assert abs(res.filtered[-1, 0] - truth) < 3 * res.std()[-1, 0]
    assert abs(res.filtered[-1, 1] - slope) < 3 * res.std()[-1, 1]
    assert res.std()[-1, 1] < 0.01 * slope * 10  # the rate is determined, not left at its prior
    # The slow random walk sits behind the ramp by many of its own standard deviations.
    assert truth - lagging.filtered[-1, 0] > 5 * lagging.std()[-1, 0]
    lo, hi = res.interval(0.9)
    assert lo.shape == (times.size, 2) and np.all(lo < hi)
    assert lo[-1, 0] < truth < hi[-1, 0]


def test_time_to_threshold_matches_the_exact_ratio_and_flags_an_unreached_threshold():
    from pinnrl.process import time_to_threshold

    # No uncertainty: the deterministic crossing time.
    exact = time_to_threshold(2.0, 0.5, np.zeros((2, 2)), threshold=3.0)
    assert exact.median == exact.lower == exact.upper == 2.0
    assert exact.probability == 1.0
    # A well-determined rate: the interval is the delta-method one to a few percent.
    cov = np.diag([0.05**2, 0.01**2])
    tt = time_to_threshold(2.0, 0.5, cov, threshold=3.0, n_samples=40000)
    sd = np.sqrt((0.05 / 0.5) ** 2 + (1.0 * 0.01 / 0.5**2) ** 2)
    np.testing.assert_allclose(tt.median, 2.0, rtol=0.01)
    np.testing.assert_allclose(tt.upper - tt.lower, 2 * 1.645 * sd, rtol=0.05)
    # A rate that may be zero or negative: part of the draws never arrive.
    vague = time_to_threshold(2.0, 0.01, np.diag([0.05**2, 0.02**2]), threshold=3.0)
    assert vague.probability < 0.8
    assert np.isinf(vague.upper)
    # Already past the threshold: zero. A falling quantity approaches from above.
    assert time_to_threshold(3.5, 0.5, cov, threshold=3.0, rising=True).median == 0.0
    falling = time_to_threshold(5.0, -0.5, cov, threshold=3.0)
    np.testing.assert_allclose(falling.median, 4.0, rtol=0.02)
