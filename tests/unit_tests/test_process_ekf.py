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
