"""The compiled evaporator integrator (pinnrl.native.evaporator, I16) against the SciPy paths."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pinnrl.native.evaporator import evaporator_available, evaporator_solve
from pinnrl.process import (
    ExtendedKalmanFilter,
    FallingFilmEvaporator,
    MovingHorizonEstimator,
    monitor_fouling,
)

pytestmark = pytest.mark.skipif(not evaporator_available(), reason="no compiled integrator here")

SMOOTH = dict(
    T_s=lambda t: 355.15 + 2.0 * np.sin(t / 600.0),
    W_in=lambda t: 0.1 + 0.01 * np.sin(t / 900.0),
    R_f=lambda t: 1e-4 * t / 3600.0,
)


def _record_model(seed=0):
    """A model driven by a noisy steam record, interpolated linearly as plant data would be."""
    tt = np.arange(0.0, 1801.0, 10.0)
    Ts = (
        355.15
        + 2.0 * np.sin(tt / 600.0)
        + 0.05 * np.random.default_rng(seed).standard_normal(tt.size)
    )
    return FallingFilmEvaporator(
        T_s=lambda t: float(np.interp(t, tt, Ts)), R_f=lambda t: 1e-4 * t / 3600.0
    )


@pytest.mark.parametrize("nz", [11, 51])
def test_reference_matches_a_tight_bdf_solve(nz):
    m = FallingFilmEvaporator(**SMOOTH)
    tight = m.reference(1800.0, nz=nz, nt=31, rtol=1e-11, atol=1e-14)
    fast = m.reference(1800.0, nz=nz, nt=31, method="native")
    assert fast.X.u.shape == tight.X.u.shape and np.array_equal(fast.z, tight.z)
    for a, b in ((fast.X.u, tight.X.u), (fast.W.u, tight.W.u), (fast.q, tight.q)):
        assert np.abs(a - b).max() <= 3e-7 * np.abs(b).max()
    loose = m.reference(1800.0, nz=nz, nt=31, method="native", rtol=1e-6, atol=1e-9)
    assert np.abs(loose.X.u - tight.X.u).max() <= 1e-5 * np.abs(tight.X.u).max()
    # the inlet row carries the signals and the first row the initial state
    assert np.allclose(fast.W.u[0], [SMOOTH["W_in"](t) for t in fast.t])
    assert np.allclose(fast.X.u[:, 0], tight.X.u[:, 0])


def test_reference_from_a_given_profile_and_a_noisy_record():
    m = _record_model()
    ss = m.steady_state(21, t=0.0)
    start = {"W": ss["W"] * 1.05, "X": ss["X"] * 0.95}
    bdf = m.reference(600.0, nz=21, nt=13, initial=start)
    nat = m.reference(600.0, nz=21, nt=13, initial=start, method="native")
    assert np.allclose(nat.X.u[1:, 0], start["X"][1:]) and np.allclose(
        nat.W.u[1:, 0], start["W"][1:]
    )
    # the native path splines the sampled record where SciPy interpolates it linearly:
    # a modelling difference far below the record's noise, not an integration error
    assert np.abs(nat.X.u - bdf.X.u).max() <= 2e-4 * np.abs(bdf.X.u).max()
    with pytest.raises(ValueError, match="s0 must have"):
        evaporator_solve(
            5,
            m._geometry(),
            np.array([0.0, 1.0]),
            np.zeros((2, 4)),
            np.zeros(4),
            np.array([0.0, 1.0]),
        )


def test_propagator_matches_solve_ivp_for_both_state_layouts():
    m = FallingFilmEvaporator(**SMOOTH)  # smooth inputs: the two paths solve the same model
    nz = 21
    f, h, s0 = m.fouling_state_space(nz, steam_bias=True)
    n = s0.size
    # the monitor's layout: [s, level, rate, bias], the fouling a ramp inside the step
    x = np.concatenate([s0, [5e-5, 2e-8, 0.3]])

    def trend_dynamics(t, xx):
        out = np.zeros_like(xx)
        out[:n] = f(t, xx[:n], xx[[n, n + 2]])
        out[n] = xx[n + 1]
        return out

    ref = solve_ivp(trend_dynamics, (100.0, 160.0), x, method="LSODA", rtol=1e-10, atol=1e-13).y[
        :, -1
    ]
    got = m.native_propagator(nz, steam_bias=True, trend=True)(100.0, 160.0, x)
    assert np.abs(got - ref).max() <= 1e-6 * np.abs(ref).max()
    assert got[n] == pytest.approx(5e-5 + 2e-8 * 60.0) and got[n + 1] == 2e-8 and got[n + 2] == 0.3
    # the plain layout: [s, R_f] held constant between records
    f1, _, s1 = m.fouling_state_space(nz)
    x1 = np.concatenate([s1, [8e-5]])
    ref1 = solve_ivp(
        lambda t, xx: np.append(f1(t, xx[:n], xx[n:]), 0.0),
        (0.0, 30.0),
        x1,
        method="LSODA",
        rtol=1e-10,
        atol=1e-13,
    ).y[:, -1]
    prop = m.native_propagator(nz)
    assert np.abs(prop(0.0, 30.0, x1) - ref1).max() <= 1e-6 * np.abs(ref1).max()
    assert np.array_equal(prop(5.0, 5.0, x1), x1)


def test_filters_and_monitor_agree_between_the_integrators():
    m = _record_model(2)
    nz = 11
    truth = m.reference(600.0, nz=nz, nt=61)
    times = truth.t[1:]
    rng = np.random.default_rng(3)
    records = np.stack(
        [
            truth.X.u[-1, 1:] + 0.002 * rng.standard_normal(times.size),
            truth.W.u[-1, 1:] + 0.001 * rng.standard_normal(times.size),
        ],
        1,
    )
    runs = {k: monitor_fouling(m, times, records, nz=nz, integrator=k) for k in ("scipy", "native")}
    assert np.abs(runs["native"].fouling - runs["scipy"].fouling).max() <= 1e-3 * (
        np.abs(runs["scipy"].fouling).max() + 1e-4
    )
    assert np.allclose(runs["native"].misfit, runs["scipy"].misfit, rtol=1e-2)
    auto = monitor_fouling(m, times, records, nz=nz)
    assert np.allclose(auto.fouling, runs["native"].fouling)
    with pytest.raises(ValueError, match="integrator must be"):
        monitor_fouling(m, times, records, nz=nz, integrator="fast")
    # the EKF and the MHE take the propagator directly
    f, h, s0 = m.fouling_state_space(nz)
    n = s0.size

    def dyn(t, x):
        return np.append(f(t, x[:n], x[n:]), 0.0)

    Q = np.append(np.full(n, 1e-12), 1e-12)
    kwargs = dict(Q=Q, R=np.array([0.002**2, 0.001**2]))
    x0 = np.append(s0, 0.0)
    P0 = np.diag(np.append(np.full(n, 1e-6), 1e-8))
    prop = m.native_propagator(nz)
    a = ExtendedKalmanFilter(dyn, lambda t, x: h(t, x[:n]), **kwargs).run(times, records, x0, P0)
    b = ExtendedKalmanFilter(dyn, lambda t, x: h(t, x[:n]), propagate=prop, **kwargs).run(
        times, records, x0, P0
    )
    assert np.abs(a.filtered - b.filtered).max() <= 1e-5 * np.abs(a.filtered).max()
    mhe = MovingHorizonEstimator(dyn, lambda t, x: h(t, x[:n]), horizon=5, propagate=prop, **kwargs)
    assert mhe.filter._propagate is prop
