"""Tests for the numerical reference solutions in :mod:`pinnrl.benchmarks.reference`."""

import numpy as np

from pinnrl.benchmarks.reference import (
    ReferenceSolution,
    allen_cahn_reference,
    burgers_cole_hopf,
    burgers_reference,
)


def test_burgers_matches_basdevant_benchmark():
    """Max |u_x| at x=0, t=1.6037/pi is 152.00516 (Basdevant et al., 1986)."""
    nu, t_star, h = 0.01 / np.pi, 1.6037 / np.pi, 1e-6
    up, um = burgers_cole_hopf(np.array([h, -h]), t_star, nu)
    assert abs((up - um) / (2 * h) + 152.00516) < 1e-3


def test_burgers_grid_initial_and_boundary_values():
    """The grid starts at -sin(pi x) and keeps u(±1, t) = 0."""
    ref = burgers_reference(nx=33, nt=5)
    assert ref.u.shape == (33, 5)
    np.testing.assert_allclose(ref.u[:, 0], -np.sin(np.pi * ref.x), atol=1e-12)
    assert np.abs(ref.u[[0, -1], :]).max() < 1e-10


def test_allen_cahn_is_grid_converged():
    """Doubling the solver resolution changes the field by far less than PINN errors."""
    coarse = allen_cahn_reference(nx=65, nt=11, refine=8)
    fine = allen_cahn_reference(nx=65, nt=11, refine=16)
    assert coarse.u.shape == (65, 11)
    assert np.all(coarse.u[[0, -1], :] == -1.0)
    np.testing.assert_allclose(coarse.u[:, 0], coarse.x**2 * np.cos(np.pi * coarse.x))
    assert np.linalg.norm(coarse.u - fine.u) / np.linalg.norm(fine.u) < 1e-4


def test_relative_l2_and_flat_points():
    """Flattened points follow ij order and relative L2 is scale-free."""
    ref = ReferenceSolution(x=np.array([0.0, 1.0]), t=np.array([0.0, 0.5, 1.0]), u=np.ones((2, 3)))
    x, t = ref.flat_points()
    assert x.shape == (6, 1) and t[:3, 0].tolist() == [0.0, 0.5, 1.0]
    assert ref.relative_l2(np.ones(6)) == 0.0
    assert np.isclose(ref.relative_l2(np.full(6, 1.1)), 0.1)
