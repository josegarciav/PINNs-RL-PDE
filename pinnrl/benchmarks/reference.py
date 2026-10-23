"""High-accuracy reference solutions for PDEs without a closed form.

The sampling study scores PINNs against these fields instead of the
``exact_solution`` methods on the PDE classes, which only cover special
cases (the Burgers ``cole_hopf`` branch is singular, the Allen-Cahn ``tanh``
branch is stationary).

* :func:`burgers_reference` — Cole-Hopf integral for the standard
  ``u0 = -sin(pi x)`` benchmark (Basdevant et al., 1986), evaluated with a
  log-sum-exp trapezoid rule so the ``exp(±1/(2 pi nu))`` factors never
  overflow.
* :func:`allen_cahn_reference` — second-order finite differences in space
  with an adaptive Runge-Kutta integrator in time (method of lines).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class ReferenceSolution:
    """A solution sampled on a tensor-product ``(x, t)`` grid.

    Attributes:
        x: Spatial grid, shape ``(nx,)``.
        t: Temporal grid, shape ``(nt,)``.
        u: Solution values, shape ``(nx, nt)`` with ``u[i, j] = u(x[i], t[j])``.
    """

    x: np.ndarray
    t: np.ndarray
    u: np.ndarray

    def flat_points(self) -> tuple[np.ndarray, np.ndarray]:
        """Grid coordinates flattened to ``(nx * nt, 1)`` columns, ``ij`` order."""
        xx, tt = np.meshgrid(self.x, self.t, indexing="ij")
        return xx.reshape(-1, 1), tt.reshape(-1, 1)

    def relative_l2(self, u_pred: np.ndarray) -> float:
        """``||u_pred - u||_2 / ||u||_2`` over the grid."""
        u_ref = self.u.reshape(-1)
        diff = np.asarray(u_pred).reshape(-1) - u_ref
        return float(np.linalg.norm(diff) / np.linalg.norm(u_ref))


def burgers_reference(
    nu: float = 0.01 / np.pi,
    nx: int = 256,
    nt: int = 101,
    t_max: float = 1.0,
    n_quad: int = 6001,
    z_max: float = 15.0,
) -> ReferenceSolution:
    """Burgers ``u_t + u u_x = nu u_xx`` on ``[-1, 1]`` with ``u0 = -sin(pi x)``.

    Cole-Hopf gives

        u(x, t) = -∫ sin(pi (x - η)) f(x - η) G(η, t) dη / ∫ f(x - η) G(η, t) dη

    with ``f(y) = exp(-cos(pi y) / (2 pi nu))`` and the heat kernel ``G``.
    Substituting ``η = sqrt(4 nu t) z`` turns ``G`` into ``exp(-z²)``; the
    integrand is periodic times Gaussian, so the trapezoid rule converges
    spectrally. ``z_max = 15`` keeps the far peaks of ``f`` (which dominate
    near ``x = 0`` once the shock forms) inside the window.

    Args:
        nu: Viscosity. The default is the classic ``0.01 / pi`` benchmark.
        nx: Number of spatial grid points on ``[-1, 1]``.
        nt: Number of time grid points on ``[0, t_max]``.
        t_max: Final time.
        n_quad: Trapezoid nodes on ``[-z_max, z_max]``.
        z_max: Half-width of the quadrature window in ``z``.

    Returns:
        The reference field on the ``(nx, nt)`` grid.
    """
    x = np.linspace(-1.0, 1.0, nx)
    t = np.linspace(0.0, t_max, nt)
    u = np.empty((nx, nt))
    z = np.linspace(-z_max, z_max, n_quad)
    for j, tj in enumerate(t):
        u[:, j] = _burgers_cole_hopf(x, tj, nu, z)
    return ReferenceSolution(x=x, t=t, u=u)


def _burgers_cole_hopf(x: np.ndarray, t: float, nu: float, z: np.ndarray) -> np.ndarray:
    """Cole-Hopf quotient at one time level for every ``x``."""
    if t == 0.0:
        return -np.sin(np.pi * x)
    y = x[:, None] - np.sqrt(4.0 * nu * t) * z[None, :]
    log_w = -np.cos(np.pi * y) / (2.0 * np.pi * nu) - z[None, :] ** 2
    w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
    return -np.sum(np.sin(np.pi * y) * w, axis=1) / np.sum(w, axis=1)


def allen_cahn_reference(
    epsilon: float = np.sqrt(0.0002),
    t_max: float = 5.0,
    nx: int = 257,
    nt: int = 101,
    refine: int = 8,
    boundary_value: float = -1.0,
) -> ReferenceSolution:
    """Allen-Cahn ``u_t = eps² u_xx + u - u³`` on ``[-1, 1]``, ``u0 = x² cos(pi x)``.

    Dirichlet ``u(±1, t) = boundary_value``. The defaults are the
    ``u_t = 0.001 u_xx + 5 (u - u³)`` benchmark on ``t ∈ [0, 1]`` rewritten
    in the time variable ``τ = 5 t``, which matches the reaction coefficient
    of :class:`pinnrl.pdes.allen_cahn.AllenCahnEquation`.

    The solver grid is ``refine`` times finer than the output grid and the
    output points are a subset of it, so no interpolation error enters.

    Args:
        epsilon: Interface parameter in the residual ``u_t - eps² u_xx - u + u³``.
        t_max: Final time.
        nx: Output spatial points on ``[-1, 1]`` (``nx - 1`` must divide evenly).
        nt: Output time points on ``[0, t_max]``.
        refine: Solver-grid refinement factor relative to the output grid.
        boundary_value: Dirichlet value at both ends.

    Returns:
        The reference field on the ``(nx, nt)`` grid.
    """
    n_fine = (nx - 1) * refine + 1
    x_fine = np.linspace(-1.0, 1.0, n_fine)
    dx = x_fine[1] - x_fine[0]
    t = np.linspace(0.0, t_max, nt)
    eps2 = epsilon**2

    u0 = x_fine**2 * np.cos(np.pi * x_fine)

    def rhs(_t: float, v: np.ndarray) -> np.ndarray:
        full = np.concatenate(([boundary_value], v, [boundary_value]))
        u_xx = (full[2:] - 2.0 * full[1:-1] + full[:-2]) / dx**2
        return eps2 * u_xx + v - v**3

    sol = solve_ivp(rhs, (0.0, t_max), u0[1:-1], t_eval=t, method="RK45", rtol=1e-9, atol=1e-11)
    if not sol.success:
        raise RuntimeError(f"Allen-Cahn reference solve failed: {sol.message}")

    u_fine = np.empty((n_fine, nt))
    u_fine[0, :] = boundary_value
    u_fine[-1, :] = boundary_value
    u_fine[1:-1, :] = sol.y
    return ReferenceSolution(x=x_fine[::refine], t=t, u=u_fine[::refine, :])
