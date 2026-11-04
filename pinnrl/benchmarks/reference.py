"""High-accuracy reference solutions for PDEs without a closed form.

The sampling study scores PINNs against these fields instead of the
``exact_solution`` methods on the PDE classes where those only cover special
cases (the Allen-Cahn ``tanh`` branch is stationary). ``BurgersEquation``'s
``cole_hopf`` solution is :func:`burgers_cole_hopf` from this module.

* :func:`burgers_reference` — Cole-Hopf integral for the standard
  ``u0 = -sin(pi x)`` benchmark (Basdevant et al., 1986), evaluated with a
  log-sum-exp trapezoid rule so the ``exp(±1/(2 pi nu))`` factors never
  overflow.
* :func:`allen_cahn_reference` — second-order finite differences in space
  with an adaptive Runge-Kutta integrator in time (method of lines).
* :func:`heat_variable_reference` — the heat equation with a coefficient that
  varies in space and time, ``u_t = (alpha(x, t) u_x)_x``, in conservative
  form by the method of lines; the reference for recovering ``alpha``.
"""

from __future__ import annotations

from collections.abc import Callable
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

    def interpolate(self, x: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Bilinear interpolation of the field at scattered ``(x, t)`` (any matching shapes)."""
        from scipy.interpolate import RegularGridInterpolator

        interp = RegularGridInterpolator(
            (self.x, self.t), self.u, method="linear", bounds_error=False, fill_value=None
        )
        x = np.asarray(x, dtype=float)
        pts = np.stack([x.reshape(-1), np.asarray(t, dtype=float).reshape(-1)], axis=1)
        return interp(pts).reshape(x.shape)

    def as_exact_solution(self) -> Callable:
        """A callable ``u(x, t) -> (N, 1)`` over torch tensors, for ``CustomPDE(exact_solution=...)``."""
        import torch

        def exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            vals = self.interpolate(x[:, 0:1].detach().cpu().numpy(), t.detach().cpu().numpy())
            return torch.as_tensor(vals, dtype=x.dtype, device=x.device).reshape(-1, 1)

        return exact


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
    u = burgers_cole_hopf(x[:, None], t[None, :], nu, n_quad=n_quad, z_max=z_max)
    return ReferenceSolution(x=x, t=t, u=u)


def burgers_cole_hopf(
    x,
    t,
    nu: float,
    amplitude: float = -1.0,
    frequency: float = 1.0,
    n_quad: int = 6001,
    z_max: float = 15.0,
    chunk: int = 512,
) -> np.ndarray:
    """Exact Burgers solution for ``u0 = amplitude * sin(frequency * pi * x)``.

    Evaluates the Cole-Hopf quotient

        u(x, t) = ∫ u0(y) φ0(y) e^(-z²) dz / ∫ φ0(y) e^(-z²) dz,   y = x - sqrt(4 nu t) z,

    with ``φ0(y) = exp(amplitude cos(frequency pi y) / (2 nu frequency pi))``, by
    the trapezoid rule on ``[-z_max, z_max]``. ``x`` and ``t`` broadcast against
    each other; points with ``t = 0`` return ``u0``. Points are processed in
    chunks so memory stays at ``chunk * n_quad`` floats.
    """
    x, t = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(t, dtype=np.float64))
    shape = x.shape
    x, t = x.ravel(), t.ravel()
    k = frequency * np.pi
    u = amplitude * np.sin(k * x)
    z = np.linspace(-z_max, z_max, n_quad)
    later = np.flatnonzero(t > 0)
    for start in range(0, later.size, chunk):
        idx = later[start : start + chunk]
        y = x[idx, None] - np.sqrt(4.0 * nu * t[idx, None]) * z[None, :]
        log_w = amplitude * np.cos(k * y) / (2.0 * nu * k) - z[None, :] ** 2
        w = np.exp(log_w - log_w.max(axis=1, keepdims=True))
        u[idx] = np.sum(amplitude * np.sin(k * y) * w, axis=1) / np.sum(w, axis=1)
    return u.reshape(shape)


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
        nx: Output spatial points on ``[-1, 1]``.
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


def heat_variable_reference(
    alpha: Callable[[np.ndarray, float], np.ndarray],
    u0: Callable[[np.ndarray], np.ndarray] | None = None,
    domain: tuple[float, float] = (0.0, 1.0),
    t_max: float = 1.0,
    nx: int = 201,
    nt: int = 201,
    refine: int = 4,
    boundary_values: tuple[float, float] = (0.0, 0.0),
) -> ReferenceSolution:
    """Heat equation with a variable coefficient, ``u_t = (alpha(x, t) u_x)_x``, Dirichlet ends.

    The conservative form is discretised with the coefficient at the cell faces,
    ``(alpha_{i+1/2} (u_{i+1} - u_i) - alpha_{i-1/2} (u_i - u_{i-1})) / dx²``,
    and integrated in time with the implicit BDF method (the system is stiff).
    When ``alpha`` depends on ``t`` only, this is ``u_t = alpha(t) u_xx``; the
    solution for ``u0 = sin(pi x)`` is then ``exp(-pi² ∫alpha) sin(pi x)``, which
    the tests check against.

    Args:
        alpha: ``alpha(x, t) -> array like x`` with ``x`` a NumPy array and ``t`` a float.
        u0: Initial condition ``u0(x)``; default ``sin(pi (x - x_lo) / (x_hi - x_lo))``.
        domain: Spatial interval.
        t_max: Final time.
        nx: Output spatial points (including both ends).
        nt: Output time points on ``[0, t_max]``.
        refine: Solver-grid refinement factor relative to the output grid.
        boundary_values: Dirichlet values at ``x_lo`` and ``x_hi``.

    Returns:
        The reference field on the ``(nx, nt)`` output grid.

    Raises:
        RuntimeError: If the time integrator fails.
    """
    x_lo, x_hi = float(domain[0]), float(domain[1])
    n_fine = (nx - 1) * refine + 1
    x_fine = np.linspace(x_lo, x_hi, n_fine)
    dx = x_fine[1] - x_fine[0]
    x_faces = 0.5 * (x_fine[1:] + x_fine[:-1])
    t = np.linspace(0.0, t_max, nt)
    lo_val, hi_val = float(boundary_values[0]), float(boundary_values[1])

    if u0 is None:
        u_init = np.sin(np.pi * (x_fine - x_lo) / (x_hi - x_lo))
    else:
        u_init = np.asarray(u0(x_fine), dtype=float)

    def rhs(tau: float, v: np.ndarray) -> np.ndarray:
        full = np.concatenate(([lo_val], v, [hi_val]))
        a = np.broadcast_to(np.asarray(alpha(x_faces, float(tau)), dtype=float), x_faces.shape)
        flux = a * (full[1:] - full[:-1]) / dx
        return (flux[1:] - flux[:-1]) / dx

    sol = solve_ivp(rhs, (0.0, t_max), u_init[1:-1], t_eval=t, method="BDF", rtol=1e-8, atol=1e-10)
    if not sol.success:
        raise RuntimeError(f"variable-coefficient heat reference failed: {sol.message}")

    u_fine = np.empty((n_fine, nt))
    u_fine[0, :] = lo_val
    u_fine[-1, :] = hi_val
    u_fine[1:-1, :] = sol.y
    return ReferenceSolution(x=x_fine[::refine], t=t, u=u_fine[::refine, :])
