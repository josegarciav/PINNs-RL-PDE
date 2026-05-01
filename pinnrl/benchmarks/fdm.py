"""Classical finite-difference solvers for the 1-D heat and wave equations.

These are deliberately small, self-contained NumPy routines used only as
PINN accuracy baselines from the ``pinnrl-benchmark`` CLI. They are not
production solvers — for those, see :mod:`pinnrl.numerical_solvers` or a
dedicated package such as FEniCS.

Both solvers integrate the PDE on a periodic spatial domain, which matches
the boundary conditions used by the analytical solutions in
:class:`pinnrl.pdes.heat_equation.HeatEquation` and
:class:`pinnrl.pdes.wave_equation.WaveEquation`. They return both the full
``(nt, nx)`` solution grid and the L2 / max errors against the closed-form
solution at the final time, which is what the CLI prints.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class FDMResult:
    """Output bundle from an FDM run.

    Attributes:
        u: Predicted solution grid, shape ``(nt, nx)``.
        u_exact_final: Analytical solution at ``t = t_max`` on the same
            spatial grid, shape ``(nx,)``.
        x: Spatial grid, shape ``(nx,)``.
        t: Time grid, shape ``(nt,)``.
        l2_error: Mean-squared error between FDM and analytical at final time.
        max_error: Inf-norm error between FDM and analytical at final time.
        wall_time_s: Solver wall time in seconds (excluding setup).
    """

    u: np.ndarray
    u_exact_final: np.ndarray
    x: np.ndarray
    t: np.ndarray
    l2_error: float
    max_error: float
    wall_time_s: float


def _heat_exact(x: np.ndarray, t: float, alpha: float, frequency: float) -> np.ndarray:
    """Analytical heat solution ``A·exp(-α·k²·t)·sin(k·x)`` with ``k = 2πf``."""
    k = 2.0 * np.pi * frequency
    return np.exp(-alpha * k**2 * t) * np.sin(k * x)


def solve_heat_1d(
    alpha: float = 0.1,
    frequency: float = 1.0,
    domain: Tuple[float, float] = (0.0, 1.0),
    t_max: float = 1.0,
    nx: int = 101,
    nt: int = 4001,
) -> FDMResult:
    """Explicit-Euler FDM solver for the 1-D heat equation on a periodic domain.

    Solves ``u_t = α·u_xx`` with initial condition ``sin(2π·f·x)`` and
    periodic boundaries. The default ``(nx, nt)`` keeps the diffusion
    number ``r = α·dt/dx²`` well below the explicit-stability cap of 0.5.

    Args:
        alpha: Thermal diffusivity.
        frequency: Spatial frequency of the sinusoidal initial condition.
        domain: Spatial domain ``(x_min, x_max)``.
        t_max: Final time.
        nx: Number of spatial grid points (``dx = (x_max - x_min) / (nx - 1)``).
        nt: Number of time steps.

    Returns:
        :class:`FDMResult` containing the full solution grid, the analytical
        solution at ``t_max``, and L2/max error metrics.

    Raises:
        ValueError: If the explicit-Euler scheme would be unstable for the
            chosen ``(nx, nt, alpha)``.
    """
    x_min, x_max = float(domain[0]), float(domain[1])
    x = np.linspace(x_min, x_max, nx, dtype=np.float64)
    t = np.linspace(0.0, t_max, nt, dtype=np.float64)
    dx = (x_max - x_min) / (nx - 1)
    dt = t_max / (nt - 1)
    r = alpha * dt / (dx**2)
    if r > 0.5:
        raise ValueError(
            f"FDM heat scheme unstable: r = α·dt/dx² = {r:.3f} > 0.5. "
            f"Increase nt (currently {nt}) or decrease nx (currently {nx})."
        )

    u = np.zeros((nt, nx), dtype=np.float64)
    u[0] = np.sin(2.0 * np.pi * frequency * x)

    start = time.perf_counter()
    for n in range(nt - 1):
        # Periodic stencil via np.roll: u_xx ≈ (u[i+1] - 2u[i] + u[i-1]) / dx²
        lap = np.roll(u[n], -1) - 2.0 * u[n] + np.roll(u[n], 1)
        u[n + 1] = u[n] + r * lap
    wall = time.perf_counter() - start

    u_exact_final = _heat_exact(x, t_max, alpha, frequency)
    err = u[-1] - u_exact_final
    return FDMResult(
        u=u,
        u_exact_final=u_exact_final,
        x=x,
        t=t,
        l2_error=float(np.mean(err**2)),
        max_error=float(np.max(np.abs(err))),
        wall_time_s=wall,
    )


def _wave_exact(x: np.ndarray, t: float, c: float) -> np.ndarray:
    """Analytical 1-D wave solution ``sin(2π(x − c·t))`` matching ``WaveEquation``."""
    return np.sin(2.0 * np.pi * (x - c * t))


def solve_wave_1d(
    c: float = 1.0,
    domain: Tuple[float, float] = (0.0, 1.0),
    t_max: float = 1.0,
    nx: int = 201,
    nt: int = 4001,
) -> FDMResult:
    """Leap-frog FDM solver for the 1-D wave equation on a periodic domain.

    Solves ``u_tt = c²·u_xx`` with the travelling-wave initial condition
    ``u(x, 0) = sin(2πx)``, ``u_t(x, 0) = -2π·c·cos(2πx)`` so that the
    closed-form solution is ``sin(2π(x − c·t))`` — matching
    :meth:`pinnrl.pdes.wave_equation.WaveEquation.exact_solution`.

    Args:
        c: Wave speed.
        domain: Spatial domain ``(x_min, x_max)``.
        t_max: Final time.
        nx: Number of spatial grid points.
        nt: Number of time steps.

    Returns:
        :class:`FDMResult` containing the full solution grid, the analytical
        solution at ``t_max``, and L2/max error metrics.

    Raises:
        ValueError: If the leap-frog CFL condition ``c·dt/dx > 1`` is violated.
    """
    x_min, x_max = float(domain[0]), float(domain[1])
    x = np.linspace(x_min, x_max, nx, dtype=np.float64)
    t = np.linspace(0.0, t_max, nt, dtype=np.float64)
    dx = (x_max - x_min) / (nx - 1)
    dt = t_max / (nt - 1)
    cfl = c * dt / dx
    if cfl > 1.0:
        raise ValueError(
            f"FDM wave scheme violates CFL: c·dt/dx = {cfl:.3f} > 1. "
            f"Increase nt (currently {nt}) or decrease nx (currently {nx})."
        )

    u = np.zeros((nt, nx), dtype=np.float64)
    u[0] = np.sin(2.0 * np.pi * x)
    # First step from u_t(x, 0) = -2πc·cos(2πx). Taylor expansion:
    #   u(x, dt) ≈ u(x, 0) + dt·u_t + 0.5·dt²·u_tt   with u_tt = c²·u_xx
    u_t0 = -2.0 * np.pi * c * np.cos(2.0 * np.pi * x)
    lap0 = np.roll(u[0], -1) - 2.0 * u[0] + np.roll(u[0], 1)
    u[1] = u[0] + dt * u_t0 + 0.5 * (cfl**2) * lap0

    start = time.perf_counter()
    for n in range(1, nt - 1):
        lap = np.roll(u[n], -1) - 2.0 * u[n] + np.roll(u[n], 1)
        u[n + 1] = 2.0 * u[n] - u[n - 1] + (cfl**2) * lap
    wall = time.perf_counter() - start

    u_exact_final = _wave_exact(x, t_max, c)
    err = u[-1] - u_exact_final
    return FDMResult(
        u=u,
        u_exact_final=u_exact_final,
        x=x,
        t=t,
        l2_error=float(np.mean(err**2)),
        max_error=float(np.max(np.abs(err))),
        wall_time_s=wall,
    )
