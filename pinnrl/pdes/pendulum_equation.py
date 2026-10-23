"""Pendulum equation.

Application domains: Classical mechanics, nonlinear oscillators. Complexity: Nonlinear ODE, 2nd-order in time.
"""

from collections.abc import Callable
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from .pde_base import PDEBase, PDEConfig


@lru_cache(maxsize=16)
def _pendulum_trajectory(theta0, omega0, damping, g_over_l, t_max):
    """Dense ``θ(t)`` of ``θ'' + b θ' + (g/L) sin θ = 0`` (tight-tolerance RK45)."""
    from scipy.integrate import solve_ivp

    def rhs(_t, y):
        return [y[1], -damping * y[1] - g_over_l * np.sin(y[0])]

    return solve_ivp(
        rhs, (0.0, t_max), [theta0, omega0], rtol=1e-11, atol=1e-12, dense_output=True
    ).sol


class PendulumEquation(PDEBase):
    """Damped nonlinear pendulum ``θ'' + b θ' + (g/L) sin θ = 0``.

    The angle ``θ(t)`` is the network output; the spatial input is a dummy
    coordinate the solution does not depend on.

    Parameters: ``g`` (default 9.81), ``L`` (default 1), ``damping`` b (default 0).

    Second order in time: the initial loss enforces ``θ(0) = initial_angle`` and
    ``θ'(0) = velocity`` (default 0).

    Initial condition: ``small_angle`` (alias ``fixed``) with ``initial_angle``
    (alias ``value``, default 0.5).

    Exact solution: ``ode`` — the full nonlinear damped trajectory from the
    configured initial angle and velocity, integrated with ``scipy`` to 1e-11
    relative tolerance (there is no elementary closed form; the small-angle
    formula ``θ0 cos(ωt)`` is only an approximation).
    """

    time_order = 2

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def g(self):
        """Gravitational acceleration."""
        return self.get_parameter("g", default=9.81)

    @property
    def L(self):
        """Pendulum length."""
        return self.get_parameter("L", default=1.0)

    @property
    def damping(self):
        """Linear damping coefficient b."""
        return self.get_parameter("damping", default=0.0)

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``θ'' + b θ' + (g/L) sin θ``."""
        d = self.compute_derivatives(model, x, t, temporal_derivatives=[1, 2])
        return d["dt2"] + self.damping * d["dt"] + (self.g / self.L) * torch.sin(d["u"])

    def _create_initial_condition(self, params: dict[str, Any]) -> Callable:
        if params.get("type") in ("small_angle", "fixed"):
            theta0 = float(params.get("initial_angle", params.get("value", 0.5)))
            return lambda x, t: torch.full_like(x[:, 0:1], theta0)
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.exact_solution.get("type") != "ode":
            return super()._exact_solution(x, t)
        ic = self.config.initial_condition
        trajectory = _pendulum_trajectory(
            float(ic.get("initial_angle", ic.get("value", 0.5))),
            float(ic.get("velocity", 0.0)),
            float(self.damping),
            float(self.g) / float(self.L),
            float(self.time_domain[1]),
        )
        theta = trajectory(t.detach().cpu().numpy().ravel())[0]
        return torch.as_tensor(theta, dtype=t.dtype, device=t.device).reshape(-1, 1)

    def compute_energy(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Mechanical energy per unit mass, ``½ L² θ'² + g L (1 - cos θ)``."""
        d = self.compute_derivatives(model, x, t, temporal_derivatives=[1])
        return 0.5 * self.L**2 * d["dt"] ** 2 + self.g * self.L * (1 - torch.cos(d["u"]))

    def compute_phase_space(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Phase-space coordinates ``(θ, θ')``."""
        d = self.compute_derivatives(model, x, t, temporal_derivatives=[1])
        return d["u"], d["dt"]
