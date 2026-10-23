# Convection (linear advection) equation
# Application domains: Transport of a passive scalar
# Complexity: Linear, 1st-order

from typing import Any, Callable, Dict, List

import torch

from .pde_base import PDEBase, PDEConfig


class ConvectionEquation(PDEBase):
    """Linear advection ``u_t + v · ∇u = 0``.

    Parameters: ``velocity`` (a number, or one entry per spatial axis).

    Initial conditions: ``sine`` (alias ``sin``) — the plane wave
    ``u0 = A sin(kπ Σ_d x_d)`` (``amplitude`` A, ``frequency`` k).

    Exact solution: ``sine`` (alias ``sin``) — the initial profile transported
    with the flow, ``u(x, t) = u0(x - v t)``.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def velocity(self) -> List[float]:
        """Advection velocity, one entry per spatial axis."""
        velocity = self.get_parameter("velocity", default=1.0)
        if isinstance(velocity, (int, float)) or (
            isinstance(velocity, torch.Tensor) and velocity.ndim == 0  # trainable scalar
        ):
            return [velocity] * self.dimension
        return list(velocity)

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_t + Σ_d v_d u_{x_d}``."""
        x, t, u = self._prepare_inputs(model, x, t)
        grad_u = self._grad(u, x)
        convection = sum(v * grad_u[:, d : d + 1] for d, v in enumerate(self.velocity))
        return self._grad(u, t) + convection

    def _plane_wave(self, x: torch.Tensor) -> torch.Tensor:
        ic = self.config.initial_condition
        a, k = ic.get("amplitude", 1.0), ic.get("frequency", 1.0)
        return a * torch.sin(k * torch.pi * torch.sum(x, dim=1, keepdim=True))

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        if params.get("type") in ("sine", "sin"):
            return lambda x, t: self._plane_wave(x)
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.exact_solution.get("type") in ("sine", "sin") and (
            self.config.initial_condition.get("type") in ("sine", "sin")
        ):
            shift = torch.cat([v * t for v in self.velocity], dim=1)
            return self._plane_wave(x - shift)
        return super()._exact_solution(x, t)
