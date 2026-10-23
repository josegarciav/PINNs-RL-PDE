# Allen-Cahn equation
# Application domains: Phase separation, interface motion
# Complexity: Nonlinear reaction-diffusion, 2nd-order

import math
from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


class AllenCahnEquation(PDEBase):
    """Allen-Cahn equation ``u_t = epsilon² ∇²u + r (u - u³)``.

    Parameters: ``epsilon`` (interface width, default 0.1), ``reaction_rate``
    r (default 1).

    Initial conditions:

    * ``tanh`` — the planar kink normal to ``(1, ..., 1)/√d``,
      ``u = tanh(√r · Σ_d x_d / (√(2d) · epsilon))``. It solves
      ``epsilon² φ'' + r (φ - φ³) = 0``, so it is stationary.
    * ``x2_cos`` — the benchmark ``u0 = x² cos(π x)`` (no closed form; score it
      against :func:`pinnrl.benchmarks.reference.allen_cahn_reference`).

    Exact solution: ``tanh`` — the stationary kink above.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def epsilon(self):
        """Interface width parameter."""
        return self.get_parameter("epsilon", default=0.1)

    @property
    def reaction_rate(self):
        """Strength of the double-well reaction term."""
        return self.get_parameter("reaction_rate", default=1.0)

    def _kink(self, x: torch.Tensor) -> torch.Tensor:
        rate = torch.as_tensor(self.reaction_rate, dtype=x.dtype, device=x.device)
        scale = torch.sqrt(rate) / (math.sqrt(2 * self.dimension) * self.epsilon)
        return torch.tanh(scale * torch.sum(x, dim=1, keepdim=True))

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_t - epsilon² ∇²u - r (u - u³)``."""
        d = self.compute_derivatives(model, x, t, temporal_derivatives=[1], spatial_derivatives=[2])
        u = d["u"]
        return d["dt"] - self.epsilon**2 * d["laplacian"] - self.reaction_rate * (u - u**3)

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        ic_type = params.get("type")
        if ic_type == "tanh":
            return lambda x, t: self._kink(x)
        if ic_type == "x2_cos":
            return lambda x, t: x[:, 0:1] ** 2 * torch.cos(torch.pi * x[:, 0:1])
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.exact_solution.get("type") == "tanh":
            return self._kink(x)
        return super()._exact_solution(x, t)
