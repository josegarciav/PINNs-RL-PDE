# Korteweg-de Vries (KdV) equation
# Application domains: Shallow-water waves, plasma physics, solitons
# Complexity: Nonlinear, 3rd-order

import math
from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


class KdVEquation(PDEBase):
    """Korteweg-de Vries equation ``u_t + alpha u u_x + beta u_xxx = 0`` (1-D).

    Parameters: ``alpha`` (nonlinearity, default 6), ``beta`` (dispersion,
    default 1).

    Initial condition and exact solution: ``soliton`` with ``speed`` c (default 1)
    and optional ``position`` x0::

        u = (3c / alpha) · sech²( ½ √(c / beta) · (x - x0 - c t) )

    Substituting ``u = f(x - c t)`` and integrating twice shows this travels at
    exactly ``c`` and vanishes far from ``x0``.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        if config.dimension != 1:
            raise ValueError("KdVEquation is 1-D only")
        super().__init__(config)

    @property
    def alpha(self):
        """Nonlinear coefficient."""
        return self.get_parameter("alpha", default=6.0)

    @property
    def beta(self):
        """Dispersion coefficient."""
        return self.get_parameter("beta", default=1.0)

    @property
    def speed(self):
        """Speed of the soliton in the initial condition."""
        return float(self.config.initial_condition.get("speed", 1.0))

    def _soliton(self, params: Dict[str, Any], x: torch.Tensor, t) -> torch.Tensor:
        c = float(params.get("speed", 1.0))
        x0 = params.get("position", 0.0)
        width = 0.5 * math.sqrt(float(c) / float(self.beta))
        return (3 * c / self.alpha) / torch.cosh(width * (x - x0 - c * t)) ** 2

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_t + alpha u u_x + beta u_xxx``."""
        d = self.compute_derivatives(
            model, x, t, temporal_derivatives=[1], spatial_derivatives=[1, 3]
        )
        return d["dt"] + self.alpha * d["u"] * d["dx"] + self.beta * d["dx3"]

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        if params.get("type") == "soliton":
            return lambda x, t: self._soliton(params, x, 0.0)
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.exact_solution.get("type") == "soliton":
            return self._soliton(self.config.initial_condition, x, t)
        return super()._exact_solution(x, t)
