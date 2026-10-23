# Wave equation
# Application domains: Acoustics, vibrating strings and membranes
# Complexity: Linear, 2nd-order in time and space

import math
from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


class WaveEquation(PDEBase):
    """Wave equation ``u_tt = c² ∇²u``.

    Parameters: ``c`` (wave speed, default 1).

    Second order in time: the initial loss also enforces
    ``u_t(x, 0) = initial_condition["velocity"]`` (default 0).

    Initial conditions, each with the exact solution of the same ``type`` —
    standing waves that vanish on the faces of ``[0, 1]^d`` for integer
    frequencies (``amplitude`` A, ``frequency`` k):

    * ``sine``: ``u = A Π_d sin(kπ x_d) · cos(ω t)`` with ``ω = c k π √d``,
      since ``∇²`` of the product is ``-d (kπ)²`` times itself.
    * ``sine_2d`` (``frequency_x`` kx, ``frequency_y`` ky):
      ``u = A sin(kx π x) sin(ky π y) cos(ω t)`` with ``ω = c π √(kx² + ky²)``.

    Both start at rest (``u_t(x, 0) = 0``).
    """

    time_order = 2

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def c(self):
        """Wave speed."""
        return self.get_parameter("c", default=1.0)

    def _mode(self, params: Dict[str, Any], x: torch.Tensor):
        """Spatial shape and angular frequency of the configured standing wave."""
        a = params.get("amplitude", 1.0)
        if params.get("type") == "sine_2d":
            kx, ky = params.get("frequency_x", 2.0), params.get("frequency_y", 2.0)
            shape = torch.sin(kx * torch.pi * x[:, 0:1]) * torch.sin(ky * torch.pi * x[:, 1:2])
            return a * shape, self.c * math.pi * math.sqrt(kx**2 + ky**2)
        k = params.get("frequency", 1.0)
        shape = torch.prod(torch.sin(k * torch.pi * x), dim=1, keepdim=True)
        return a * shape, self.c * k * math.pi * math.sqrt(self.dimension)

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_tt - c² ∇²u``."""
        d = self.compute_derivatives(model, x, t, temporal_derivatives=[2], spatial_derivatives=[2])
        return d["dt2"] - self.c**2 * d["laplacian"]

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        if params.get("type") == "sine" or (
            params.get("type") == "sine_2d" and self.dimension == 2
        ):
            return lambda x, t: self._mode(params, x)[0]
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        params = self.config.initial_condition
        if self.config.exact_solution.get("type") == params.get("type") and (
            params.get("type") == "sine"
            or (params.get("type") == "sine_2d" and self.dimension == 2)
        ):
            shape, omega = self._mode(params, x)
            return shape * torch.cos(omega * t)
        return super()._exact_solution(x, t)
