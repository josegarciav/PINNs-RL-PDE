# Heat equation
# Application domains: Heat conduction, diffusion processes
# Complexity: Linear, 2nd-order

import math
from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


class HeatEquation(PDEBase):
    """Heat equation ``u_t = alpha * ∇²u``.

    Parameters: ``alpha`` (thermal diffusivity, required).

    Initial conditions, each with the exact solution of the same ``type``
    (``amplitude`` A, ``frequency`` k; ``L_d`` is the length of spatial axis d):

    * ``sin_exp_decay`` (alias ``sine``)::

          u = A · Π_d sin(2π k x_d / L_d) · exp(-alpha · Σ_d (2π k / L_d)² · t)

      Each factor is an eigenfunction of ``∂²/∂x_d²`` with eigenvalue
      ``-(2π k / L_d)²``, so the product decays at the summed rate.
    * ``sine_2d`` (2-D; ``frequency_x`` kx, ``frequency_y`` ky)::

          u = A · sin(kx π x) · sin(ky π y) · exp(-alpha π² (kx² + ky²) t)
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def alpha(self):
        """Thermal diffusivity coefficient."""
        return self.get_parameter("alpha", required=True)

    def _wave_numbers(self, k: float):
        return [2 * math.pi * k / (hi - lo) for lo, hi in self.domain]

    def _calculate_decay_rate(self, k: float):
        """Decay rate ``alpha * Σ_d (2π k / L_d)²`` of the ``sin_exp_decay`` mode."""
        return self.alpha * sum(w**2 for w in self._wave_numbers(k))

    def _sine_mode(self, params: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        a, k = params.get("amplitude", 1.0), params.get("frequency", 2.0)
        mode = torch.ones_like(x[:, 0:1])
        for d, w in enumerate(self._wave_numbers(k)):
            mode = mode * torch.sin(w * x[:, d : d + 1])
        return a * mode

    def _sine_2d_mode(self, params: Dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        a = params.get("amplitude", 1.0)
        kx, ky = params.get("frequency_x", 2.0), params.get("frequency_y", 2.0)
        return a * torch.sin(kx * torch.pi * x[:, 0:1]) * torch.sin(ky * torch.pi * x[:, 1:2])

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_t - alpha ∇²u``."""
        d = self.compute_derivatives(model, x, t, temporal_derivatives=[1], spatial_derivatives=[2])
        return d["dt"] - self.alpha * d["laplacian"]

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        ic_type = params.get("type")
        if ic_type in ("sin_exp_decay", "sine"):
            return lambda x, t: self._sine_mode(params, x)
        if ic_type == "sine_2d" and self.dimension == 2:
            return lambda x, t: self._sine_2d_mode(params, x)
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        params = self.config.initial_condition
        solution_type = self.config.exact_solution.get("type")
        if solution_type in ("sin_exp_decay", "sine") and params.get("type") in (
            "sin_exp_decay",
            "sine",
        ):
            decay = self._calculate_decay_rate(params.get("frequency", 2.0))
            return self._sine_mode(params, x) * torch.exp(-decay * t)
        if solution_type == "sine_2d" and params.get("type") == "sine_2d" and self.dimension == 2:
            kx, ky = params.get("frequency_x", 2.0), params.get("frequency_y", 2.0)
            decay = self.alpha * torch.pi**2 * (kx**2 + ky**2)
            return self._sine_2d_mode(params, x) * torch.exp(-decay * t)
        return super()._exact_solution(x, t)
