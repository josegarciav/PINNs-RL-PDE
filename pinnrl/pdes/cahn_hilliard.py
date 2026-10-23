# Cahn-Hilliard equation
# Application domains: Spinodal decomposition, phase separation with mass conservation
# Complexity: Nonlinear, 4th-order

from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


class CahnHilliardEquation(PDEBase):
    """Cahn-Hilliard equation ``u_t = M ∇²μ`` with ``μ = u³ - u - kappa ∇²u``.

    Parameters: ``mobility`` M (alias ``M``, default 1) and ``kappa`` (gradient
    energy; if absent, ``epsilon²`` with ``epsilon`` default 0.1).

    Initial conditions:

    * ``tanh`` — the planar kink ``tanh(Σ_d x_d / √(2 d kappa))``. It makes
      ``μ ≡ 0``, so it is a stationary solution.
    * ``random`` (from the base class) — a smooth, reproducible perturbation for
      spinodal-decomposition runs. No closed form.

    Exact solution: ``tanh`` — the stationary kink above.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def mobility(self):
        """Mobility M."""
        name = "M" if "M" in (self.config.parameters or {}) else "mobility"
        return self.get_parameter(name, default=1.0)

    @property
    def kappa(self):
        """Gradient-energy coefficient (interface width squared)."""
        if "kappa" in (self.config.parameters or {}):
            return self.get_parameter("kappa")
        return self.get_parameter("epsilon", default=0.1) ** 2

    @property
    def epsilon(self):
        """Interface width ``sqrt(kappa)``."""
        return self.kappa**0.5

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_t - M ∇²(u³ - u - kappa ∇²u)``."""
        x, t, u = self._prepare_inputs(model, x, t)
        # Clamp u inside μ so u³ cannot overflow for an untrained network.
        u_clamped = torch.clamp(u, -10.0, 10.0)
        mu = u_clamped**3 - u_clamped - self.kappa * self.laplacian(u, x)
        return self._grad(u, t) - self.mobility * self.laplacian(mu, x)

    def _kink(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(
            torch.sum(x, dim=1, keepdim=True) / (2 * self.dimension * self.kappa) ** 0.5
        )

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        ic_type = params.get("type")
        if ic_type == "tanh":
            return lambda x, t: self._kink(x)
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.exact_solution.get("type") == "tanh":
            return self._kink(x)
        return super()._exact_solution(x, t)
