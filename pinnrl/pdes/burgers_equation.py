# Burgers' equation
# Application domains: Fluid dynamics, traffic flow, shock waves
# Complexity: Nonlinear, 2nd-order

from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


class BurgersEquation(PDEBase):
    """Viscous Burgers equation ``u_t + u (Σ_d u_{x_d}) = nu ∇²u``.

    Parameters: ``nu`` (viscosity; ``viscosity`` is accepted as an alias).

    Initial conditions: ``sine`` — ``u0 = A Π_d sin(kπ x_d)`` (``amplitude`` A,
    default -1; ``frequency`` k, default 1).

    Exact solution (1-D): ``cole_hopf`` — the Cole-Hopf integral for the
    configured ``sine`` initial condition and ``nu``, evaluated by
    :func:`pinnrl.benchmarks.reference.burgers_cole_hopf`.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def nu(self):
        """Viscosity coefficient."""
        name = "viscosity" if "viscosity" in (self.config.parameters or {}) else "nu"
        return self.get_parameter(name, default=0.01)

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``u_t + u Σ_d u_{x_d} - nu ∇²u``."""
        d = self.compute_derivatives(
            model, x, t, temporal_derivatives=[1], spatial_derivatives=[1, 2]
        )
        u = d["u"]
        if self.dimension == 1:
            convection = u * d["dx"]
        else:
            convection = sum(u * d[f"dx{dim + 1}"] for dim in range(self.dimension))
        return d["dt"] + convection - self.nu * d["laplacian"]

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        if params.get("type") == "sine":
            a, k = params.get("amplitude", -1.0), params.get("frequency", 1.0)
            return lambda x, t: a * torch.prod(torch.sin(k * torch.pi * x), dim=1, keepdim=True)
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        ic = self.config.initial_condition
        if (
            self.config.exact_solution.get("type") == "cole_hopf"
            and self.dimension == 1
            and ic.get("type") == "sine"
        ):
            from pinnrl.benchmarks.reference import burgers_cole_hopf

            u = burgers_cole_hopf(
                x.detach().cpu().numpy(),
                t.detach().cpu().numpy(),
                float(self.nu),
                amplitude=ic.get("amplitude", -1.0),
                frequency=ic.get("frequency", 1.0),
            )
            return torch.as_tensor(u, dtype=x.dtype, device=x.device)
        return super()._exact_solution(x, t)
