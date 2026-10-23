# Black-Scholes equation
# Application domains: Option pricing
# Complexity: Linear, 2nd-order, variable coefficients

import math
from typing import Any, Callable, Dict

import torch

from .pde_base import PDEBase, PDEConfig


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


class BlackScholesEquation(PDEBase):
    """Black-Scholes equation in time to maturity ``τ`` (the ``t`` input)::

        V_τ = ½ σ² Σ_d S_d² V_{S_d S_d} + r Σ_d S_d V_{S_d} - r V

    Writing it in ``τ = T - t`` turns the payoff at maturity into the initial
    condition at ``τ = 0``, which is what the PINN loss imposes.

    Parameters: ``sigma`` (volatility, default 0.2), ``r`` (risk-free rate,
    default 0.05).

    Initial condition: ``option`` (alias ``call_option``) — the payoff
    ``max(S - K, 0)`` for ``option_type: call`` or ``max(K - S, 0)`` for ``put``,
    with ``strike`` K (alias ``strike_price``) on ``Σ_d S_d``.

    Exact solution (1-D): ``black_scholes`` — the closed-form European price
    ``S Φ(d1) - K e^{-rτ} Φ(d2)`` (call) or ``K e^{-rτ} Φ(-d2) - S Φ(-d1)`` (put)
    for the configured option.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    @property
    def sigma(self):
        """Volatility."""
        return self.get_parameter("sigma", default=0.2)

    @property
    def r(self):
        """Risk-free interest rate."""
        return self.get_parameter("r", default=0.05)

    def _option(self):
        ic = self.config.initial_condition
        strike = float(ic.get("strike", ic.get("strike_price", 100.0)))
        return strike, ic.get("option_type", "call")

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """``V_τ - ½σ² Σ S² V_SS - r Σ S V_S + r V``."""
        d = self.compute_derivatives(
            model, x, t, temporal_derivatives=[1], spatial_derivatives=[1, 2]
        )
        if self.dimension == 1:
            v_s, v_ss = d["dx"], d["dx2"]
        else:
            v_s = torch.cat([d[f"dx{i + 1}"] for i in range(self.dimension)], dim=1)
            v_ss = torch.cat([d["d" + f"x{i + 1}" * 2] for i in range(self.dimension)], dim=1)
        return (
            d["dt"]
            - 0.5 * self.sigma**2 * torch.sum(x**2 * v_ss, dim=1, keepdim=True)
            - self.r * torch.sum(x * v_s, dim=1, keepdim=True)
            + self.r * d["u"]
        )

    def _create_initial_condition(self, params: Dict[str, Any]) -> Callable:
        if params.get("type") in ("option", "call_option"):
            strike, option_type = self._option()
            sign = 1.0 if option_type == "call" else -1.0
            return lambda x, t: torch.clamp(
                sign * (torch.sum(x, dim=1, keepdim=True) - strike), min=0.0
            )
        return super()._create_initial_condition(params)

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self.config.exact_solution.get("type") != "black_scholes" or self.dimension != 1:
            return super()._exact_solution(x, t)
        strike, option_type = self._option()
        tau = torch.clamp(t, min=1e-12)
        vol = self.sigma * torch.sqrt(tau)
        d1 = (
            torch.log(torch.clamp(x, min=1e-12) / strike) + (self.r + 0.5 * self.sigma**2) * tau
        ) / vol
        d2 = d1 - vol
        discount = strike * torch.exp(-self.r * tau)
        if option_type == "call":
            price = x * _normal_cdf(d1) - discount * _normal_cdf(d2)
            payoff = torch.clamp(x - strike, min=0.0)
        else:
            price = discount * _normal_cdf(-d2) - x * _normal_cdf(-d1)
            payoff = torch.clamp(strike - x, min=0.0)
        # Prices are non-negative; the clamp only removes float32 cancellation (~1e-6).
        return torch.where(t > 0, torch.clamp(price, min=0.0), payoff)
