# Black-Scholes equation
# Application domains: Financial derivatives, option pricing
# Complexity: Nonlinear, 2nd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class BlackScholesEquation(PDEBase):
    """
    Implementation of the Black-Scholes Equation: ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    where V is the option price, S is the stock price, σ is volatility, and r is the risk-free rate.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        """
        Initialize the Black-Scholes Equation.

        :param config: PDEConfig instance containing all necessary parameters
        :param kwargs: Additional keyword arguments
        """
        super().__init__(config)

    def _validate_parameters(self):
        """Validate required parameters for Black-Scholes equation."""
        super()._validate_parameters()
        # Black-Scholes equation requires volatility and risk-free rate
        self.get_parameter("sigma", default=0.2)
        self.get_parameter("r", default=0.05)

    @property
    def sigma(self):
        """Volatility coefficient."""
        return self.get_parameter("sigma", default=0.2)

    @property
    def r(self):
        """Risk-free interest rate."""
        return self.get_parameter("r", default=0.05)

    def compute_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Black-Scholes equation residual.

        :param model: Neural network model
        :param x: Spatial coordinates (stock prices)
        :param t: Time coordinates (time to maturity)
        :return: Residual tensor
        """
        # Ensure tensors require gradients
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)

        # Ensure model parameters require gradients
        for param in model.parameters():
            param.requires_grad_(True)

        # Set model to training mode
        model.train()

        # Get derivatives
        derivatives = self.compute_derivatives(
            model, x, t, spatial_derivatives=[1, 2], temporal_derivatives=[1]
        )

        # Get the derivatives we need
        V_t = derivatives["dt"]
        V_S = derivatives["dx"] if self.dimension == 1 else derivatives["dx1"]
        V_SS = derivatives["dx2"] if self.dimension == 1 else derivatives["dx1x1"]

        # Black-Scholes equation: V_t + (1/2)σ²S²V_SS + rSV_S - rV = 0
        V = model(torch.cat([x, t], dim=1))

        if self.dimension == 1:
            residual = (
                V_t + 0.5 * self.sigma**2 * x**2 * V_SS + self.r * x * V_S - self.r * V
            )
        else:
            # For higher dimensions, sum over all dimensions
            residual = (
                V_t
                + 0.5 * self.sigma**2 * torch.sum(x**2 * V_SS, dim=1, keepdim=True)
                + self.r * torch.sum(x * V_S, dim=1, keepdim=True)
                - self.r * V
            )

        return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (European call option).

        :param x: Spatial coordinates (stock prices)
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if not self.config.exact_solution:
            return None

        K = self.config.exact_solution.get("strike_price", 1.0)  # Strike price

        if self.dimension == 1:
            # 1D Black-Scholes solution (European call option)
            d1 = (torch.log(x / K) + (self.r + 0.5 * self.sigma**2) * t) / (
                self.sigma * torch.sqrt(t)
            )
            d2 = d1 - self.sigma * torch.sqrt(t)
            return x * torch.erf(d1) - K * torch.exp(-self.r * t) * torch.erf(d2)
        else:
            # For higher dimensions, use product of 1D solutions
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                d1 = (
                    torch.log(x[:, dim : dim + 1] / K)
                    + (self.r + 0.5 * self.sigma**2) * t
                ) / (self.sigma * torch.sqrt(t))
                d2 = d1 - self.sigma * torch.sqrt(t)
                solution *= x[:, dim : dim + 1] * torch.erf(d1) - K * torch.exp(
                    -self.r * t
                ) * torch.erf(d2)
            return solution

    def _create_boundary_condition(
        self, bc_type: str, params: Dict[str, Any]
    ) -> callable:
        """
        Create boundary condition function from parameters.

        :param bc_type: Type of boundary condition
        :param params: Parameters for the boundary condition
        :return: Boundary condition function
        """
        if bc_type == "initial":
            ic_type = params.get("type", "call_option")
            if ic_type == "call_option":
                K = params.get("strike_price", 1.0)
                if self.dimension == 1:
                    return lambda x, t: torch.maximum(x - K, torch.zeros_like(x))
                else:
                    return lambda x, t: torch.maximum(
                        torch.sum(x, dim=1, keepdim=True) - K,
                        torch.zeros_like(x[:, 0:1]),
                    )
            else:
                raise ValueError(f"Unsupported initial condition type: {ic_type}")
        else:
            return super()._create_boundary_condition(bc_type, params)

    def validate(self, model, num_points=1000):
        """
        Validate the model's solution against exact solution.

        :param model: Neural network model
        :param num_points: Number of validation points
        :return: Dictionary of error metrics
        """
        x, t = self.generate_collocation_points(num_points)
        u_pred = model(torch.cat([x, t], dim=1))
        u_exact = self.exact_solution(x, t)
        error = torch.abs(u_pred - u_exact)
        return {
            "l2_error": torch.mean(error**2).item(),
            "max_error": torch.max(error).item(),
            "mean_error": torch.mean(error).item(),
        }
