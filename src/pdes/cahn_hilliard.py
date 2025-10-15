# Cahn-Hilliard equation
# Application domains: Phase separation, material science, pattern formation
# Complexity: Nonlinear, 4th-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class CahnHilliardEquation(PDEBase):
    """
    Implementation of the Cahn-Hilliard Equation: ∂u/∂t = ∇²(ε²∇²u + u - u³)
    where ε is the interface width parameter and ∇² is the Laplacian operator.
    This equation describes phase separation and coarsening in binary mixtures.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        """
        Initialize the Cahn-Hilliard Equation.

        :param config: PDEConfig instance containing all necessary parameters
        :param kwargs: Additional keyword arguments
        """
        super().__init__(config)

    def _validate_parameters(self):
        """Validate required parameters for Cahn-Hilliard equation."""
        super()._validate_parameters()
        # Cahn-Hilliard equation requires epsilon parameter
        self.get_parameter("epsilon", default=0.1)

    @property
    def epsilon(self):
        """Interface width parameter."""
        return self.get_parameter("epsilon", default=0.1)

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Cahn-Hilliard equation residual.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Residual tensor
        """
        # Ensure input tensors require gradients
        x = x.requires_grad_(True)
        t = t.requires_grad_(True)

        # Combine inputs
        xt = torch.cat([x, t], dim=1)

        # Get model prediction
        u = model(xt)

        # Compute time derivative
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True, allow_unused=True
        )[0]
        if u_t is None:
            u_t = torch.zeros_like(u)

        # Compute chemical potential μ = -ε²∇²u + f'(u)
        # where f'(u) = u³ - u is the derivative of the double-well potential
        if self.dimension == 1:
            u_x = torch.autograd.grad(
                u,
                x,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                allow_unused=True,
            )[0]
            if u_x is None:
                u_x = torch.zeros_like(u)
            u_xx = torch.autograd.grad(
                u_x,
                x,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True,
                allow_unused=True,
            )[0]
            if u_xx is None:
                u_xx = torch.zeros_like(u)
            laplacian = u_xx
        else:
            # For higher dimensions, compute Laplacian as sum of second derivatives
            laplacian = torch.zeros_like(u)
            for dim in range(self.dimension):
                u_x = torch.autograd.grad(
                    u,
                    x[:, dim : dim + 1],
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if u_x is not None:
                    u_xx = torch.autograd.grad(
                        u_x,
                        x[:, dim : dim + 1],
                        grad_outputs=torch.ones_like(u_x),
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    if u_xx is not None:
                        laplacian += u_xx

        # Chemical potential μ = -ε²∇²u + f'(u)
        mu = -self.epsilon**2 * laplacian + u**3 - u

        # Compute ∇²μ
        if self.dimension == 1:
            mu_x = torch.autograd.grad(
                mu,
                x,
                grad_outputs=torch.ones_like(mu),
                create_graph=True,
                allow_unused=True,
            )[0]
            if mu_x is None:
                mu_x = torch.zeros_like(mu)
            mu_xx = torch.autograd.grad(
                mu_x,
                x,
                grad_outputs=torch.ones_like(mu_x),
                create_graph=True,
                allow_unused=True,
            )[0]
            if mu_xx is None:
                mu_xx = torch.zeros_like(mu)
            laplacian_mu = mu_xx
        else:
            # For higher dimensions, compute Laplacian as sum of second derivatives
            laplacian_mu = torch.zeros_like(mu)
            for dim in range(self.dimension):
                mu_x = torch.autograd.grad(
                    mu,
                    x[:, dim : dim + 1],
                    grad_outputs=torch.ones_like(mu),
                    create_graph=True,
                    allow_unused=True,
                )[0]
                if mu_x is not None:
                    mu_xx = torch.autograd.grad(
                        mu_x,
                        x[:, dim : dim + 1],
                        grad_outputs=torch.ones_like(mu_x),
                        create_graph=True,
                        allow_unused=True,
                    )[0]
                    if mu_xx is not None:
                        laplacian_mu += mu_xx

        # Cahn-Hilliard equation: u_t = ∇²μ
        return u_t - laplacian_mu

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (tanh profile).

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            # 1D Cahn-Hilliard solution (tanh profile)
            return torch.tanh(x / (2 * self.epsilon))
        else:
            # For higher dimensions, use product of tanh profiles
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                solution *= torch.tanh(x[:, dim : dim + 1] / (2 * self.epsilon))
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
            ic_type = params.get("type", "tanh")
            if ic_type == "tanh":
                if self.dimension == 1:
                    return lambda x, t: torch.tanh(x / (2 * self.epsilon))
                else:
                    return lambda x, t: torch.tanh(
                        torch.sum(x, dim=1, keepdim=True) / (2 * self.epsilon)
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
