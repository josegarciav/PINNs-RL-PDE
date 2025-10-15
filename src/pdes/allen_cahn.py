# Allen-Cahn equation
# Application domains: Phase transitions, material science, pattern formation
# Complexity: Nonlinear, 2nd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class AllenCahnEquation(PDEBase):
    """
    Implementation of the Allen-Cahn Equation: ∂u/∂t = ε²∇²u + u - u³
    where ε is the interface width parameter and ∇² is the Laplacian operator.
    This equation describes phase separation in binary alloys.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        """
        Initialize the Allen-Cahn Equation.

        :param config: PDEConfig instance containing all necessary parameters
        :param kwargs: Additional keyword arguments
        """
        super().__init__(config)

    def _validate_parameters(self):
        """Validate required parameters for Allen-Cahn equation."""
        super()._validate_parameters()
        # Allen-Cahn equation requires epsilon parameter
        self.get_parameter("epsilon", default=0.1)

    @property
    def epsilon(self):
        """Interface width parameter."""
        return self.get_parameter("epsilon", default=0.1)

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Allen-Cahn equation residual.

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

        # Compute Laplacian based on dimension
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

        # Compute residual (∂u/∂t = ε²∇²u + u - u³)
        return u_t - self.epsilon**2 * laplacian - u + u**3

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (tanh profile).

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            # 1D Allen-Cahn solution (tanh profile)
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
