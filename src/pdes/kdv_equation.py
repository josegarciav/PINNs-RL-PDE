# Korteweg-de Vries (KdV) equation
# Application domains: Water waves, plasma physics, nonlinear optics
# Complexity: Nonlinear, 3rd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class KdVEquation(PDEBase):
    """
    Implementation of the Korteweg-de Vries (KdV) Equation: ∂u/∂t + 6u∂u/∂x + ∂³u/∂x³ = 0
    This equation describes shallow water waves and solitons.
    """

    def __init__(
        self,
        config: PDEConfig,
        **kwargs
    ):
        """
        Initialize the KdV equation.

        :param config: PDEConfig instance containing all necessary parameters
        :param kwargs: Additional keyword arguments
        """
        # Initialize speed before calling super().__init__
        self.speed = config.parameters.get("speed", 1.0)  # Default soliton speed
        super().__init__(config)

    def compute_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the KdV equation residual.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
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
            model, x, t,
            spatial_derivatives=[1, 2, 3],  # Need first, second, and third derivatives
            temporal_derivatives=[1]
        )

        # Get the derivatives we need
        u = model(torch.cat([x, t], dim=1))
        u_t = derivatives["dt"]

        if self.dimension == 1:
            # Get spatial derivatives
            u_x = derivatives["dx"]
            u_xxx = derivatives["dx3"]

            # Compute residual (∂u/∂t + 6u∂u/∂x + ∂³u/∂x³)
            residual = u_t + 6 * u * u_x + u_xxx
        else:
            # For higher dimensions, compute derivatives for each dimension
            residual = u_t
            for dim in range(self.dimension):
                dim_name = f"x{dim+1}"
                u_x = derivatives[f"d{dim_name}"]
                u_xxx = derivatives[f"d{dim_name*3}"]
                residual += 6 * u * u_x + u_xxx

        return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (soliton).

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if not self.config.exact_solution:
            return None

        c = torch.tensor(
            self.speed, dtype=x.dtype, device=x.device
        )  # Convert speed to tensor
        
        if self.dimension == 1:
            return 2 * c * (1 / torch.cosh(torch.sqrt(c) * (x - c * t))) ** 2
        else:
            # For higher dimensions, use the sum of coordinates
            x_sum = torch.sum(x, dim=1, keepdim=True)
            return 2 * c * (1 / torch.cosh(torch.sqrt(c) * (x_sum - c * t))) ** 2

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
            ic_type = params.get("type", "soliton")
            if ic_type == "soliton":
                c = torch.tensor(
                    params.get("speed", self.speed), dtype=torch.float32, device=self.device
                )
                if self.dimension == 1:
                    return lambda x, t: 2 * c * (1 / torch.cosh(torch.sqrt(c) * x)) ** 2
                else:
                    return (
                        lambda x, t: 2
                        * c
                        * (
                            1
                            / torch.cosh(
                                torch.sqrt(c) * torch.sum(x, dim=1, keepdim=True)
                            )
                        )
                        ** 2
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
