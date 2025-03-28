# Heat equation
# Application domains: Heat conduction, diffusion processes
# Complexity: Linear, 2nd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class HeatEquation(PDEBase):
    """
    Implementation of the Heat Equation: ∂u/∂t = α∇²u
    where α is the thermal diffusivity and ∇² is the Laplacian operator.
    """

    def __init__(
        self,
        config: PDEConfig,
        **kwargs
    ):
        """
        Initialize the Heat Equation.

        :param config: PDEConfig instance containing all necessary parameters
        :param kwargs: Additional keyword arguments
        """
        super().__init__(config)
        self.alpha = self.config.parameters.get("alpha", 0.01)

    def compute_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the residual of the heat equation.
        
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
            spatial_derivatives=[2],
            temporal_derivatives=[1]
        )

        # Heat equation residual: u_t - α∇²u = 0
        u_t = derivatives["dt"]
        laplacian = derivatives["laplacian"]
        
        residual = u_t - self.alpha * laplacian
        return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution for the heat equation.

        The solution is of the form:
        u(x,t) = A * exp(-alpha * k^2 * pi^2 * t) * sin(k * pi * x)

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if not self.config.exact_solution:
            return None

        solution_type = self.config.exact_solution.get("type", "sine")
        if solution_type == "sine":
            A = self.config.exact_solution.get("amplitude", 1.0)
            k = self.config.exact_solution.get("frequency", 1.0)

            if self.dimension == 1:
                time_factor = torch.exp(-self.alpha * (k * torch.pi) ** 2 * t)
                space_factor = torch.sin(k * torch.pi * x)
                return A * time_factor * space_factor
            else:
                # For higher dimensions, use product of sine waves
                solution = torch.ones_like(x[:, 0:1])
                for dim in range(self.dimension):
                    time_factor = torch.exp(-self.alpha * (k * torch.pi) ** 2 * t)
                    space_factor = torch.sin(k * torch.pi * x[:, dim:dim+1])
                    solution *= A * time_factor * space_factor
                return solution
        elif solution_type == "sine_2d" and self.dimension == 2:
            A = self.config.exact_solution.get("amplitude", 1.0)
            kx = self.config.exact_solution.get("frequency_x", 1.0)
            ky = self.config.exact_solution.get("frequency_y", 1.0)
            
            time_factor = torch.exp(-self.alpha * ((kx * torch.pi) ** 2 + (ky * torch.pi) ** 2) * t)
            space_factor = torch.sin(kx * torch.pi * x[:, 0:1]) * torch.sin(ky * torch.pi * x[:, 1:2])
            return A * time_factor * space_factor
        else:
            raise ValueError(f"Unsupported exact solution type: {solution_type}")

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
            ic_type = params.get("type", "sine")
            if ic_type == "sine":
                A = params.get("amplitude", 1.0)
                k = params.get("frequency", 1.0)
                if self.dimension == 1:
                    return lambda x, t: A * torch.sin(k * torch.pi * x)
                else:
                    return lambda x, t: A * torch.prod(
                        torch.sin(k * torch.pi * x), dim=1, keepdim=True
                    )
            elif ic_type == "sine_2d" and self.dimension == 2:
                A = params.get("amplitude", 1.0)
                kx = params.get("frequency_x", 1.0)
                ky = params.get("frequency_y", 1.0)
                return lambda x, t: A * torch.sin(kx * torch.pi * x[:, 0:1]) * torch.sin(ky * torch.pi * x[:, 1:2])
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
