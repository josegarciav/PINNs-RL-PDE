# Burgers' equation
# Application domains: Turbulence, traffic modeling, shock waves.
# Complexity: Nonlinear, 2nd-order


import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class BurgersEquation(PDEBase):
    """
    Implementation of the Burgers' Equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x²
    where ν is the kinematic viscosity and u is the velocity field.
    This equation combines nonlinear convection and diffusion.
    """

    def __init__(
        self,
        nu: float,
        domain: Union[Tuple[float, float], List[Tuple[float, float]]],
        time_domain: Tuple[float, float],
        boundary_conditions: Dict[str, Dict[str, Any]],
        initial_condition: Dict[str, Any],
        exact_solution: Dict[str, Any],
        dimension: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Burgers' Equation.

        :param nu: Kinematic viscosity
        :param domain: Spatial domain (tuple for 1D, list of tuples for higher dimensions)
        :param time_domain: Temporal domain
        :param boundary_conditions: Dictionary of boundary conditions
        :param initial_condition: Dictionary of initial condition parameters
        :param exact_solution: Dictionary of exact solution parameters
        :param dimension: Problem dimension (1 for 1D, 2 for 2D, etc.)
        :param device: Device to use for computations
        """
        config = PDEConfig(
            name="Burgers' Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={"nu": nu},
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=dimension,
            device=device,
        )
        super().__init__(config)
        self.nu = nu

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the Burgers' equation residual.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Residual tensor
        """
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)

        # Compute derivatives
        u = model(xt)
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        # Compute first and second derivatives with respect to x
        if self.dimension == 1:
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            u_xx = torch.autograd.grad(
                u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
            )[0]
            convection = u * u_x
            diffusion = self.nu * u_xx
        else:
            # For higher dimensions, compute derivatives for each dimension
            convection = torch.zeros_like(u)
            diffusion = torch.zeros_like(u)
            for dim in range(self.dimension):
                u_x = torch.autograd.grad(
                    u,
                    x[:, dim : dim + 1],
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                )[0]
                u_xx = torch.autograd.grad(
                    u_x,
                    x[:, dim : dim + 1],
                    grad_outputs=torch.ones_like(u_x),
                    create_graph=True,
                )[0]
                convection += u * u_x
                diffusion += self.nu * u_xx

        # Compute residual (∂u/∂t + u∂u/∂x = ν∂²u/∂x²)
        return u_t + convection - diffusion

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (traveling wave).

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            # 1D Burgers' solution (traveling wave)
            c = 1.0  # Wave speed
            return torch.tanh((x - c * t) / (2 * self.nu))
        else:
            # For higher dimensions, use product of traveling waves
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                c = 1.0  # Wave speed
                solution *= torch.tanh((x[:, dim : dim + 1] - c * t) / (2 * self.nu))
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
                    return lambda x, t: torch.tanh(x / (2 * self.nu))
                else:
                    return lambda x, t: torch.tanh(
                        torch.sum(x, dim=1, keepdim=True) / (2 * self.nu)
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
