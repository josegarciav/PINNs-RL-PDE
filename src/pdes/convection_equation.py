# Convection equation
# Application domains: Fluid dynamics, transport phenomena
# Complexity: Linear, 1st-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent


class ConvectionEquation(PDEBase):
    """
    Implementation of the Convection Equation: ∂u/∂t + v·∇u = 0
    where v is the velocity vector and ∇ is the gradient operator.
    """

    def __init__(
        self,
        velocity: Union[float, List[float]],
        domain: Union[Tuple[float, float], List[Tuple[float, float]]],
        time_domain: Tuple[float, float],
        boundary_conditions: Dict[str, Dict[str, Any]],
        initial_condition: Dict[str, Any],
        exact_solution: Dict[str, Any],
        dimension: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Convection Equation.

        :param velocity: Velocity (scalar for 1D, list for higher dimensions)
        :param domain: Spatial domain (tuple for 1D, list of tuples for higher dimensions)
        :param time_domain: Temporal domain
        :param boundary_conditions: Dictionary of boundary conditions
        :param initial_condition: Dictionary of initial condition parameters
        :param exact_solution: Dictionary of exact solution parameters
        :param dimension: Problem dimension (1 for 1D, 2 for 2D, etc.)
        :param device: Device to use for computations
        """
        # Convert velocity to list if scalar
        if isinstance(velocity, (int, float)):
            velocity = [velocity] * dimension

        config = PDEConfig(
            name="Convection Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={"velocity": velocity},
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=dimension,
            device=device,
        )
        super().__init__(config)
        self.velocity = velocity

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the convection equation residual.

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

        # Compute gradient based on dimension
        if self.dimension == 1:
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            convection = self.velocity[0] * u_x
        else:
            # For higher dimensions, compute dot product of velocity and gradient
            convection = torch.zeros_like(u)
            for dim in range(self.dimension):
                u_x = torch.autograd.grad(
                    u,
                    x[:, dim : dim + 1],
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                )[0]
                convection += self.velocity[dim] * u_x

        return u_t + convection

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution.

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            return torch.sin(2 * torch.pi * (x - self.velocity[0] * t))
        else:
            # For higher dimensions, use product of sine waves
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                solution *= torch.sin(
                    2 * torch.pi * (x[:, dim : dim + 1] - self.velocity[dim] * t)
                )
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
            ic_type = params.get("type", "sine")
            if ic_type == "sine":
                A = params.get("amplitude", 1.0)
                k = params.get("frequency", 2.0)
                if self.dimension == 1:
                    return lambda x, t: A * torch.sin(k * torch.pi * x)
                else:
                    return lambda x, t: A * torch.sin(
                        k * torch.pi * torch.sum(x, dim=1, keepdim=True)
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
