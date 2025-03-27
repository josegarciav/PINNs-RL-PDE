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
        domain: Union[Tuple[float, float], List[Tuple[float, float]]],
        time_domain: Tuple[float, float],
        boundary_conditions: Dict[str, Dict[str, Any]],
        initial_condition: Dict[str, Any],
        exact_solution: Dict[str, Any],
        dimension: int = 1,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the KdV equation.

        :param domain: Spatial domain (tuple for 1D, list of tuples for higher dimensions)
        :param time_domain: Temporal domain
        :param boundary_conditions: Dictionary of boundary conditions
        :param initial_condition: Dictionary of initial condition parameters
        :param exact_solution: Dictionary of exact solution parameters
        :param dimension: Problem dimension (1 for 1D, 2 for 2D, etc.)
        :param device: Device to use for computations
        """
        config = PDEConfig(
            name="KdV Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={"speed": 1.0},  # Default soliton speed
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=dimension,
            device=device,
        )
        super().__init__(config)
        self.speed = config.parameters["speed"]

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the KdV equation residual.

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

        if self.dimension == 1:
            # Compute first and third derivatives
            u_x = torch.autograd.grad(
                u, x, grad_outputs=torch.ones_like(u), create_graph=True
            )[0]
            u_xx = torch.autograd.grad(
                u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
            )[0]
            u_xxx = torch.autograd.grad(
                u_xx, x, grad_outputs=torch.ones_like(u_xx), create_graph=True
            )[0]

            # Compute residual (∂u/∂t + 6u∂u/∂x + ∂³u/∂x³)
            return u_t + 6 * u * u_x + u_xxx
        else:
            # For higher dimensions, compute derivatives for each dimension
            residual = u_t
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
                u_xxx = torch.autograd.grad(
                    u_xx,
                    x[:, dim : dim + 1],
                    grad_outputs=torch.ones_like(u_xx),
                    create_graph=True,
                )[0]
                residual += 6 * u * u_x + u_xxx
            return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (soliton).

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
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
                    params.get("speed", 1.0), dtype=torch.float32, device=self.device
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
