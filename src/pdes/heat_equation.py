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

    pde_type = "heat"  # Add explicit pde_type attribute for FDM comparison detection

    def __init__(self, config: PDEConfig, **kwargs):
        """
        Initialize the Heat Equation.

        :param config: PDEConfig instance containing all necessary parameters
        :param kwargs: Additional keyword arguments
        """
        # Initialize base class first to ensure self.config is set
        super().__init__(config)

        # Now we can safely access self.config.parameters
        if not hasattr(self.config, "parameters") or not self.config.parameters:
            raise ValueError(
                "Heat equation requires 'parameters' in config with 'alpha' value"
            )

        if "alpha" not in self.config.parameters:
            raise ValueError("Heat equation requires 'alpha' parameter in config")

    @property
    def alpha(self):
        """Thermal diffusivity coefficient."""
        return self.config.parameters["alpha"]

    def _calculate_decay_rate(self, k: float) -> float:
        """
        Calculate the decay rate based on physical parameters.
        For the heat equation with solution u(x,t) = A * exp(-decay_rate * t) * sin(k * pi * x),
        the decay rate must be α*(k*π)^2 to satisfy the PDE.

        :param k: Frequency parameter
        :return: Decay rate
        """
        return self.alpha * (k * torch.pi) ** 2

    def compute_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the residual of the heat equation.

        For sin_exp_decay solution type:
        u(x,t) = A * exp(-decay_rate * t) * sin(k * pi * x)
        u_t = -decay_rate * A * exp(-decay_rate * t) * sin(k * pi * x)
        u_xx = -(k * pi)^2 * A * exp(-decay_rate * t) * sin(k * pi * x)

        The heat equation is: u_t = α * u_xx
        Therefore: -decay_rate = -α * (k * pi)^2
        This means decay_rate should equal α * (k * pi)^2 for the exact solution

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
            model, x, t, spatial_derivatives=[2], temporal_derivatives=[1]
        )

        # Heat equation residual: u_t - α∇²u = 0
        u_t = derivatives["dt"]
        laplacian = derivatives["laplacian"]

        # For sin_exp_decay solution type, verify that decay_rate matches the physics
        if self.config.exact_solution.get("type") == "sin_exp_decay":
            # Get solution parameters
            k = self.config.exact_solution.get("frequency", 1.0)
            decay_rate = self._calculate_decay_rate(k)

            # The residual should be: u_t - α*u_xx = 0
            # For the exact solution, decay_rate should equal α*(k*π)^2
            # This is enforced by the physics of the heat equation
            residual = u_t - self.alpha * laplacian
        else:
            residual = u_t - self.alpha * laplacian

        return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution for the heat equation.

        The solution is of the form:
        u(x,t) = A * exp(-decay_rate * t) * sin(k * pi * x)

        For 2D:
        u(x,y,t) = A * exp(-decay_rate * t) * sin(kx * pi * x) * sin(ky * pi * y)

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if not self.config.exact_solution:
            return None

        solution_type = self.config.exact_solution.get("type", "sine")
        if solution_type == "sin_exp_decay":
            A = self.config.exact_solution.get("amplitude", 1.0)
            k = self.config.exact_solution.get("frequency", 2.0)
            decay_rate = self._calculate_decay_rate(k)

            if self.dimension == 1:
                # Heat equation solution with exponential decay
                time_factor = torch.exp(-decay_rate * t)
                space_factor = torch.sin(k * torch.pi * x)
                return A * time_factor * space_factor
            else:
                # For higher dimensions, use product of sine waves with decay
                solution = torch.ones_like(x[:, 0:1])
                for dim in range(self.dimension):
                    space_factor = torch.sin(k * torch.pi * x[:, dim : dim + 1])
                    solution *= space_factor
                time_factor = torch.exp(-decay_rate * t)
                return A * time_factor * solution

        elif solution_type == "sine_2d" and self.dimension == 2:
            A = self.config.exact_solution.get("amplitude", 1.0)
            kx = self.config.exact_solution.get("frequency_x", 1.0)
            ky = self.config.exact_solution.get("frequency_y", 1.0)

            # Correct 2D heat equation solution
            decay_factor = (kx * torch.pi) ** 2 + (ky * torch.pi) ** 2
            time_factor = torch.exp(-decay_factor * t)
            space_factor = torch.sin(kx * torch.pi * x[:, 0:1]) * torch.sin(
                ky * torch.pi * x[:, 1:2]
            )
            return A * time_factor * space_factor

        elif solution_type == "sine":
            # Legacy support for old config format
            return self.exact_solution_sine(x, t)
        else:
            raise ValueError(f"Unsupported exact solution type: {solution_type}")

    def exact_solution_sine(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Legacy support for old sine solution format"""
        A = self.config.exact_solution.get("amplitude", 1.0)
        k = self.config.exact_solution.get("frequency", 2.0)

        if self.dimension == 1:
            time_factor = torch.exp(-self.alpha * (k * torch.pi) ** 2 * t)
            space_factor = torch.sin(k * torch.pi * x)
            return A * time_factor * space_factor
        else:
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                time_factor = torch.exp(-self.alpha * (k * torch.pi) ** 2 * t)
                space_factor = torch.sin(k * torch.pi * x[:, dim : dim + 1])
                solution *= A * time_factor * space_factor
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
                return (
                    lambda x, t: A
                    * torch.sin(kx * torch.pi * x[:, 0:1])
                    * torch.sin(ky * torch.pi * x[:, 1:2])
                )
            elif ic_type == "sin_exp_decay":
                A = params.get("amplitude", 1.0)
                k = params.get("frequency", 1.0)
                decay_rate = self._calculate_decay_rate(k)
                if self.dimension == 1:
                    # Include exponential decay term
                    return (
                        lambda x, t: A
                        * torch.sin(k * torch.pi * x)
                        * torch.exp(-decay_rate * t)
                    )
                else:
                    # For higher dimensions, use product of sine waves with exponential decay
                    return (
                        lambda x, t: A
                        * torch.prod(torch.sin(k * torch.pi * x), dim=1, keepdim=True)
                        * torch.exp(-decay_rate * t)
                    )
            else:
                raise ValueError(f"Unsupported initial condition type: {ic_type}")
        elif (
            bc_type == "dirichlet"
            and self.config.exact_solution.get("type") == "sin_exp_decay"
        ):
            # For Dirichlet boundary conditions with sin_exp_decay
            A = self.config.exact_solution.get("amplitude", 1.0)
            k = self.config.exact_solution.get("frequency", 1.0)
            decay_rate = self._calculate_decay_rate(k)
            return (
                lambda x, t: A
                * torch.sin(k * torch.pi * x)
                * torch.exp(-decay_rate * t)
            )
        else:
            return super()._create_boundary_condition(bc_type, params)

    def validate(self, model, num_points=5000):
        """
        Validate the model's solution against exact solution.

        :param model: Neural network model
        :param num_points: Number of validation points (default increased to 5000 for better resolution)
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

    def compute_loss(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for training with periodic boundary conditions.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Dictionary of loss components
        """
        # Compute PDE residual
        residual = self.compute_residual(model, x, t)
        residual_loss = torch.mean(residual**2)

        # Initialize boundary loss
        boundary_loss = torch.tensor(0.0, device=self.device)

        # Get number of points from config or use default
        num_boundary_points = self.config.training.get(
            "num_boundary_points", self.config.training.num_collocation_points // 10
        )

        # Calculate time ranges based on domain
        t_max = self.config.time_domain[1]
        t_early = t_max * 0.01  # Use 1% of total time for early points

        # Generate more points near t=0 for better initial condition handling
        t_boundary = torch.cat(
            [
                torch.linspace(
                    0, t_early, num_boundary_points // 4, device=self.device
                ),
                torch.linspace(
                    t_early, t_max, 3 * num_boundary_points // 4, device=self.device
                ),
            ]
        ).reshape(-1, 1)

        # Left boundary (x=0) points
        x_left = torch.zeros(num_boundary_points, 1, device=self.device)
        points_left = torch.cat([x_left, t_boundary], dim=1)
        points_left.requires_grad_(True)

        # Right boundary (x=1) points
        x_right = torch.ones(num_boundary_points, 1, device=self.device)
        points_right = torch.cat([x_right, t_boundary], dim=1)
        points_right.requires_grad_(True)

        # Compute values at boundaries
        u_left = model(points_left)
        u_right = model(points_right)

        # Compute derivatives at boundaries
        du_dx_left = torch.autograd.grad(
            u_left, points_left, grad_outputs=torch.ones_like(u_left), create_graph=True
        )[0][
            :, 0:1
        ]  # Extract only the x-derivative

        du_dx_right = torch.autograd.grad(
            u_right,
            points_right,
            grad_outputs=torch.ones_like(u_right),
            create_graph=True,
        )[0][
            :, 0:1
        ]  # Extract only the x-derivative

        # Periodic boundary conditions:
        # 1. Values should match: u(0,t) = u(1,t)
        boundary_loss += torch.mean((u_left - u_right) ** 2)

        # 2. Derivatives should match: du/dx(0,t) = du/dx(1,t)
        boundary_loss += torch.mean((du_dx_left - du_dx_right) ** 2)

        # Get number of initial points from config or use default
        num_initial = self.config.training.get(
            "num_initial_points", self.config.training.num_collocation_points // 5
        )

        # Calculate domain ranges
        x_min, x_max = self.config.domain[0]
        x_boundary = (x_max - x_min) * 0.1  # Use 10% of domain for boundary regions

        x_initial = torch.cat(
            [
                torch.linspace(
                    x_min, x_min + x_boundary, num_initial // 4, device=self.device
                ),
                torch.linspace(
                    x_min + x_boundary,
                    x_max - x_boundary,
                    num_initial // 2,
                    device=self.device,
                ),
                torch.linspace(
                    x_max - x_boundary, x_max, num_initial // 4, device=self.device
                ),
            ]
        ).reshape(-1, 1)

        t_initial = torch.zeros_like(x_initial)
        points_initial = torch.cat([x_initial, t_initial], dim=1)
        u_initial = model(points_initial)

        # Get initial condition function
        if "initial" in self.boundary_conditions:
            u_target = self.boundary_conditions["initial"](x_initial, t_initial)
        else:
            # Default to sine wave if no initial condition specified
            k = self.config.initial_condition.get("frequency", 2.0)
            u_target = torch.sin(k * torch.pi * x_initial)

        initial_loss = torch.mean((u_initial - u_target) ** 2)

        # Add smoothness regularization with configurable weight
        smoothness_weight = self.config.training.loss_weights.get("smoothness", 0.01)
        smoothness_loss = torch.mean(torch.abs(du_dx_right)) + torch.mean(
            torch.abs(du_dx_left)
        )

        # Total loss with weighted components from training config
        total_loss = (
            self.config.training.loss_weights.get("pde", 1.0) * residual_loss
            + self.config.training.loss_weights.get("boundary", 10.0) * boundary_loss
            + self.config.training.loss_weights.get("initial", 10.0) * initial_loss
            + smoothness_weight * smoothness_loss
        )

        return {
            "total": total_loss,
            "residual": residual_loss,
            "boundary": boundary_loss,
            "initial": initial_loss,
        }
