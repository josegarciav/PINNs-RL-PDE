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

    def _validate_parameters(self):
        """Validate required parameters for heat equation."""
        super()._validate_parameters()
        # Heat equation requires alpha parameter
        self.get_parameter("alpha", required=True)

    @property
    def alpha(self):
        """Thermal diffusivity coefficient."""
        return self.get_parameter("alpha", required=True)

    def _calculate_decay_rate(self, k: float) -> float:
        """
        Calculate the decay rate based on physical parameters.
        For the heat equation with solution u(x,t) = A * exp(-decay_rate * t) * sin(2πfx/L),
        the decay rate must be α*(2πf/L)^2 to satisfy the PDE.

        :param k: Frequency parameter
        :return: Decay rate
        """
        L = self.config.domain[0][1] - self.config.domain[0][0]  # Domain length
        wave_number = 2 * torch.pi * k / L
        return self.alpha * wave_number ** 2

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
        u(x,t) = A * exp(-decay_rate * t) * sin(2πfx/L)

        For 2D:
        u(x,y,t) = A * exp(-decay_rate * t) * sin(2πfx/Lx) * sin(2πfy/Ly)

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if not hasattr(self.config, 'exact_solution') or not self.config.exact_solution:
            # Default to initial condition if no exact solution is specified
            if hasattr(self.config, 'initial_condition'):
                ic_type = self.config.initial_condition.get("type", "sine")
                A = self.config.initial_condition.get("amplitude", 1.0)
                k = self.config.initial_condition.get("frequency", 2.0)
                L = self.config.domain[0][1] - self.config.domain[0][0]
                wave_number = 2 * torch.pi * k / L
                decay_rate = self._calculate_decay_rate(k)
                return A * torch.exp(-decay_rate * t) * torch.sin(wave_number * x)
            return torch.zeros_like(x)  # Return zeros if no solution or initial condition

        solution_type = self.config.exact_solution.get("type", "sin_exp_decay")
        
        if solution_type == "sin_exp_decay":
            A = self.config.exact_solution.get("amplitude", 1.0)
            k = self.config.exact_solution.get("frequency", 2.0)
            L = self.config.domain[0][1] - self.config.domain[0][0]  # Domain length
            wave_number = 2 * torch.pi * k / L
            decay_rate = self._calculate_decay_rate(k)

            if self.dimension == 1:
                # Heat equation solution with exponential decay
                time_factor = torch.exp(-decay_rate * t)
                space_factor = torch.sin(wave_number * x)
                return A * time_factor * space_factor
            else:
                # For higher dimensions, use product of sine waves with decay
                solution = torch.ones_like(x[:, 0:1])
                for dim in range(self.dimension):
                    L_dim = self.config.domain[dim][1] - self.config.domain[dim][0]
                    wave_number = 2 * torch.pi * k / L_dim
                    space_factor = torch.sin(wave_number * x[:, dim:dim+1])
                    solution *= space_factor
                time_factor = torch.exp(-decay_rate * t)
                return A * time_factor * solution

        elif solution_type == "sine_2d" and self.dimension == 2:
            A = self.config.exact_solution.get("amplitude", 1.0)
            kx = self.config.exact_solution.get("frequency_x", 2.0)
            ky = self.config.exact_solution.get("frequency_y", 2.0)

            # Correct 2D heat equation solution
            decay_factor = (kx * torch.pi) ** 2 + (ky * torch.pi) ** 2
            time_factor = torch.exp(-self.alpha * decay_factor * t)
            space_factor = torch.sin(kx * torch.pi * x[:, 0:1]) * torch.sin(
                ky * torch.pi * x[:, 1:2]
            )
            return A * time_factor * space_factor

        elif solution_type == "sine":
            # Legacy support for old config format
            A = self.config.exact_solution.get("amplitude", 1.0)
            k = self.config.exact_solution.get("frequency", 2.0)
            L = self.config.domain[0][1] - self.config.domain[0][0]
            wave_number = 2 * torch.pi * k / L
            decay_rate = self._calculate_decay_rate(k)
            return A * torch.exp(-decay_rate * t) * torch.sin(wave_number * x)

        # If we reach here, return a default solution based on initial condition
        if hasattr(self.config, 'initial_condition'):
            ic_type = self.config.initial_condition.get("type", "sine")
            A = self.config.initial_condition.get("amplitude", 1.0)
            k = self.config.initial_condition.get("frequency", 2.0)
            L = self.config.domain[0][1] - self.config.domain[0][0]
            wave_number = 2 * torch.pi * k / L
            decay_rate = self._calculate_decay_rate(k)
            return A * torch.exp(-decay_rate * t) * torch.sin(wave_number * x)
            
        return torch.zeros_like(x)  # Final fallback

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
            if ic_type == "sin_exp_decay":
                A = params.get("amplitude", 1.0)
                k = params.get("frequency", 2.0)
                L = self.config.domain[0][1] - self.config.domain[0][0]  # Domain length
                wave_number = 2 * torch.pi * k / L
                decay_rate = self._calculate_decay_rate(k)
                
                if self.dimension == 1:
                    def initial_condition(x, t):
                        return A * torch.sin(wave_number * x) * torch.exp(-decay_rate * t)
                    return initial_condition
                else:
                    def initial_condition(x, t):
                        solution = torch.ones_like(x[:, 0:1])
                        for dim in range(self.dimension):
                            L_dim = self.config.domain[dim][1] - self.config.domain[dim][0]
                            wave_number = 2 * torch.pi * k / L_dim
                            space_factor = torch.sin(wave_number * x[:, dim:dim+1])
                            solution *= space_factor
                        return A * solution * torch.exp(-decay_rate * t)
                    return initial_condition
            elif ic_type == "sine":
                A = params.get("amplitude", 1.0)
                k = params.get("frequency", 2.0)
                L = self.config.domain[0][1] - self.config.domain[0][0]
                wave_number = 2 * torch.pi * k / L
                def initial_condition(x, t):
                    if self.dimension == 1:
                        return A * torch.sin(wave_number * x)
                    else:
                        return A * torch.prod(torch.sin(wave_number * x), dim=1, keepdim=True)
                return initial_condition
            else:
                return super()._create_boundary_condition(bc_type, params)
        elif (
            bc_type == "dirichlet"
            and hasattr(self.config, 'exact_solution')
            and self.config.exact_solution.get("type") == "sin_exp_decay"
        ):
            # For Dirichlet boundary conditions with sin_exp_decay
            A = self.config.exact_solution.get("amplitude", 1.0)
            k = self.config.exact_solution.get("frequency", 2.0)
            L = self.config.domain[0][1] - self.config.domain[0][0]
            wave_number = 2 * torch.pi * k / L
            decay_rate = self._calculate_decay_rate(k)
            def boundary_condition(x, t):
                return A * torch.sin(wave_number * x) * torch.exp(-decay_rate * t)
            return boundary_condition
        else:
            return super()._create_boundary_condition(bc_type, params)

    def validate(self, model, num_points=5000):
        """
        Validate the model's solution against exact solution and physical constraints.

        Args:
            model: Neural network model
            num_points: Number of validation points (default 5000 for better resolution)

        Returns:
            Dictionary containing validation metrics and boolean indicating if all checks passed
        """
        validation_passed = True
        validation_messages = []
        metrics = {}

        # Generate validation points
        x, t = self.generate_collocation_points(num_points)
        input_points = torch.cat([x, t], dim=1)
        
        # Get model predictions
        u_pred = model(input_points)
        
        # Check for NaN/Inf values
        if torch.isnan(u_pred).any() or torch.isinf(u_pred).any():
            validation_passed = False
            validation_messages.append("Error: Solution contains NaN or Inf values")

        # Compute exact solution and errors
        u_exact = self.exact_solution(x, t)
        error = torch.abs(u_pred - u_exact)
        
        metrics.update({
            "l2_error": torch.mean(error**2).item(),
            "max_error": torch.max(error).item(),
            "mean_error": torch.mean(error).item(),
        })

        # Check physical bounds if configured
        if hasattr(self.config, "physical_bounds"):
            min_temp = self.config.physical_bounds.get("min_temperature", float("-inf"))
            max_temp = self.config.physical_bounds.get("max_temperature", float("inf"))
            
            if torch.any(u_pred < min_temp) or torch.any(u_pred > max_temp):
                validation_passed = False
                validation_messages.append(
                    f"Error: Solution violates physical temperature bounds [{min_temp}, {max_temp}]"
                )

        # Validate periodic boundary conditions
        if "periodic" in self.boundary_conditions:
            x_left = torch.zeros(num_points//10, 1, device=self.device)
            x_right = torch.ones(num_points//10, 1, device=self.device)
            t_boundary = torch.linspace(0, self.config.time_domain[1], 
                                      num_points//10, device=self.device).reshape(-1, 1)
            
            points_left = torch.cat([x_left, t_boundary], dim=1)
            points_right = torch.cat([x_right, t_boundary], dim=1)
            
            u_left = model(points_left)
            u_right = model(points_right)
            
            periodic_error = torch.mean((u_left - u_right)**2).item()
            metrics["periodic_bc_error"] = periodic_error
            
            if periodic_error > 1e-3:  # Tolerance threshold
                validation_passed = False
                validation_messages.append(
                    f"Warning: Periodic boundary condition error ({periodic_error:.2e}) exceeds tolerance"
                )

        # Add validation status to metrics
        metrics["validation_passed"] = validation_passed
        metrics["validation_messages"] = validation_messages

        return metrics

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

        # Compute smoothness loss if weight > 0
        smoothness_loss = torch.tensor(0.0, device=self.device)
        smoothness_weight = self.config.training.loss_weights.get("smoothness", 0.0)
        if smoothness_weight > 0:
            smoothness_loss = self._compute_smoothness_loss(model, x, t)

        # Store individual loss components
        losses = {
            "residual": residual_loss,  # PDE residual loss
            "boundary": boundary_loss,  # Boundary condition loss
            "initial": initial_loss,  # Initial condition loss
            "smoothness": smoothness_loss,  # Smoothness regularization
        }

        # Use adaptive weights if enabled
        if (
            hasattr(self.config.training, "adaptive_weights")
            and self.config.training.adaptive_weights.enabled
        ):
            # The total loss will be computed by the trainer using adaptive weights
            # We just return the individual components
            losses["total"] = (
                residual_loss
                + boundary_loss
                + initial_loss
                + smoothness_weight * smoothness_loss
            )
        else:
            # Otherwise use fixed weights from config
            total_loss = (
                self.config.training.loss_weights.get("pde", 1.0) * residual_loss
                + self.config.training.loss_weights.get("boundary", 10.0)
                * boundary_loss
                + self.config.training.loss_weights.get("initial", 10.0) * initial_loss
                + smoothness_weight * smoothness_loss
            )
            losses["total"] = total_loss

        return losses

    def _compute_smoothness_loss(self, model, x, t):
        """
        Compute smoothness regularization loss.

        Args:
            model: Neural network model
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            torch.Tensor: Smoothness loss
        """
        # Sample additional points for smoothness calculation
        batch_size = x.shape[0]
        epsilon = 1e-4

        # Create slightly shifted points to compute gradients
        x_right = x + epsilon
        x_left = x - epsilon

        # Make sure the points stay within the domain bounds
        x_right = torch.clamp(x_right, self.domain[0][0], self.domain[0][1])
        x_left = torch.clamp(x_left, self.domain[0][0], self.domain[0][1])

        # Forward pass for original and shifted points
        input_right = torch.cat([x_right, t], dim=1)
        input_left = torch.cat([x_left, t], dim=1)

        u_right = model(input_right)
        u_left = model(input_left)

        # Calculate approximate derivatives
        du_dx_right = (u_right - model(torch.cat([x, t], dim=1))) / epsilon
        du_dx_left = (model(torch.cat([x, t], dim=1)) - u_left) / epsilon

        # Smoothness loss is the sum of the absolute gradients
        smoothness_loss = torch.mean(torch.abs(du_dx_right)) + torch.mean(
            torch.abs(du_dx_left)
        )

        return smoothness_loss
