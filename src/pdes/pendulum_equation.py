# Pendulum equation
# Application domains: Mechanical systems, classical mechanics, robotics
# Complexity: Nonlinear, 2nd-order

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from .pde_base import PDEBase, PDEConfig


class PendulumEquation(PDEBase):
    """Pendulum Equation implementation.

    The pendulum equation describes the motion of a simple pendulum:
    d²θ/dt² + (g/L)sin(θ) = 0

    where:
    - θ is the angular displacement
    - g is the gravitational acceleration
    - L is the length of the pendulum
    """

    def __init__(self, config: PDEConfig, **kwargs):
        """Initialize the Pendulum Equation.

        Args:
            config: PDEConfig instance containing all necessary parameters
            kwargs: Additional keyword arguments
        """
        super().__init__(config)
        self.g = self.config.parameters.get(
            "g", 9.81
        )  # Default gravitational acceleration
        self.L = self.config.parameters.get("L", 1.0)  # Default pendulum length

    def compute_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the residual of the pendulum equation.

        Args:
            model: Neural network model
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Residual values
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
            model,
            x,
            t,
            temporal_derivatives=[1, 2],  # Need first and second time derivatives
            spatial_derivatives=set(),  # No spatial derivatives needed for pendulum equation
        )

        # Get solution and derivatives
        u = model(torch.cat([x, t], dim=1))
        u_t = derivatives["dt"]
        u_tt = derivatives["dt2"]

        # Compute residual: d²θ/dt² + (g/L)sin(θ)
        residual = u_tt + (self.g / self.L) * torch.sin(u)
        return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute exact analytical solution if available.

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Exact solution values
        """
        if not self.config.exact_solution:
            return None

        solution_type = self.config.exact_solution.get("type", "small_angle")

        if solution_type == "small_angle":
            # Small angle approximation: θ(t) = θ₀cos(ωt)
            theta_0 = self.config.exact_solution.get("initial_angle", 0.1)
            omega = torch.sqrt(self.g / self.L)
            return theta_0 * torch.cos(omega * t)

        elif solution_type == "sine":
            amplitude = self.config.exact_solution.get("amplitude", 1.0)
            frequency = self.config.exact_solution.get("frequency", 1.0)
            return amplitude * torch.sin(frequency * (x + t))

        else:
            raise ValueError(f"Unknown exact solution type: {solution_type}")

    def _create_boundary_condition(
        self, bc_type: str, params: Dict[str, Any]
    ) -> callable:
        """Create boundary condition function from parameters.

        Args:
            bc_type: Type of boundary condition
            params: Parameters for the boundary condition

        Returns:
            Boundary condition function
        """
        if bc_type == "initial":
            ic_type = params.get("type", "small_angle")

            if ic_type == "small_angle":
                theta_0 = params.get("initial_angle", 0.1)
                return lambda x, t: torch.full_like(x, theta_0)

            elif ic_type == "sine":
                amplitude = params.get("amplitude", 1.0)
                frequency = params.get("frequency", 1.0)
                return lambda x, t: amplitude * torch.sin(frequency * x)

            elif ic_type == "gaussian":
                amplitude = params.get("amplitude", 1.0)
                center = params.get("center", 0.0)
                sigma = params.get("sigma", 0.1)
                return lambda x, t: amplitude * torch.exp(
                    -((x - center) ** 2) / (2 * sigma**2)
                )

            else:
                raise ValueError(f"Unknown initial condition type: {ic_type}")
        else:
            return super()._create_boundary_condition(bc_type, params)

    def compute_energy(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the total energy of the system.

        Args:
            model: Neural network model
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Total energy values
        """
        # Get solution and its time derivative
        derivatives = self.compute_derivatives(
            model, x, t, temporal_derivatives=[1], spatial_derivatives=set()
        )

        # Get the solution
        u = model(torch.cat([x, t], dim=1))

        # Kinetic energy: (1/2) * m * L^2 * (du/dt)^2
        kinetic = 0.5 * self.L * self.L * derivatives["dt"].pow(2)

        # Potential energy: m * g * L * (1 - cos(u))
        potential = self.g * self.L * (1 - torch.cos(u))

        # Total energy is the sum of kinetic and potential
        return kinetic + potential

    def compute_phase_space(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the phase space coordinates (θ, dθ/dt).

        Args:
            model: Neural network model
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Tuple of (angular displacement, angular velocity)
        """
        # Get solution and its time derivative
        derivatives = self.compute_derivatives(model, x, t, temporal_derivatives=[1])
        u = model(torch.cat([x, t], dim=1))
        u_t = derivatives["dt"]

        return u, u_t

    def validate(self, model, num_points=1000):
        """Validate the model's solution against exact solution.

        Args:
            model: Neural network model
            num_points: Number of validation points

        Returns:
            Dictionary of error metrics
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

    def compute_initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the initial condition.

        Args:
            x: Spatial coordinates

        Returns:
            Initial condition values
        """
        if not self.config.initial_condition:
            return None

        ic_type = self.config.initial_condition.get("type", "small_angle")

        if ic_type == "small_angle":
            theta_0 = self.config.initial_condition.get("initial_angle", 0.1)
            return torch.full_like(x, theta_0)

        elif ic_type == "sine":
            amplitude = self.config.initial_condition.get("amplitude", 1.0)
            frequency = self.config.initial_condition.get("frequency", 1.0)
            return amplitude * torch.sin(frequency * x)

        elif ic_type == "gaussian":
            amplitude = self.config.initial_condition.get("amplitude", 1.0)
            center = self.config.initial_condition.get("center", 0.0)
            sigma = self.config.initial_condition.get("sigma", 0.1)
            return amplitude * torch.exp(-((x - center) ** 2) / (2 * sigma**2))

        else:
            raise ValueError(f"Unknown initial condition type: {ic_type}")

    def compute_boundary_condition(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the boundary condition.

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Boundary condition values
        """
        if not self.config.boundary_conditions:
            return None

        bc_type = self.config.boundary_conditions.get("dirichlet", {}).get(
            "type", "fixed"
        )

        if bc_type == "fixed":
            value = self.config.boundary_conditions["dirichlet"].get("value", 0.0)
            return torch.full_like(x, value)

        elif bc_type == "periodic":
            return torch.sin(2 * np.pi * x)

        else:
            raise ValueError(f"Unknown boundary condition type: {bc_type}")
