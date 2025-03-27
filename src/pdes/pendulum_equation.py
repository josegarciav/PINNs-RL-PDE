# Pendulum equation
# Application domains: Mechanical systems, classical mechanics, robotics
# Complexity: Nonlinear, 2nd-order

import torch
import numpy as np
from typing import Dict, Tuple, List, Optional
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

    def __init__(self, config: PDEConfig):
        """Initialize the Pendulum Equation.

        Args:
            config: PDEConfig object containing all necessary parameters
        """
        super().__init__(config)
        self.g = config.parameters["g"]
        self.L = config.parameters["L"]

    def compute_residual(
        self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Compute the residual of the pendulum equation.

        Args:
            x: Spatial coordinates
            t: Time coordinates
            u: Solution values

        Returns:
            Residual values
        """
        # Compute derivatives
        du_dt = self.compute_time_derivative(u, t, order=1)
        d2u_dt2 = self.compute_time_derivative(u, t, order=2)

        # Compute residual: d²θ/dt² + (g/L)sin(θ)
        return d2u_dt2 + (self.g / self.L) * torch.sin(u)

    def compute_initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the initial condition.

        Args:
            x: Spatial coordinates

        Returns:
            Initial condition values
        """
        if self.config.initial_condition["type"] == "sine":
            return self.config.initial_condition["amplitude"] * torch.sin(
                self.config.initial_condition["frequency"] * x
            )
        elif self.config.initial_condition["type"] == "gaussian":
            return self.config.initial_condition["amplitude"] * torch.exp(
                -((x - self.config.initial_condition["center"]) ** 2)
                / (2 * self.config.initial_condition["sigma"] ** 2)
            )
        elif self.config.initial_condition["type"] == "constant":
            return torch.full_like(x, self.config.initial_condition["value"])
        else:
            raise ValueError(
                f"Unknown initial condition type: {self.config.initial_condition['type']}"
            )

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
        if self.config.boundary_conditions["dirichlet"]["type"] == "fixed":
            return torch.full_like(
                x, self.config.boundary_conditions["dirichlet"]["value"]
            )
        elif self.config.boundary_conditions["dirichlet"]["type"] == "periodic":
            return torch.sin(2 * np.pi * x)
        else:
            raise ValueError(
                f"Unknown boundary condition type: {self.config.boundary_conditions['dirichlet']['type']}"
            )

    def compute_exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the exact solution if available.

        Args:
            x: Spatial coordinates
            t: Time coordinates

        Returns:
            Exact solution values
        """
        if self.config.exact_solution["type"] == "sine":
            return self.config.exact_solution["amplitude"] * torch.sin(
                self.config.exact_solution["frequency"] * (x + t)
            )
        elif self.config.exact_solution["type"] == "gaussian":
            return self.config.exact_solution["amplitude"] * torch.exp(
                -((x - self.config.exact_solution["center"]) ** 2)
                / (2 * self.config.exact_solution["sigma"] ** 2)
            )
        else:
            raise ValueError(
                f"Unknown exact solution type: {self.config.exact_solution['type']}"
            )

    def compute_time_derivative(
        self, u: torch.Tensor, t: torch.Tensor, order: int = 1
    ) -> torch.Tensor:
        """Compute time derivatives of the solution.

        Args:
            u: Solution values
            t: Time coordinates
            order: Order of the derivative

        Returns:
            Derivative values
        """
        if order == 1:
            return torch.autograd.grad(
                u,
                t,
                grad_outputs=torch.ones_like(u),
                create_graph=True,
                retain_graph=True,
            )[0]
        elif order == 2:
            du_dt = self.compute_time_derivative(u, t, order=1)
            return torch.autograd.grad(
                du_dt,
                t,
                grad_outputs=torch.ones_like(du_dt),
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            raise ValueError(f"Unsupported derivative order: {order}")

    def compute_energy(
        self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Compute the total energy of the system.

        Args:
            x: Spatial coordinates
            t: Time coordinates
            u: Solution values

        Returns:
            Total energy values
        """
        # Compute kinetic energy: 1/2 * m * (L * dθ/dt)²
        du_dt = self.compute_time_derivative(u, t, order=1)
        kinetic_energy = 0.5 * (self.L * du_dt) ** 2

        # Compute potential energy: m * g * L * (1 - cos(θ))
        potential_energy = self.g * self.L * (1 - torch.cos(u))

        return kinetic_energy + potential_energy

    def compute_phase_space(
        self, x: torch.Tensor, t: torch.Tensor, u: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the phase space coordinates (θ, dθ/dt).

        Args:
            x: Spatial coordinates
            t: Time coordinates
            u: Solution values

        Returns:
            Tuple of (angular displacement, angular velocity)
        """
        du_dt = self.compute_time_derivative(u, t, order=1)
        return u, du_dt
