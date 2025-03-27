# Base class for PDE implementations
# Provides common functionality for all PDEs

import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from scipy.stats import qmc
from src.rl_agent import RLAgent


@dataclass
class PDEConfig:
    """Configuration dataclass for PDE parameters."""

    name: str
    domain: Union[
        Tuple[float, float], List[Tuple[float, float]]
    ]  # For 1D, list of tuples for higher dimensions
    time_domain: Tuple[float, float]
    parameters: Dict[str, float]
    boundary_conditions: Dict[str, Dict[str, Any]]
    initial_condition: Dict[str, Any]
    exact_solution: Dict[str, Any]
    dimension: int = 1  # Default to 1D
    device: Optional[torch.device] = None


class PDEBase:
    """
    Base class for defining PDEs for Physics-Informed Neural Networks (PINNs).
    Provides common functionality for all PDE implementations.
    """

    def __init__(self, config: PDEConfig):
        """
        Initialize the PDE base class.

        :param config: PDEConfig object containing all necessary parameters
        """
        self.config = config
        self.device = config.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dimension = config.dimension

        # Handle domain based on dimension
        if self.dimension == 1:
            self.domain = config.domain
        else:
            self.domain = config.domain  # List of tuples for each dimension

        self._setup_boundary_conditions()
        self._setup_validation_points()
        self.metrics_history = []

        # Initialize RL agent for adaptive sampling
        self.rl_agent = RLAgent(
            state_dim=self.dimension + 1,  # spatial dimensions + time
            action_dim=1,  # sampling probability
            hidden_dim=64,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            device=self.device,
        )

    def _setup_boundary_conditions(self):
        """Set up boundary condition functions from configuration."""
        self.boundary_conditions = {}
        for bc_type, params in self.config.boundary_conditions.items():
            self.boundary_conditions[bc_type] = self._create_boundary_condition(
                bc_type, params
            )

    def _setup_validation_points(self):
        """Set up validation points for exact solution comparison."""
        self.validation_points = None

    def _create_boundary_condition(
        self, bc_type: str, params: Dict[str, Any]
    ) -> callable:
        """
        Create boundary condition function from parameters.

        :param bc_type: Type of boundary condition
        :param params: Parameters for the boundary condition
        :return: Boundary condition function
        """
        # Map left/right to dirichlet for backward compatibility
        if bc_type in ["left", "right"]:
            bc_type = "dirichlet"

        if bc_type == "dirichlet":
            value = params.get("value", 0.0)
            return lambda x, t: torch.full_like(
                x[:, 0:1], value
            )  # Keep shape consistent
        elif bc_type == "neumann":
            value = params.get("value", 0.0)
            return lambda x, t: torch.full_like(x[:, 0:1], value)
        elif bc_type == "periodic":
            if self.dimension == 1:
                return lambda x, t: torch.sin(2 * torch.pi * x[:, 0:1])
            else:
                return lambda x, t: torch.sin(
                    2 * torch.pi * torch.sum(x, dim=1, keepdim=True)
                )
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the PDE residual. Must be implemented by subclasses.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Residual tensor
        """
        raise NotImplementedError("Subclasses must implement compute_residual")

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution. Must be implemented by subclasses.

        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        raise NotImplementedError("Subclasses must implement exact_solution")

    def generate_collocation_points(
        self, num_points: int, strategy: str = "uniform"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collocation points for training.

        :param num_points: Number of points to generate
        :param strategy: Sampling strategy ('uniform', 'latin_hypercube', 'adaptive')
        :return: Tuple of spatial and temporal points
        """
        if strategy == "uniform":
            if self.dimension == 1:
                # For 1D, domain is a list with one tuple
                x = torch.linspace(
                    self.domain[0][0], self.domain[0][1], num_points
                ).reshape(-1, 1)
            else:
                # Generate points for each dimension
                x = []
                for dim in range(self.dimension):
                    x.append(
                        torch.linspace(
                            self.domain[dim][0], self.domain[dim][1], num_points
                        )
                    )
                x = torch.stack(x, dim=1)

            t = torch.linspace(
                self.config.time_domain[0], self.config.time_domain[1], num_points
            ).reshape(-1, 1)

        elif strategy == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=self.dimension + 1)  # +1 for time dimension
            sample = sampler.random(n=num_points)

            # Scale spatial coordinates
            x = []
            for dim in range(self.dimension):
                x.append(
                    torch.tensor(
                        qmc.scale(
                            sample[:, [dim]],
                            l_bounds=[float(self.domain[dim][0])],
                            u_bounds=[float(self.domain[dim][1])],
                        ),
                        dtype=torch.float32,
                    )
                )
            x = torch.cat(x, dim=1)

            # Scale time coordinate
            t = torch.tensor(
                qmc.scale(
                    sample[:, [-1]],
                    l_bounds=[float(self.config.time_domain[0])],
                    u_bounds=[float(self.config.time_domain[1])],
                ),
                dtype=torch.float32,
            )

        elif strategy == "adaptive":
            # Use RL agent for adaptive sampling
            points = []
            while len(points) < num_points:
                # Generate random point
                if self.dimension == 1:
                    x = (
                        torch.rand(1, 1) * (self.domain[0][1] - self.domain[0][0])
                        + self.domain[0][0]
                    )
                else:
                    x = []
                    for dim in range(self.dimension):
                        x.append(
                            torch.rand(1, 1)
                            * (self.domain[dim][1] - self.domain[dim][0])
                            + self.domain[dim][0]
                        )
                    x = torch.cat(x, dim=1)

                t = (
                    torch.rand(1, 1)
                    * (self.config.time_domain[1] - self.config.time_domain[0])
                    + self.config.time_domain[0]
                )

                # Get RL agent's action (sampling probability)
                state = torch.cat([x, t], dim=1)
                action = self.rl_agent.select_action(state.to(self.device))

                # Accept/reject based on action
                if torch.rand(1).to(self.device) < action:
                    points.append((x, t))

            # Stack accepted points
            x = torch.cat([p[0] for p in points], dim=0)
            t = torch.cat([p[1] for p in points], dim=0)

        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        return x.to(self.device), t.to(self.device)

    def compute_loss(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components for training.

        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Dictionary of loss components
        """
        # Compute PDE residual
        residual = self.compute_residual(model, x, t)
        residual_loss = torch.mean(residual**2)

        # Compute boundary condition loss
        if self.dimension == 1:
            x_boundary = torch.tensor(
                [self.domain[0][0], self.domain[0][1]],
                dtype=torch.float32,
                device=self.device,
            ).reshape(-1, 1)
        else:
            x_boundary = []
            for dim in range(self.dimension):
                x_boundary.extend([self.domain[dim][0], self.domain[dim][1]])
            x_boundary = torch.tensor(
                x_boundary, dtype=torch.float32, device=self.device
            ).reshape(-1, 1)

        t_boundary = torch.linspace(
            self.config.time_domain[0],
            self.config.time_domain[1],
            100,
            device=self.device,
        ).reshape(-1, 1)

        # Create boundary points by combining spatial and temporal coordinates
        x_boundary = x_boundary.repeat_interleave(len(t_boundary), dim=0)
        t_boundary = t_boundary.repeat(len(x_boundary) // len(t_boundary), 1)

        # Compute boundary condition loss
        boundary_loss = torch.tensor(0.0, device=self.device)
        for bc_type, bc_func in self.boundary_conditions.items():
            u_boundary = model(torch.cat([x_boundary, t_boundary], dim=1))
            u_target = bc_func(x_boundary, t_boundary)
            boundary_loss += torch.mean((u_boundary - u_target) ** 2)

        # Compute initial condition loss
        x_initial = torch.linspace(
            self.domain[0][0], self.domain[0][1], 100, device=self.device
        ).reshape(-1, 1)
        t_initial = torch.zeros_like(x_initial, device=self.device)

        u_initial = model(torch.cat([x_initial, t_initial], dim=1))
        u_target = self.config.initial_condition["amplitude"] * torch.sin(
            self.config.initial_condition["frequency"] * torch.pi * x_initial
        )
        initial_loss = torch.mean((u_initial - u_target) ** 2)

        # Total loss
        total_loss = residual_loss + boundary_loss + initial_loss

        return {
            "total": total_loss,
            "residual": residual_loss,
            "boundary": boundary_loss,
            "initial": initial_loss,
        }

    def validate(
        self, model: torch.nn.Module, num_points: int = 1000
    ) -> Dict[str, float]:
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

    def plot_solution(
        self,
        model: torch.nn.Module,
        num_points: int = 1000,
        save_path: Optional[str] = None,
    ):
        """
        Plot the model's solution and exact solution.

        :param model: Neural network model
        :param num_points: Number of points for plotting
        :param save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt

        x, t = self.generate_collocation_points(num_points)
        u_pred = model(torch.cat([x, t], dim=1))
        u_exact = self.exact_solution(x, t)

        plt.figure(figsize=(10, 6))
        plt.scatter(
            x.cpu().numpy(), u_pred.detach().cpu().numpy(), label="Predicted", alpha=0.5
        )
        plt.scatter(x.cpu().numpy(), u_exact.cpu().numpy(), label="Exact", alpha=0.5)
        plt.xlabel("x")
        plt.ylabel("u")
        plt.title(f"{self.config.name} Solution")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        plt.close()

    def save_state(self, path: str):
        """
        Save the PDE state to a file.

        :param path: Path to save the state
        """
        state = {
            "config": self.config,
            "metrics_history": self.metrics_history,
            "validation_points": self.validation_points,
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """
        Load the PDE state from a file.

        :param path: Path to load the state from
        """
        state = torch.load(path)
        self.config = state["config"]
        self.metrics_history = state["metrics_history"]
        self.validation_points = state["validation_points"]
        self._setup_boundary_conditions()

    def update_sampling_strategy(
        self, x: torch.Tensor, t: torch.Tensor, residual: torch.Tensor
    ):
        """
        Update the RL agent's sampling strategy based on the residual.

        :param x: Spatial coordinates
        :param t: Time coordinates
        :param residual: PDE residual at these points
        """
        # Compute reward based on residual magnitude
        reward = torch.mean(torch.abs(residual))

        # Update RL agent
        state = torch.cat([x, t], dim=1)
        self.rl_agent.update(state, reward)

    def visualize_sampling_strategy(self, num_points: int = 1000):
        """
        Visualize the current sampling strategy.

        :param num_points: Number of points to generate for visualization
        :return: Tuple of (x, t) coordinates and their acceptance probabilities
        """
        x = torch.rand((num_points, self.dimension), device=self.device)
        t = torch.rand((num_points, 1), device=self.device)

        # Scale to domain
        x = self._scale_to_domain(x, self.domain)
        t = self._scale_to_domain(t, self.time_domain)

        # Get RL agent's actions for all points
        states = torch.cat([x, t], dim=1)
        actions = torch.zeros(num_points, device=self.device)
        for i in range(num_points):
            actions[i] = self.rl_agent.select_action(states[i : i + 1])

        return x, t, actions
