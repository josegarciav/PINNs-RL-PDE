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

    def __init__(
        self, 
        config: PDEConfig,
        rl_agent=None
    ):
        """
        Initialize PDE with given configuration.

        :param config: Configuration dataclass
        :param rl_agent: Optional RL agent for adaptive sampling
        """
        self.config = config
        self.domain = config.domain
        self.rl_agent = rl_agent  # Store RL agent
        
        # Make domain always a list for uniform handling
        if not isinstance(self.domain[0], (list, tuple)):
            self.domain = [self.domain]  # Convert to list
            
        # Setup device
        self.device = config.device or torch.device("cpu")
        
        # Store dimensionality
        self.dimension = config.dimension
            
        # Setup boundary and initial conditions
        self._setup_boundary_conditions()
        
        # Setup validation points
        self._setup_validation_points()
        
        # For tracking point distribution
        self.collocation_history = []

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
                    self.domain[0][0], self.domain[0][1], int(np.sqrt(num_points))
                ).reshape(-1, 1)
                t = torch.linspace(
                    self.config.time_domain[0], self.config.time_domain[1], int(np.sqrt(num_points))
                ).reshape(-1, 1)
                
                # Create meshgrid for even coverage
                X, T = torch.meshgrid(x.squeeze(), t.squeeze(), indexing='ij')
                x = X.reshape(-1, 1)
                t = T.reshape(-1, 1)
                
                # Add some noise for better training
                x_noise = (self.domain[0][1] - self.domain[0][0]) * 0.01
                t_noise = (self.config.time_domain[1] - self.config.time_domain[0]) * 0.01
                x = x + torch.randn_like(x) * x_noise
                t = t + torch.randn_like(t) * t_noise
                
                # Clip to domain
                x = torch.clamp(x, self.domain[0][0], self.domain[0][1])
                t = torch.clamp(t, self.config.time_domain[0], self.config.time_domain[1])
                
            else:
                # For multi-dimensional domains
                grid_points = []
                for dim in range(self.dimension):
                    grid_dim = torch.linspace(
                        self.domain[dim][0], self.domain[dim][1], 
                        max(2, int(num_points**(1/(self.dimension+1))))
                    )
                    grid_points.append(grid_dim)
                
                # Add time dimension
                grid_points.append(torch.linspace(
                    self.config.time_domain[0], self.config.time_domain[1],
                    max(2, int(num_points**(1/(self.dimension+1))))
                ))
                
                # Create meshgrid
                grid_tensors = torch.meshgrid(*grid_points, indexing='ij')
                
                # Reshape to points
                points = torch.stack([g.reshape(-1) for g in grid_tensors], dim=1)
                
                # Add noise for better training
                noise_scale = 0.01
                noise = torch.randn_like(points) * noise_scale
                points = points + noise
                
                # Clip to domain
                for dim in range(self.dimension):
                    points[:, dim] = torch.clamp(
                        points[:, dim], self.domain[dim][0], self.domain[dim][1]
                    )
                points[:, -1] = torch.clamp(
                    points[:, -1], self.config.time_domain[0], self.config.time_domain[1]
                )
                
                # Extract x and t
                x = points[:, :self.dimension]
                t = points[:, -1].reshape(-1, 1)

        elif strategy == "latin_hypercube":
            sampler = qmc.LatinHypercube(d=self.dimension + 1)  # +1 for time dimension
            sample = sampler.random(n=num_points)

            # Scale spatial coordinates
            x = []
            for dim in range(self.dimension):
                x.append(
                    torch.tensor(
                        qmc.scale(
                            sample[:, dim].reshape(-1, 1),
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
                    sample[:, -1].reshape(-1, 1),
                    l_bounds=[float(self.config.time_domain[0])],
                    u_bounds=[float(self.config.time_domain[1])],
                ),
                dtype=torch.float32,
            )

        elif strategy == "adaptive":
            # Use RL agent for adaptive sampling if available
            if self.rl_agent is not None:
                # Use the adaptive sampling from RL agent
                grid_size = min(100, max(10, int(np.sqrt(num_points))))
                
                # Create a grid of points for the RL agent to sample from
                if self.dimension == 1:
                    x_grid = torch.linspace(
                        self.domain[0][0], self.domain[0][1], grid_size, device=self.device
                    )
                    t_grid = torch.linspace(
                        self.config.time_domain[0], self.config.time_domain[1], grid_size, device=self.device
                    )
                    X, T = torch.meshgrid(x_grid, t_grid, indexing='ij')
                    points = torch.stack([X.flatten(), T.flatten()], dim=1)
                else:
                    # Generate multi-dimensional grid
                    grid_points = []
                    for dim in range(self.dimension):
                        grid_points.append(torch.linspace(
                            self.domain[dim][0], self.domain[dim][1], grid_size, device=self.device
                        ))
                    grid_points.append(torch.linspace(
                        self.config.time_domain[0], self.config.time_domain[1], grid_size, device=self.device
                    ))
                    
                    grid_tensors = torch.meshgrid(*grid_points, indexing='ij')
                    points = torch.stack([g.flatten() for g in grid_tensors], dim=1)
                
                # Get sampling probabilities from RL agent
                with torch.no_grad():
                    probs = self.rl_agent.select_action(points)
                    probs = torch.abs(probs) # Ensure positive probabilities
                    probs = probs / torch.sum(probs) # Normalize
                
                # Sample points based on probabilities
                selected_indices = torch.multinomial(
                    probs.flatten(), min(num_points, len(points)), replacement=True
                )
                selected_points = points[selected_indices]
                
                # Add some small noise to avoid exact grid points
                noise_scale = min(
                    0.01,
                    min([(self.domain[d][1] - self.domain[d][0])/grid_size for d in range(self.dimension)]),
                    (self.config.time_domain[1] - self.config.time_domain[0])/grid_size
                )
                noise = torch.randn_like(selected_points) * noise_scale
                selected_points = selected_points + noise
                
                # Clip to domain bounds
                for dim in range(self.dimension):
                    selected_points[:, dim] = torch.clamp(
                        selected_points[:, dim], self.domain[dim][0], self.domain[dim][1]
                    )
                selected_points[:, -1] = torch.clamp(
                    selected_points[:, -1], self.config.time_domain[0], self.config.time_domain[1]
                )
                
                # Split into spatial and temporal coordinates
                if self.dimension == 1:
                    x = selected_points[:, 0].reshape(-1, 1)
                else:
                    x = selected_points[:, :self.dimension]
                t = selected_points[:, -1].reshape(-1, 1)
                
                # Store points for visualization
                self.collocation_history.append(selected_points.cpu().numpy())
                
                # Reward the RL agent based on the diversity of selected points
                if len(self.collocation_history) > 1:
                    prev_points = torch.tensor(self.collocation_history[-2], device=self.device)
                    reward = torch.mean(torch.min(torch.cdist(selected_points, prev_points), dim=1)[0])
                    self.rl_agent.update_epsilon(len(self.collocation_history)) # Update exploration rate
            else:
                # Fallback to uniform sampling if no RL agent
                return self.generate_collocation_points(num_points, strategy="uniform")
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
            "collocation_history": self.collocation_history,
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
        self.collocation_history = state["collocation_history"]
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

    def visualize_collocation_evolution(
        self,
        save_path: Optional[str] = None,
        num_snapshots: int = 5
    ):
        """
        Visualize the evolution of collocation points during training.
        
        :param save_path: Path to save the visualization
        :param num_snapshots: Number of snapshots to visualize
        """
        if not self.collocation_history or len(self.collocation_history) < 2:
            print("Not enough collocation history to visualize evolution")
            return
            
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs("visualizations", exist_ok=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        plt.suptitle("Evolution of Collocation Points Network", fontsize=16)
        
        # Define custom colormap for evolution
        colors = ["#0d3b66", "#1b9aaa", "#ef476f", "#ffc43d"]
        cmap = LinearSegmentedColormap.from_list("evolution_cmap", colors, N=len(self.collocation_history))
        
        # Select snapshots to visualize
        indices = np.linspace(0, len(self.collocation_history)-1, num_snapshots).astype(int)
        
        # First plot: Progression of points over time
        ax = axes[0, 0]
        for i, idx in enumerate(indices):
            points = self.collocation_history[idx]
            if self.dimension == 1:
                x_pts = points[:, 0]
                y_pts = points[:, 1]  # time dimension
            else:
                # For 2D problems, use the first two spatial dimensions
                x_pts = points[:, 0]
                y_pts = points[:, 1]
                
            alpha = 0.3 + 0.7 * (i / (len(indices)-1))  # Increasing alpha
            size = 5 + 20 * (i / (len(indices)-1))      # Increasing size
            ax.scatter(x_pts, y_pts, alpha=alpha, s=size, 
                      color=cmap(i/(len(indices)-1)), 
                      label=f"Snapshot {idx}")
                      
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)
        ax.set_title("Progression of Points", fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Second plot: Density evolution (first snapshot)
        points = self.collocation_history[0]
        self._plot_density_snapshot(
            axes[0, 1], points, "Initial Distribution", cmap="Blues")
            
        # Third plot: Density evolution (middle snapshot)
        mid_idx = len(self.collocation_history) // 2
        points = self.collocation_history[mid_idx]
        self._plot_density_snapshot(
            axes[1, 0], points, f"Intermediate Distribution (Snapshot {mid_idx})", cmap="Greens")
            
        # Fourth plot: Density evolution (last snapshot)
        points = self.collocation_history[-1]
        self._plot_density_snapshot(
            axes[1, 1], points, "Final Distribution", cmap="Reds")
            
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
        else:
            plt.savefig("visualizations/collocation_evolution.png", dpi=300)
        plt.close()
    
    def _plot_density_snapshot(self, ax, points, title, cmap="viridis"):
        """Helper method to plot density snapshot."""
        from scipy.stats import gaussian_kde
        
        if self.dimension == 1:
            x_pts = points[:, 0]
            y_pts = points[:, 1]  # time dimension
        else:
            # For 2D problems, use the first two spatial dimensions
            x_pts = points[:, 0]
            y_pts = points[:, 1]
            
        # Define grid for density estimation
        xmin, xmax = np.min(x_pts), np.max(x_pts)
        ymin, ymax = np.min(y_pts), np.max(y_pts)
        
        # Add small padding
        x_padding = (xmax - xmin) * 0.05
        y_padding = (ymax - ymin) * 0.05 if ymax > ymin else 0.05
        
        xmin -= x_padding
        xmax += x_padding
        ymin -= y_padding
        ymax += y_padding
        
        # Create meshgrid
        x_grid, y_grid = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([x_grid.ravel(), y_grid.ravel()])
        
        try:
            # Kernel density estimation
            values = np.vstack([x_pts, y_pts])
            kernel = gaussian_kde(values)
            density = np.reshape(kernel(positions), x_grid.shape)
            
            # Plot density heatmap
            im = ax.imshow(
                density.T, origin='lower', extent=[xmin, xmax, ymin, ymax], 
                aspect='auto', cmap=cmap, alpha=0.8
            )
            
            # Add contour lines
            contour = ax.contour(
                x_grid, y_grid, density, colors='white', alpha=0.3, levels=5
            )
            
            plt.colorbar(im, ax=ax, label="Density")
        except Exception as e:
            # Fallback to histogram
            h = ax.hist2d(x_pts, y_pts, bins=50, cmap=cmap)
            plt.colorbar(h[3], ax=ax, label="Count")
            
        # Add points overlay
        ax.scatter(x_pts, y_pts, s=3, c='white', alpha=0.2)
        
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(alpha=0.3)
