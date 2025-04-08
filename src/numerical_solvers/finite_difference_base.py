import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Callable, List, Union
from dataclasses import dataclass
import logging
from abc import ABC, abstractmethod

@dataclass
class FDMConfig:
    """Configuration for Finite Difference Method."""
    nx: int  # Number of spatial points
    nt: int  # Number of time points
    x_domain: Tuple[float, float]  # Spatial domain (x_min, x_max)
    t_domain: Tuple[float, float]  # Time domain (t_min, t_max)
    parameters: Dict[str, float]  # PDE parameters (e.g., alpha for heat equation)
    boundary_conditions: Dict[str, Any]  # Boundary conditions
    initial_condition: Dict[str, Any] = None  # Initial condition
    device: str = 'cpu'  # Device to use for computations
    domain: List[Tuple[float, float]] = None  # Alternative spatial domain format

class FiniteDifferenceSolver(ABC):
    """
    Base class for finite difference solvers.
    Implements common functionality for FDM solvers.
    """
    def __init__(self, config: FDMConfig):
        """
        Initialize the finite difference solver.
        
        Args:
            config: Configuration object containing solver parameters
        """
        self.config = config
        self.dx = config.dx[0] if hasattr(config, 'dx') else (config.x_domain[1] - config.x_domain[0]) / (config.nx - 1)
        self.dt = config.dt if hasattr(config, 'dt') else (config.t_domain[1] - config.t_domain[0]) / (config.nt - 1)
        
        # Initialize grid points
        self.x = np.linspace(config.x_domain[0], config.x_domain[1], config.nx)
        self.t = np.linspace(config.t_domain[0], config.t_domain[1], config.nt)
        self.X, self.T = np.meshgrid(self.x, self.t)
        
        # Initialize solution array
        if self.dimension == 1:
            self.u = np.zeros((config.nt, config.nx))
        else:
            raise NotImplementedError("Multi-dimensional problems not implemented yet")
        
        # Check stability (to be implemented by child classes)
        self.check_stability()

        # Configurar logging
        self.logger = logging.getLogger(__name__)

    @property
    def dimension(self):
        """Get the problem dimension."""
        # For now, we only support 1D problems
        return 1

    @abstractmethod
    def check_stability(self) -> None:
        """Check stability conditions for the numerical scheme."""
        pass

    @abstractmethod
    def discretize_equation(self) -> None:
        """Discretize the PDE using finite differences."""
        pass

    @abstractmethod
    def approximate_differential_terms(self, u: np.ndarray, n: int) -> Dict[str, np.ndarray]:
        """
        Approximate differential terms using finite differences.
        
        Args:
            u: Solution array at current time step
            n: Current time index
            
        Returns:
            Dictionary containing approximated terms
        """
        pass

    @abstractmethod
    def apply_boundary_conditions(self, n: int) -> None:
        """
        Apply boundary conditions at time step n.
        
        Args:
            n: Current time index
        """
        pass

    def set_initial_condition(self, initial_condition_fn: Callable = None) -> None:
        """
        Set the initial condition.
        
        Args:
            initial_condition_fn: Function that takes x and returns initial values
        """
        if initial_condition_fn is None:
            # Default initial condition: sinusoidal wave
            self.u[0, 1:-1] = np.sin(np.pi * self.x[1:-1])
        else:
            # Apply the provided function
            self.u[0, :] = initial_condition_fn(self.x)
        
        # Apply boundary conditions for t=0
        self.apply_boundary_conditions(0)

    def solve(self) -> np.ndarray:
        """
        Solve the PDE using finite differences.
        
        Returns:
            Solution array
        """
        self.logger.info("Solving equation using finite differences...")
        
        # Apply initial condition if not done already
        if np.all(self.u[0] == 0):
            self.set_initial_condition()
        
        # Discretize and solve the equation
        self.discretize_equation()
        
        self.logger.info("Solution completed successfully")
        
        return self.u

    def get_solution_at_time(self, t: float) -> np.ndarray:
        """
        Get the solution at a specific time.
        
        Args:
            t: Time at which to get the solution
            
        Returns:
            Solution at time t
        """
        if t < self.config.t_domain[0] or t > self.config.t_domain[1]:
            raise ValueError(f"Time {t} is outside the domain {self.config.t_domain}")
        
        # Find the closest time index
        t_idx = int(round((t - self.config.t_domain[0]) / self.dt))
        t_idx = min(max(t_idx, 0), self.config.nt - 1)  # Ensure it's within range
        
        return self.u[t_idx]

    def get_solution(self) -> np.ndarray:
        """
        Get the complete solution.
        
        Returns:
            Matrix with the complete solution
        """
        return self.u
        
    # Visualization methods
    def plot_solution(self, save_path=None, num_time_steps=5):
        """
        Visualize and save the numerical solution at multiple time steps.
        
        Args:
            save_path (str): Path where to save the image. If None, no save is performed.
            num_time_steps (int): Number of time steps to visualize.
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create figure to display multiple time steps
            plt.figure(figsize=(12, 8))
            
            # Select uniformly distributed time steps
            time_steps = np.linspace(0, self.config.nt-1, num_time_steps, dtype=int)
            
            # Create x matrix for horizontal axis
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            
            # Plot each time step
            for i, t_idx in enumerate(time_steps):
                plt.subplot(num_time_steps, 1, i+1)
                plt.plot(x, self.u[t_idx], 'b-', linewidth=2)
                plt.title(f'FDM Solution at t = {t_idx * self.dt:.3f}')
                plt.ylabel('u(x,t)')
                if i == num_time_steps - 1:
                    plt.xlabel('x')
                plt.grid(True)
                plt.ylim(np.min(self.u) - 0.1, np.max(self.u) + 0.1)
            
            plt.tight_layout()
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the figure
                plt.savefig(save_path, format='png', dpi=300)
                plt.close()
                print(f"FDM solution saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError as e:
            print(f"Error importing necessary libraries for plotting: {e}")
        except Exception as e:
            print(f"Error plotting FDM solution: {e}")
            
    def plot_solution_multiline(self, save_path=None, num_time_steps=4):
        """
        Visualize and save the numerical solution with multiple time steps on a single plot.
        
        Args:
            save_path (str): Path where to save the image. If None, no save is performed.
            num_time_steps (int): Number of time steps to visualize.
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Select uniformly distributed time steps
            time_steps = np.linspace(0, self.config.nt-1, num_time_steps, dtype=int)
            
            # Create x matrix for horizontal axis
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            
            # Generate colors for each time step
            colors = plt.cm.viridis(np.linspace(0, 1, num_time_steps))
            
            # Plot each time step
            for i, t_idx in enumerate(time_steps):
                t_value = t_idx * self.dt
                plt.plot(x, self.u[t_idx], color=colors[i], linewidth=2, label=f't = {t_value:.3f}')
            
            plt.title('Heat Equation Solution')
            plt.xlabel('x')
            plt.ylabel('u(x,t)')
            plt.grid(True)
            plt.legend()
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the figure
                plt.savefig(save_path, format='png', dpi=300)
                plt.close()
                print(f"FDM multiline plot saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError as e:
            print(f"Error importing necessary libraries for plotting: {e}")
        except Exception as e:
            print(f"Error plotting FDM multiline solution: {e}")
    
    def plot_analytical_comparison(self, analytical_func, save_path=None, num_time_steps=4):
        """
        Compare numerical and analytical solutions.
        
        Args:
            analytical_func: Function to compute analytical solution
            save_path (str): Path where to save the image. If None, no save is performed.
            num_time_steps (int): Number of time steps to visualize.
        """
        try:
            import matplotlib.pyplot as plt
            import os
            
            plt.figure(figsize=(15, 10))
            
            # Get the grid points
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            
            # Select specific time steps for comparison
            time_indices = np.linspace(0, self.config.nt-1, num_time_steps, dtype=int)
            
            # Plot each time step
            for i, idx in enumerate(time_indices):
                t_value = idx * self.dt
                
                # Plot numerical solution
                plt.subplot(2, 2, i+1)
                plt.plot(x, self.u[idx], 'b-', linewidth=2, label='FDM')
                
                # Plot analytical solution if provided
                if analytical_func is not None:
                    analytical = analytical_func(x, t_value)
                    plt.plot(x, analytical, 'r--', linewidth=2, label='Analytical')
                    
                    # Calculate error
                    error = np.abs(self.u[idx] - analytical).mean()
                    plt.title(f't = {t_value:.3f}, Mean Error = {error:.6f}')
                else:
                    plt.title(f't = {t_value:.3f}')
                
                plt.xlabel('x')
                plt.ylabel('u(x,t)')
                plt.grid(True)
                plt.legend()
            
            plt.tight_layout()
            
            # Save the figure if a path is provided
            if save_path:
                # Ensure the directory exists
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # Save the figure
                plt.savefig(save_path, format='png', dpi=300)
                plt.close()
                print(f"Comparison with analytical solution saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError as e:
            print(f"Error importing necessary libraries for plotting: {e}")
        except Exception as e:
            print(f"Error plotting analytical comparison: {e}")
    
    def plot_solution_3d(self, save_path=None, title=None, figsize=(10, 8)):
        """
        Creates a 3D surface plot of the finite difference solution.
        
        Args:
            save_path: Path to save the plot
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            The figure object
        """
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import numpy as np
            
            # Create mesh grid for x and t
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            t = np.linspace(self.config.t_domain[0], self.config.t_domain[1], self.config.nt)
            X, T = np.meshgrid(x, t)
            
            # Get the solution (transpose to match grid)
            Z = self.u.T
            
            # Create 3D surface plot
            fig = make_subplots(specs=[[{"type": "surface"}]])
            
            fig.add_trace(
                go.Surface(
                    x=X, 
                    y=T, 
                    z=Z,
                    colorscale='Viridis',
                    colorbar=dict(
                        title=dict(text="u(x,t)", font=dict(size=14)),
                        thickness=20,
                    )
                )
            )
            
            plot_title = title or f"{self.__class__.__name__} 3D Solution"
            fig.update_layout(
                title=dict(
                    text=plot_title,
                    x=0.5,
                    xanchor='center',
                    font=dict(size=18)
                ),
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                    aspectratio=dict(x=1, y=1, z=0.8),
                    camera=dict(
                        eye=dict(x=1.5, y=-1.5, z=0.8)
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=60),
                width=figsize[0]*100,
                height=figsize[1]*100,
            )
            
            # Save the interactive plot as HTML if path provided
            if save_path:
                # Ensure output directories exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Save as HTML for interactive visualization
                html_path = save_path.replace('.png', '.html')
                fig.write_html(html_path)
                
                # Save as static image
                fig.write_image(save_path)
                print(f"3D solution plot saved to: {save_path}")
                print(f"Interactive 3D visualization saved to: {html_path}")
                
            return fig
            
        except ImportError as e:
            print(f"Error: {e}. Please install plotly with: pip install plotly")
            return None
        except Exception as e:
            print(f"Error creating 3D plot: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def plot_comparison_with_pinn(self, pinn_model, save_path=None, device='cpu', title=None, figsize=(12, 10)):
        """
        Compare the FDM numerical solution with the PINN solution.
        
        Args:
            pinn_model: PyTorch model trained with PINN approach
            save_path: Path to save the plot
            device: Device for torch calculations ('cpu', 'cuda', 'mps')
            title: Plot title
            figsize: Figure size as (width, height)
            
        Returns:
            The figure object
        """
        try:
            import torch
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.gridspec import GridSpec
            
            # Set model to evaluation mode
            pinn_model.eval()
            
            # Create coordinates
            x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
            t_indices = np.linspace(0, self.config.nt-1, 4, dtype=int)  # Select 4 time points
            t_selected = np.array([self.config.t_domain[0] + idx * self.dt for idx in t_indices])
            
            # Prepare figure
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(3, 2, figure=fig)
            
            # Plot at different time steps
            ax1 = fig.add_subplot(gs[0, :])
            ax2 = fig.add_subplot(gs[1, 0])
            ax3 = fig.add_subplot(gs[1, 1])
            ax4 = fig.add_subplot(gs[2, 0])
            ax5 = fig.add_subplot(gs[2, 1])
            
            axes = [ax2, ax3, ax4, ax5]
            
            # Plot full 2D heatmap comparison (FDM vs PINN)
            self._plot_heatmap_comparison(ax1, pinn_model, device)
            
            # Plot solutions at specific time points
            for i, t_idx in enumerate(t_indices):
                t_val = t_selected[i]
                
                # Get FDM solution at time t
                fdm_solution = self.u[t_idx, :]
                
                # Get PINN solution
                with torch.no_grad():
                    x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
                    t_tensor = torch.tensor(t_val, dtype=torch.float32).to(device)
                    # Create a tensor of t values repeated for each x
                    t_repeated = torch.full_like(x_tensor, t_val).to(device)
                    
                    # Stack inputs for model
                    inputs = torch.cat([x_tensor, t_repeated], dim=1)
                    pinn_solution = pinn_model(inputs).cpu().numpy().flatten()
                
                # Plot on current axis
                ax = axes[i]
                ax.plot(x, fdm_solution, 'b-', label='FDM')
                ax.plot(x, pinn_solution, 'r--', label='PINN')
                ax.set_title(f't = {t_val:.2f}')
                ax.set_xlabel('x')
                ax.set_ylabel('u(x,t)')
                ax.grid(True, alpha=0.3)
                ax.legend()
            
            # Add common title
            plt_title = title or "Comparison: FDM vs PINN Solutions"
            fig.suptitle(plt_title, fontsize=16)
            plt.tight_layout()
            
            # Save plot if path provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"FDM vs PINN comparison saved to: {save_path}")
            
            return fig
            
        except Exception as e:
            print(f"Error comparing FDM with PINN: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _plot_heatmap_comparison(self, ax, pinn_model, device='cpu'):
        """
        Plot 2D heatmap comparison between FDM and PINN solutions.
        
        Args:
            ax: Matplotlib axis
            pinn_model: PyTorch model trained with PINN approach
            device: Device for torch calculations
        """
        import torch
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        
        # Create a grid for the subplot
        gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=ax.get_subplotspec(), width_ratios=[1, 1, 0.05])
        ax1 = plt.subplot(gs[0])  # FDM solution
        ax2 = plt.subplot(gs[1])  # PINN solution
        cax = plt.subplot(gs[2])  # Colorbar
        
        # Get X and T grids for plots
        x = np.linspace(self.config.x_domain[0], self.config.x_domain[1], self.config.nx)
        t = np.linspace(self.config.t_domain[0], self.config.t_domain[1], self.config.nt)
        X, T = np.meshgrid(x, t)
        
        # Get FDM solution
        fdm_sol = self.u
        
        # Get PINN solution on the same grid
        pinn_sol = np.zeros((self.config.nt, self.config.nx))
        with torch.no_grad():
            for i, t_val in enumerate(t):
                x_tensor = torch.tensor(x, dtype=torch.float32).reshape(-1, 1).to(device)
                t_repeated = torch.full_like(x_tensor, t_val).to(device)
                inputs = torch.cat([x_tensor, t_repeated], dim=1)
                pinn_sol[i, :] = pinn_model(inputs).cpu().numpy().flatten()
        
        # Find common color scale
        vmin = min(fdm_sol.min(), pinn_sol.min())
        vmax = max(fdm_sol.max(), pinn_sol.max())
        
        # Plot FDM solution
        im1 = ax1.pcolormesh(X, T, fdm_sol, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax1.set_title('FDM Solution')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        
        # Plot PINN solution
        im2 = ax2.pcolormesh(X, T, pinn_sol, shading='auto', cmap='viridis', vmin=vmin, vmax=vmax)
        ax2.set_title('PINN Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('')  # No y label for the second plot
        
        # Add colorbar
        plt.colorbar(im2, cax=cax, label='u(x,t)')
        
        return ax1, ax2 