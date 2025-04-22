import numpy as np
import torch
import logging
from typing import Dict, Callable, Any, Optional, Tuple, List
import os
from src.pdes.heat_equation import HeatEquation
from src.pdes.pde_base import PDEConfig
from dataclasses import dataclass


@dataclass
class FDMConfig:
    """Configuration for Finite Difference Methods."""
    nx: int = 1000  # Number of spatial points
    nt: int = 4000  # Number of time steps
    domain: List[List[float]] = None  # Spatial domain
    time_domain: List[float] = None  # Time domain
    parameters: Dict[str, Any] = None  # PDE parameters
    boundary_conditions: Dict[str, Any] = None  # Boundary conditions
    initial_condition: Dict[str, Any] = None  # Initial condition
    exact_solution: Dict[str, Any] = None  # Exact solution
    dimension: int = 1  # Spatial dimension
    device: str = "cpu"  # Device for computations

    def __post_init__(self):
        """Set default values if not provided."""
        if self.domain is None:
            self.domain = [[0.0, 2.0]]
        if self.time_domain is None:
            self.time_domain = [0.0, 10.0]
        if self.parameters is None:
            self.parameters = {}
        if self.boundary_conditions is None:
            self.boundary_conditions = {}
        if self.initial_condition is None:
            self.initial_condition = {}
        if self.exact_solution is None:
            self.exact_solution = {}


class HeatEquationFDM:
    """
    Finite Difference Method solver for the Heat Equation.
    Uses the main HeatEquation class for validation and configuration.
    """
    
    def __init__(self, config: Dict[str, Any], device: str = "cpu"):
        """
        Initialize the FDM solver.
        
        Args:
            config: Configuration dictionary containing PDE parameters
            device: Device to use for computations
        """
        # Create FDMConfig from the input config
        fdm_config = FDMConfig(
            nx=config.get("nx", 100),
            nt=config.get("nt", 100),
            domain=config.get("domain", [[0, 1]]),
            time_domain=config.get("time_domain", [0, 1]),
            parameters=config.get("parameters", {"alpha": 0.01}),
            boundary_conditions=config.get("boundary_conditions", {}),
            initial_condition=config.get("initial_condition", {}),
            exact_solution=config.get("exact_solution", {}),
            dimension=config.get("dimension", 1),
            device=device
        )
        
        # Store configuration
        self.config = fdm_config
        self.nx = fdm_config.nx
        self.nt = fdm_config.nt
        self.dx = (fdm_config.domain[0][1] - fdm_config.domain[0][0]) / (self.nx - 1)
        self.dt = (fdm_config.time_domain[1] - fdm_config.time_domain[0]) / (self.nt - 1)
        
        # Create spatial and temporal grids
        self.x = np.linspace(fdm_config.domain[0][0], fdm_config.domain[0][1], self.nx)
        self.t = np.linspace(fdm_config.time_domain[0], fdm_config.time_domain[1], self.nt)
        
        # Create PDEConfig for HeatEquation
        pde_config = PDEConfig(
            name="Heat Equation",
            domain=config.get("domain", [[0, 1]]),
            time_domain=config.get("time_domain", [0, 1]),
            parameters=config.get("parameters", {"alpha": 0.01}),
            boundary_conditions=config.get("boundary_conditions", {}),
            initial_condition=config.get("initial_condition", {}),
            exact_solution=config.get("exact_solution", {}),
            dimension=config.get("dimension", 1),
            device=device
        )
        
        # Initialize the main HeatEquation class
        self.pde = HeatEquation(config=pde_config)
        
        # Initialize solution array
        self.u = np.zeros((self.nt, self.nx))
        
        # Stability check
        self._check_stability()
        
    def _check_stability(self):
        """Check if the numerical scheme is stable."""
        r = self.pde.alpha * self.dt / (self.dx ** 2)
        if r > 0.5:
            raise ValueError(
                f"Numerical scheme is unstable. Current r = {r:.3f}, must be <= 0.5"
            )
            
    def solve(self) -> np.ndarray:
        """
        Solve the heat equation using the finite difference method.
        
        Returns:
            np.ndarray: Solution array of shape (nt, nx)
        """
        # Convert to torch tensors for boundary condition evaluation
        x_tensor = torch.tensor(self.x, dtype=torch.float32, device=self.pde.device).reshape(-1, 1)
        t_tensor = torch.zeros_like(x_tensor)
        
        # Set initial condition
        self.u[0] = self.pde.boundary_conditions["initial"](x_tensor, t_tensor).cpu().numpy().flatten()
        
        # Set boundary conditions based on type
        if "dirichlet" in self.pde.boundary_conditions:
            # Fixed value boundary conditions
            t_tensor = torch.tensor(self.t, dtype=torch.float32, device=self.pde.device).reshape(-1, 1)
            x_left = torch.zeros(1, dtype=torch.float32, device=self.pde.device).reshape(-1, 1)
            x_right = torch.ones(1, dtype=torch.float32, device=self.pde.device).reshape(-1, 1)
            
            # Evaluate boundary conditions for all time steps
            for i in range(self.nt):
                t_i = t_tensor[i:i+1]
                self.u[i, 0] = self.pde.boundary_conditions["dirichlet"](x_left, t_i).item()
                self.u[i, -1] = self.pde.boundary_conditions["dirichlet"](x_right, t_i).item()
                
        elif "periodic" in self.pde.boundary_conditions:
            # Periodic boundary conditions - values at boundaries are equal
            for i in range(self.nt):
                self.u[i, -1] = self.u[i, 0]
        
        # Time stepping
        r = self.pde.alpha * self.dt / (self.dx ** 2)
        for n in range(0, self.nt - 1):
            for i in range(1, self.nx - 1):
                self.u[n + 1, i] = (
                    self.u[n, i]
                    + r * (self.u[n, i + 1] - 2 * self.u[n, i] + self.u[n, i - 1])
                )
            
            # For periodic boundary conditions, update the boundary points
            if "periodic" in self.pde.boundary_conditions:
                self.u[n + 1, 0] = self.u[n + 1, -2]  # Copy second-to-last point to first
                self.u[n + 1, -1] = self.u[n + 1, 1]  # Copy second point to last
                
        return self.u
    
    def validate_solution(self, n: int) -> bool:
        """
        Validate the solution at time step n using the main HeatEquation class.
        
        Args:
            n: Time step to validate
            
        Returns:
            bool: True if solution is valid, False otherwise
        """
        # Convert FDM solution to torch tensor
        x = torch.linspace(self.pde.domain[0][0], self.pde.domain[0][1], self.nx)
        t = torch.full_like(x, self.pde.time_domain[0] + n * self.dt)
        u_fdm = torch.tensor(self.u[n])
        
        # Create a simple model that returns the FDM solution
        class FDMModel(torch.nn.Module):
            def __init__(self, solution):
                super().__init__()
                self.solution = solution
                
            def forward(self, x):
                return self.solution
                
        fdm_model = FDMModel(u_fdm)
        
        # Use HeatEquation's validate method
        metrics = self.pde.validate(fdm_model, num_points=self.nx)
        
        return metrics["validation_passed"]
        
    def get_error(self) -> Tuple[float, float]:
        """
        Compute error metrics between FDM and exact solution.
        
        Returns:
            Tuple[float, float]: (L2 error, max error)
        """
        x = torch.linspace(self.pde.domain[0][0], self.pde.domain[0][1], self.nx)
        t = torch.linspace(self.pde.time_domain[0], self.pde.time_domain[1], self.nt)
        
        # Get exact solution
        u_exact = self.pde.exact_solution(x, t)
        
        # Compute errors
        l2_error = np.sqrt(np.mean((self.u - u_exact.numpy()) ** 2))
        max_error = np.max(np.abs(self.u - u_exact.numpy()))
        
        return l2_error, max_error
    
    def plot_solution(self, save_path: str = None):
        """Plot the FDM solution."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.pcolormesh(self.x, self.t, self.u, shading='auto')
        plt.colorbar(label='u(x,t)')
        plt.xlabel('x')
        plt.ylabel('t')
        plt.title('Heat Equation - FDM Solution')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_comparison_with_pinn(self, model, save_path: str = None, device: str = "cpu"):
        """Plot comparison between FDM, PINN, and exact solutions at different time steps."""
        import matplotlib.pyplot as plt
        
        # Select time steps to plot (0%, 33%, 67%, 100% of total time)
        t_indices = [0, self.nt // 3, 2 * self.nt // 3, self.nt - 1]
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Heat Equation (α={self.pde.alpha:.2f}): FDM vs PINN vs Exact Comparison')
        axes = axes.flatten()
        
        # Evaluate PINN and exact solution at selected time steps
        for idx, t_idx in enumerate(t_indices):
            t_val = self.t[t_idx]
            
            # Create input points for PINN and exact solution
            x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(x_tensor, t_val)
            points = torch.cat([x_tensor, t_tensor], dim=1)
            
            # Evaluate PINN
            with torch.no_grad():
                u_pinn = model(points).cpu().numpy()
            
            # Get exact solution
            u_exact = self.pde.exact_solution(x_tensor, t_tensor).cpu().numpy()
            
            # Plot all solutions
            ax = axes[idx]
            ax.plot(self.x, self.u[t_idx], 'b-', label='FDM', linewidth=2)
            ax.plot(self.x, u_pinn, 'r--', label='PINN', linewidth=2)
            ax.plot(self.x, u_exact, 'g:', label='Exact', linewidth=2)
            ax.set_xlabel('x')
            ax.set_ylabel('u(x,t)')
            ax.set_title(f't = {t_val:.2f}')
            ax.grid(True)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
        # Calculate and return error metrics against both FDM and exact solution
        pinn_solution = self._evaluate_pinn_full(model, device)
        exact_solution = np.zeros_like(pinn_solution)
        
        # Calculate exact solution for all points
        for i, t_val in enumerate(self.t):
            x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(x_tensor, t_val)
            exact_solution[i] = self.pde.exact_solution(x_tensor, t_tensor).cpu().numpy()
        
        return {
            'fdm_pinn_l2_error': np.mean((self.u - pinn_solution)**2),
            'fdm_pinn_max_error': np.max(np.abs(self.u - pinn_solution)),
            'fdm_pinn_mean_error': np.mean(np.abs(self.u - pinn_solution)),
            # 'exact_pinn_l2_error': np.mean((exact_solution - pinn_solution)**2),
            # 'exact_pinn_max_error': np.max(np.abs(exact_solution - pinn_solution)),
            # 'exact_pinn_mean_error': np.mean(np.abs(exact_solution - pinn_solution)),
            'exact_fdm_l2_error': np.mean((exact_solution - self.u)**2),
            'exact_fdm_max_error': np.max(np.abs(exact_solution - self.u)),
            'exact_fdm_mean_error': np.mean(np.abs(exact_solution - self.u))
        }
        
    def _evaluate_pinn_full(self, model, device):
        """Evaluate PINN on the full space-time grid."""
        x_grid, t_grid = np.meshgrid(self.x, self.t)
        points = np.stack([x_grid.flatten(), t_grid.flatten()], axis=1)
        points_tensor = torch.tensor(points, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            return model(points_tensor).cpu().numpy().reshape(self.nt, self.nx)

    @staticmethod
    def generate_fdm_comparison_plots(pde, model, device, viz_dir, logger=None):
        """Generate comparison plots between FDM and PINN solutions.
        
        Args:
            pde: The PDE object
            model: The trained neural network model
            device: The device to use
            viz_dir: Directory to save visualizations
            logger: Logger for logging messages
            
        Returns:
            Dictionary of error metrics
        """
        if logger is None:
            logger = logging.getLogger("HeatEquationFDM")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)
        
        try:
            # Create FDM solver with higher resolution
            config = {
                "domain": pde.domain,
                "time_domain": pde.time_domain,
                "parameters": {"alpha": pde.alpha},
                "boundary_conditions": pde.config.boundary_conditions,
                "initial_condition": pde.config.initial_condition,
                "exact_solution": pde.config.exact_solution,
                "dimension": pde.dimension,
                "nx": 1000,
                "nt": 4000
            }
            
            solver = HeatEquationFDM(config, device)
            
            # Solve
            solver.solve()
            
            # Create comparison plots
            os.makedirs(viz_dir, exist_ok=True)
            
            # Plot FDM solution
            solver.plot_solution(os.path.join(viz_dir, 'fdm_solution.png'))
            
            # Plot comparison with PINN and exact solution
            metrics = solver.plot_comparison_with_pinn(model, 
                                          os.path.join(viz_dir, 'fdm_vs_pinn_comparison.png'),
                                          device)
            
            logger.info(f"Comparison plots saved to {viz_dir}")
            logger.info(f"Error metrics between solutions:")
            logger.info(f"FDM vs PINN - L2={metrics['fdm_pinn_l2_error']:.6f}, Max={metrics['fdm_pinn_max_error']:.6f}, Mean={metrics['fdm_pinn_mean_error']:.6f}")
            logger.info(f"PINN vs Exact - L2={metrics['exact_pinn_l2_error']:.6f}, Max={metrics['exact_pinn_max_error']:.6f}, Mean={metrics['exact_pinn_mean_error']:.6f}")
            logger.info(f"FDM vs Exact - L2={metrics['exact_fdm_l2_error']:.6f}, Max={metrics['exact_fdm_max_error']:.6f}, Mean={metrics['exact_fdm_mean_error']:.6f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error generating FDM comparison: {str(e)}")
            return {}
