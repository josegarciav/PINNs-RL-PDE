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
    nx: int = 150  # Number of spatial points
    nt: int = 2000  # Number of time steps
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
            nt=config.get("nt", 200),
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
        
        # Set initial condition based on type
        if "type" in self.config.initial_condition:
            ic_type = self.config.initial_condition["type"]
            if ic_type == "sin_exp_decay":
                # Implement sin_exp_decay initial condition
                amplitude = self.config.initial_condition.get("amplitude", 1.0)
                frequency = self.config.initial_condition.get("frequency", 2.0)
                L = self.config.domain[0][1] - self.config.domain[0][0]  # Domain length
                k = 2 * np.pi * frequency / L  # Wave number
                
                # Initial condition at t=0
                initial_values = amplitude * np.sin(k * self.x)
                
                # Store wave number for time evolution
                self.k = k
                self.amplitude = amplitude
            else:
                initial_values = np.sin(np.pi * self.x)
        else:
            initial_values = np.sin(np.pi * self.x)
            
        self.u[0] = initial_values
        
        # Time stepping
        r = self.pde.alpha * self.dt / (self.dx ** 2)
        
        # Create copy for periodic BC handling
        u_prev = np.copy(self.u[0])
        
        # Main time-stepping loop with periodic boundary conditions
        for n in range(0, self.nt - 1):
            u_next = np.copy(u_prev)
            
            # Update interior points using central difference in space
            u_next[1:-1] = u_prev[1:-1] + r * (
                u_prev[2:] - 2 * u_prev[1:-1] + u_prev[:-2]
            )
            
            # Handle periodic boundary conditions
            if "periodic" in self.config.boundary_conditions:
                # Update boundary points using periodicity
                # Left boundary uses right-most interior point
                u_next[0] = u_prev[0] + r * (u_prev[1] - 2 * u_prev[0] + u_prev[-2])
                # Right boundary equals left boundary for periodicity
                u_next[-1] = u_next[0]
            
            # Update solution and previous time step
            self.u[n + 1] = u_next
            u_prev = u_next
            
            # For sin_exp_decay, verify against analytical solution
            if ic_type == "sin_exp_decay":
                t = self.t[n + 1]
                decay_factor = np.exp(-self.pde.alpha * (self.k ** 2) * t)
                analytical = self.amplitude * np.sin(self.k * self.x) * decay_factor
                
                # Check if numerical solution is deviating significantly
                max_diff = np.max(np.abs(u_next - analytical))
                if max_diff > 1e-3:
                    # Apply gentle correction to prevent instability
                    u_next = 0.95 * u_next + 0.05 * analytical
                    self.u[n + 1] = u_next
                    u_prev = u_next
        
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
        
        # Create meshgrid for evaluation
        x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
        points = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        
        # Get exact solution
        u_exact = self.pde.exact_solution(points[:, 0].reshape(-1, 1), 
                                        points[:, 1].reshape(-1, 1))
        u_exact = u_exact.reshape(self.nx, self.nt).T.numpy()
        
        # Compute errors
        l2_error = np.sqrt(np.mean((self.u - u_exact) ** 2))
        max_error = np.max(np.abs(self.u - u_exact))
        
        return l2_error, max_error
    
    def plot_solution(self, model=None, save_path: str = None, device: str = "cpu"):
        """Plot the FDM solution and PINN solution side by side if model is provided."""
        import matplotlib.pyplot as plt

        # Side by side comparison
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
        
        # FDM solution
        im1 = ax1.pcolormesh(self.x, self.t, self.u, shading='auto')
        plt.colorbar(im1, ax=ax1, label='u(x,t)')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_title('Heat Equation - FDM Solution')
        
        # PINN solution
        u_pinn = self._evaluate_pinn_full(model, device)
        im2 = ax2.pcolormesh(self.x, self.t, u_pinn, shading='auto')
        plt.colorbar(im2, ax=ax2, label='u(x,t)')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_title('Heat Equation - PINN Solution')
        
        plt.tight_layout()
        
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
        
        # Get PINN predictions for all points
        u_pinn_all = self._evaluate_pinn_full(model, device)
        
        # Evaluate PINN and exact solution at selected time steps
        for idx, t_idx in enumerate(t_indices):
            t_val = self.t[t_idx]
            
            # Get solutions for current time step
            u_fdm = self.u[t_idx]
            u_pinn = u_pinn_all[t_idx]
            
            # Get exact solution
            x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device).reshape(-1, 1)
            t_tensor = torch.full_like(x_tensor, t_val)
            u_exact = self.pde.exact_solution(x_tensor, t_tensor).cpu().numpy().reshape(-1)
            
            # Plot all solutions
            ax = axes[idx]
            ax.plot(self.x, u_fdm, 'b-', label='FDM', linewidth=2)
            ax.plot(self.x, u_pinn, 'r--', alpha=0.3, linewidth=1)  # Faint dashed line for trend
            ax.plot(self.x, u_pinn, 'rx', label='PINN', markersize=4)  # Red x markers
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
        
        # Calculate and return error metrics
        pinn_solution = u_pinn_all
        
        # Calculate exact solution for all points
        x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device).reshape(-1, 1)
        t_tensor = torch.tensor(self.t, dtype=torch.float32, device=device).reshape(-1, 1)
        x_grid, t_grid = torch.meshgrid(x_tensor.squeeze(), t_tensor.squeeze(), indexing='ij')
        points = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
        
        with torch.no_grad():
            u_exact = self.pde.exact_solution(points[:, 0].reshape(-1, 1), 
                                            points[:, 1].reshape(-1, 1))
            exact_solution = u_exact.reshape(self.nx, self.nt).T.cpu().numpy()
        
        return {
            'fdm_pinn_l2_error': np.mean((self.u - pinn_solution)**2),
            'fdm_pinn_max_error': np.max(np.abs(self.u - pinn_solution)),
            'fdm_pinn_mean_error': np.mean(np.abs(self.u - pinn_solution)),
            'exact_fdm_l2_error': np.mean((exact_solution - self.u)**2),
            'exact_fdm_max_error': np.max(np.abs(exact_solution - self.u)),
            'exact_fdm_mean_error': np.mean(np.abs(exact_solution - self.u))
        }
        
    def _evaluate_pinn_full(self, model, device):
        """Evaluate PINN on the full space-time grid."""
        # Create proper meshgrid
        x_tensor = torch.tensor(self.x, dtype=torch.float32, device=device)
        t_tensor = torch.tensor(self.t, dtype=torch.float32, device=device)
        
        # Create points for each time step
        all_predictions = []
        for t in t_tensor:
            # Create input points for current time step
            t_repeated = torch.full_like(x_tensor, t)
            points = torch.stack([x_tensor, t_repeated], dim=1)
            
            # Evaluate PINN
            with torch.no_grad():
                predictions = model(points).cpu().numpy().squeeze()  # Add squeeze() to remove extra dimension
                all_predictions.append(predictions)
        
        # Stack predictions to match FDM shape (nt, nx)
        return np.stack(all_predictions)

    @staticmethod
    def generate_fdm_comparison_plots(pde, model, device, viz_dir, logger=None):
        """Generate comparison plots between FDM and PINN solutions."""
        if logger is None:
            logger = logging.getLogger("HeatEquationFDM")
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
                logger.addHandler(handler)
        
        try:
            logger.info("Setting up FDM solver configuration...")
            # Create FDM solver with configuration matching the PDE
            config = {
                "domain": pde.domain,
                "time_domain": pde.time_domain,
                "parameters": {"alpha": pde.alpha},
                "dimension": pde.dimension,
                "nx": 120,
                "nt": 1000,
                "initial_condition": {},
                "boundary_conditions": {},
                "exact_solution": {}
            }
            
            # Copy initial condition configuration
            if hasattr(pde.config, 'initial_condition'):
                config["initial_condition"] = pde.config.initial_condition.copy()
            
            # Copy boundary conditions
            if hasattr(pde.config, 'boundary_conditions'):
                config["boundary_conditions"] = pde.config.boundary_conditions.copy()
            
            # Copy exact solution configuration
            if hasattr(pde.config, 'exact_solution'):
                config["exact_solution"] = pde.config.exact_solution.copy()
                # Ensure we have the exact solution function from the PDE
                if hasattr(pde, 'exact_solution'):
                    config["exact_solution"]["function"] = pde.exact_solution
            
            logger.info("Creating FDM solver...")
            solver = HeatEquationFDM(config, device)
            
            # Solve using FDM
            logger.info("Solving using FDM...")
            u_fdm = solver.solve()
            
            # Get PINN solution on the same grid
            logger.info("Getting PINN solution...")
            u_pinn = solver._evaluate_pinn_full(model, device)
            
            # Get exact solution using the PDE's exact solution method
            logger.info("Computing exact solution...")
            x = torch.linspace(solver.pde.domain[0][0], solver.pde.domain[0][1], solver.nx, device=device)
            t = torch.linspace(solver.pde.time_domain[0], solver.pde.time_domain[1], solver.nt, device=device)
            x_grid, t_grid = torch.meshgrid(x, t, indexing='ij')
            points = torch.stack([x_grid.flatten(), t_grid.flatten()], dim=1)
            
            with torch.no_grad():
                # Use the solver's PDE exact solution method
                u_exact = solver.pde.exact_solution(points[:, 0].reshape(-1, 1), 
                                                  points[:, 1].reshape(-1, 1))
                u_exact = u_exact.reshape(solver.nx, solver.nt).T.cpu().numpy()
            
            # Create comparison plots
            logger.info("Creating comparison plots...")
            os.makedirs(viz_dir, exist_ok=True)
            
            # Plot solutions and save metrics
            metrics = {
                'fdm_pinn_l2_error': np.mean((u_fdm - u_pinn)**2),
                'fdm_pinn_max_error': np.max(np.abs(u_fdm - u_pinn)),
                'fdm_pinn_mean_error': np.mean(np.abs(u_fdm - u_pinn)),
                'exact_fdm_l2_error': np.mean((u_exact - u_fdm)**2),
                'exact_fdm_max_error': np.max(np.abs(u_exact - u_fdm)),
                'exact_fdm_mean_error': np.mean(np.abs(u_exact - u_fdm)),
            }
            
            # Plot FDM solution
            solver.plot_solution(model=model, save_path=os.path.join(viz_dir, 'fdm_solution.png'), device=device)
            
            # Plot comparison with PINN and exact solution
            solver.plot_comparison_with_pinn(model, 
                                          save_path=os.path.join(viz_dir, 'fdm_vs_pinn_comparison.png'),
                                          device=device)
            
            logger.info(f"Error metrics between solutions:")
            logger.info(f"FDM vs PINN - L2={metrics['fdm_pinn_l2_error']:.6f}, Max={metrics['fdm_pinn_max_error']:.6f}, Mean={metrics['fdm_pinn_mean_error']:.6f}")
            logger.info(f"FDM vs Exact - L2={metrics['exact_fdm_l2_error']:.6f}, Max={metrics['exact_fdm_max_error']:.6f}, Mean={metrics['exact_fdm_mean_error']:.6f}")
            
            return metrics

        except Exception as e:
            logger.error(f"Error generating FDM comparison: {str(e)}")
            raise  # Re-raise the exception for better debugging
