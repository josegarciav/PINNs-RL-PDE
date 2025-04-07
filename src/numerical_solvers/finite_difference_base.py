import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional, Callable
from dataclasses import dataclass

@dataclass
class FDMConfig:
    """Configuration for Finite Difference Method."""
    nx: int  # Number of spatial points
    nt: int  # Number of time points
    x_domain: Tuple[float, float]  # Spatial domain (x_min, x_max)
    t_domain: Tuple[float, float]  # Time domain (t_min, t_max)
    parameters: Dict[str, float]  # PDE parameters (e.g., alpha for heat equation)
    boundary_conditions: Dict[str, Any]  # Boundary conditions
    initial_condition: Dict[str, Any]  # Initial condition
    device: str = 'cpu'  # Device to use for computations

class FiniteDifferenceSolver:
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
        self.dx = (config.x_domain[1] - config.x_domain[0]) / (config.nx - 1)
        self.dt = (config.t_domain[1] - config.t_domain[0]) / (config.nt - 1)
        
        # Initialize grid points
        self.x = np.linspace(config.x_domain[0], config.x_domain[1], config.nx)
        self.t = np.linspace(config.t_domain[0], config.t_domain[1], config.nt)
        self.X, self.T = np.meshgrid(self.x, self.t)
        
        # Initialize solution array
        self.u = np.zeros((config.nt, config.nx))
        
        # Check stability (to be implemented by child classes)
        self.check_stability()

    def check_stability(self) -> None:
        """Check stability conditions for the numerical scheme."""
        raise NotImplementedError("Stability check must be implemented by child classes")

    def discretize_equation(self) -> None:
        """Discretize the PDE using finite differences."""
        raise NotImplementedError("Discretization must be implemented by child classes")

    def approximate_differential_terms(self, u: np.ndarray, n: int) -> Dict[str, np.ndarray]:
        """
        Approximate differential terms using finite differences.
        
        Args:
            u: Solution array at current time step
            n: Current time index
            
        Returns:
            Dictionary containing approximated terms
        """
        raise NotImplementedError("Term approximation must be implemented by child classes")

    def apply_boundary_conditions(self, n: int) -> None:
        """
        Apply boundary conditions at time step n.
        
        Args:
            n: Current time index
        """
        raise NotImplementedError("Boundary conditions must be implemented by child classes")

    def set_initial_condition(self, initial_condition_fn: Callable) -> None:
        """
        Set the initial condition.
        
        Args:
            initial_condition_fn: Function that takes x and returns initial values
        """
        x_tensor = torch.tensor(self.x.reshape(-1, 1), dtype=torch.float32)
        t_tensor = torch.zeros_like(x_tensor)
        self.u[0, :] = initial_condition_fn(x_tensor, t_tensor).numpy().flatten()

    def solve(self, initial_condition_fn: Callable) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the PDE using finite differences.
        
        Args:
            initial_condition_fn: Function that takes x and returns initial values
            
        Returns:
            Tuple containing (x grid, t grid, solution array)
        """
        self.set_initial_condition(initial_condition_fn)
        self.discretize_equation()
        return self.x, self.t, self.u

    def get_solution(self) -> np.ndarray:
        """Get the current solution array."""
        return self.u

    def interpolate(self, x_query: torch.Tensor, t_query: torch.Tensor) -> torch.Tensor:
        """
        Interpolate solution to arbitrary query points.
        
        Args:
            x_query: Query points in space
            t_query: Query points in time
            
        Returns:
            Interpolated solution values
        """
        from scipy.interpolate import RegularGridInterpolator
        
        interpolator = RegularGridInterpolator(
            (self.t, self.x), 
            self.u,
            method='linear',
            bounds_error=False,
            fill_value=None
        )
        
        x_np = x_query.numpy().flatten()
        t_np = t_query.numpy().flatten()
        query_points = np.column_stack((t_np, x_np))
        u_interp = interpolator(query_points)
        
        # Convert back to torch tensor and reshape
        return torch.tensor(u_interp, dtype=torch.float32).reshape(x_query.shape) 