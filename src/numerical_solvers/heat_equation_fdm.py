import numpy as np
from typing import Dict
from src.numerical_solvers.finite_difference_base import FiniteDifferenceSolver, FDMConfig

class HeatEquationFDM(FiniteDifferenceSolver):
    """Finite difference solver for the heat equation."""
    
    def check_stability(self) -> None:
        """Check the stability condition for FTCS scheme."""
        alpha = self.config.parameters.get('alpha', 0.1)
        stability = alpha * self.dt / (self.dx * self.dx)
        if stability > 0.5:
            print(f"Warning: Scheme might be unstable. Stability parameter = {stability} > 0.5")

    def discretize_equation(self) -> None:
        """Implement FTCS scheme for the heat equation."""
        alpha = self.config.parameters.get('alpha', 0.1)
        r = alpha * self.dt / (self.dx * self.dx)
        
        for n in range(0, self.config.nt - 1):
            # Compute approximations
            terms = self.approximate_differential_terms(self.u[n], n)
            
            # Update solution
            self.u[n + 1, 1:-1] = self.u[n, 1:-1] + r * terms['d2u_dx2']
            
            # Apply boundary conditions
            self.apply_boundary_conditions(n + 1)

    def approximate_differential_terms(self, u: np.ndarray, n: int) -> Dict[str, np.ndarray]:
        """Approximate spatial derivatives for the heat equation."""
        # Second derivative in space (central difference)
        d2u_dx2 = (u[2:] - 2 * u[1:-1] + u[:-2])
        
        return {'d2u_dx2': d2u_dx2}

    def apply_boundary_conditions(self, n: int) -> None:
        """Apply boundary conditions for the heat equation."""
        bc_type = next(iter(self.config.boundary_conditions))  # Get first BC type
        
        if bc_type == 'periodic':
            # Periodic boundary conditions
            self.u[n, 0] = self.u[n, -2]  # Left boundary
            self.u[n, -1] = self.u[n, 1]  # Right boundary
        elif bc_type == 'dirichlet':
            # Dirichlet boundary conditions
            value = self.config.boundary_conditions[bc_type].get('value', 0.0)
            self.u[n, 0] = value  # Left boundary
            self.u[n, -1] = value  # Right boundary
        else:
            raise ValueError(f"Unsupported boundary condition type: {bc_type}")
