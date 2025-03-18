# Cahn-Hilliard equation
# Application domains: Phase separation, material science, pattern formation
# Complexity: Nonlinear, 4th-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent

class CahnHilliardEquation(PDEBase):
    """
    Implementation of the Cahn-Hilliard Equation: ∂u/∂t = ∇²(ε²∇²u + u - u³)
    where ε is the interface width parameter and ∇² is the Laplacian operator.
    This equation describes phase separation and coarsening in binary mixtures.
    """
    
    def __init__(self, epsilon: float, domain: Union[Tuple[float, float], List[Tuple[float, float]]],
                 time_domain: Tuple[float, float], boundary_conditions: Dict[str, Dict[str, Any]],
                 initial_condition: Dict[str, Any], exact_solution: Dict[str, Any],
                 dimension: int = 1, device: Optional[torch.device] = None):
        """
        Initialize the Cahn-Hilliard Equation.
        
        :param epsilon: Interface width parameter
        :param domain: Spatial domain (tuple for 1D, list of tuples for higher dimensions)
        :param time_domain: Temporal domain
        :param boundary_conditions: Dictionary of boundary conditions
        :param initial_condition: Dictionary of initial condition parameters
        :param exact_solution: Dictionary of exact solution parameters
        :param dimension: Problem dimension (1 for 1D, 2 for 2D, etc.)
        :param device: Device to use for computations
        """
        config = PDEConfig(
            name="Cahn-Hilliard Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={'epsilon': epsilon},
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=dimension,
            device=device
        )
        super().__init__(config)
        self.epsilon = epsilon
    
    def compute_residual(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the Cahn-Hilliard equation residual.
        
        :param model: Neural network model
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Residual tensor
        """
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        
        # Compute derivatives
        u = model(xt)
        u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                                create_graph=True)[0]
        
        # Compute chemical potential (ε²∇²u + u - u³)
        if self.dimension == 1:
            u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                    create_graph=True)[0]
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                                     create_graph=True)[0]
            chemical_potential = self.epsilon**2 * u_xx + u - u**3
            mu_xx = torch.autograd.grad(chemical_potential, x, grad_outputs=torch.ones_like(chemical_potential),
                                      create_graph=True)[0]
            mu_xx = torch.autograd.grad(mu_xx, x, grad_outputs=torch.ones_like(mu_xx),
                                      create_graph=True)[0]
        else:
            # For higher dimensions, compute Laplacian for each dimension
            chemical_potential = torch.zeros_like(u)
            for dim in range(self.dimension):
                u_x = torch.autograd.grad(u, x[:, dim:dim+1], grad_outputs=torch.ones_like(u),
                                        create_graph=True)[0]
                u_xx = torch.autograd.grad(u_x, x[:, dim:dim+1], grad_outputs=torch.ones_like(u_x),
                                         create_graph=True)[0]
                chemical_potential += self.epsilon**2 * u_xx
            chemical_potential += u - u**3
            
            # Compute Laplacian of chemical potential
            mu_xx = torch.zeros_like(u)
            for dim in range(self.dimension):
                mu_x = torch.autograd.grad(chemical_potential, x[:, dim:dim+1],
                                         grad_outputs=torch.ones_like(chemical_potential),
                                         create_graph=True)[0]
                mu_xx += torch.autograd.grad(mu_x, x[:, dim:dim+1],
                                           grad_outputs=torch.ones_like(mu_x),
                                           create_graph=True)[0]
        
        # Compute residual (∂u/∂t = ∇²(ε²∇²u + u - u³))
        return u_t - mu_xx
    
    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (tanh profile).
        
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            # 1D Cahn-Hilliard solution (tanh profile)
            return torch.tanh(x / (2 * self.epsilon))
        else:
            # For higher dimensions, use product of tanh profiles
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                solution *= torch.tanh(x[:, dim:dim+1] / (2 * self.epsilon))
            return solution
    
    def _create_boundary_condition(self, bc_type: str, params: Dict[str, Any]) -> callable:
        """
        Create boundary condition function from parameters.
        
        :param bc_type: Type of boundary condition
        :param params: Parameters for the boundary condition
        :return: Boundary condition function
        """
        if bc_type == 'initial':
            ic_type = params.get('type', 'tanh')
            if ic_type == 'tanh':
                if self.dimension == 1:
                    return lambda x, t: torch.tanh(x / (2 * self.epsilon))
                else:
                    return lambda x, t: torch.tanh(torch.sum(x, dim=1, keepdim=True) / (2 * self.epsilon))
            else:
                raise ValueError(f"Unsupported initial condition type: {ic_type}")
        else:
            return super()._create_boundary_condition(bc_type, params)
    
    def validate(self, model, num_points=1000):
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
            'l2_error': torch.mean(error**2).item(),
            'max_error': torch.max(error).item(),
            'mean_error': torch.mean(error).item()
        }
