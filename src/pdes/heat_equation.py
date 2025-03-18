# Heat equation
# Application domains: Heat conduction, diffusion processes
# Complexity: Linear, 2nd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent

class HeatEquation(PDEBase):
    """
    Implementation of the Heat Equation: ∂u/∂t = α∇²u
    where α is the thermal diffusivity and ∇² is the Laplacian operator.
    """
    
    def __init__(self, alpha: float, domain: Union[Tuple[float, float], List[Tuple[float, float]]],
                 time_domain: Tuple[float, float], boundary_conditions: Dict[str, Dict[str, Any]],
                 initial_condition: Dict[str, Any], exact_solution: Dict[str, Any],
                 dimension: int = 1, device: Optional[torch.device] = None):
        """
        Initialize the Heat Equation.
        
        :param alpha: Thermal diffusivity
        :param domain: Spatial domain (tuple for 1D, list of tuples for higher dimensions)
        :param time_domain: Temporal domain
        :param boundary_conditions: Dictionary of boundary conditions
        :param initial_condition: Dictionary of initial condition parameters
        :param exact_solution: Dictionary of exact solution parameters
        :param dimension: Problem dimension (1 for 1D, 2 for 2D, etc.)
        :param device: Device to use for computations
        """
        config = PDEConfig(
            name="Heat Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={'alpha': alpha},
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=dimension,
            device=device
        )
        super().__init__(config)
        self.alpha = alpha
    
    def compute_residual(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the heat equation residual.
        
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
        
        # Compute Laplacian based on dimension
        if self.dimension == 1:
            u_xx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                     create_graph=True)[0]
            u_xx = torch.autograd.grad(u_xx, x, grad_outputs=torch.ones_like(u_xx),
                                     create_graph=True)[0]
            laplacian = u_xx
        else:
            # For higher dimensions, compute Laplacian as sum of second derivatives
            laplacian = torch.zeros_like(u)
            for dim in range(self.dimension):
                u_xx = torch.autograd.grad(u, x[:, dim:dim+1], grad_outputs=torch.ones_like(u),
                                         create_graph=True)[0]
                u_xx = torch.autograd.grad(u_xx, x[:, dim:dim+1], grad_outputs=torch.ones_like(u_xx),
                                         create_graph=True)[0]
                laplacian += u_xx
        
        return u_t - self.alpha * laplacian
    
    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution.
        
        :param x: Spatial coordinates
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            return torch.exp(-self.alpha * (2 * torch.pi)**2 * t) * torch.sin(2 * torch.pi * x)
        else:
            # For higher dimensions, use product of sine waves
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                solution *= torch.exp(-self.alpha * (2 * torch.pi)**2 * t) * torch.sin(2 * torch.pi * x[:, dim:dim+1])
            return solution
    
    def _create_boundary_condition(self, bc_type: str, params: Dict[str, Any]) -> callable:
        """
        Create boundary condition function from parameters.
        
        :param bc_type: Type of boundary condition
        :param params: Parameters for the boundary condition
        :return: Boundary condition function
        """
        if bc_type == 'initial':
            ic_type = params.get('type', 'sine')
            if ic_type == 'sine':
                A = params.get('amplitude', 1.0)
                k = params.get('frequency', 2.0)
                if self.dimension == 1:
                    return lambda x, t: A * torch.sin(k * torch.pi * x)
                else:
                    return lambda x, t: A * torch.sin(k * torch.pi * torch.sum(x, dim=1, keepdim=True))
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
