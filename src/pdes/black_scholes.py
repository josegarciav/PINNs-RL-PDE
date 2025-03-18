# Black-Scholes equation
# Application domains: Financial derivatives, option pricing
# Complexity: Nonlinear, 2nd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional, Union, Tuple, List
from src.rl_agent import RLAgent

class BlackScholesEquation(PDEBase):
    """
    Implementation of the Black-Scholes Equation: ∂V/∂t + (1/2)σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0
    where V is the option price, S is the stock price, σ is volatility, and r is the risk-free rate.
    """
    
    def __init__(self, sigma: float, r: float, domain: Union[Tuple[float, float], List[Tuple[float, float]]],
                 time_domain: Tuple[float, float], boundary_conditions: Dict[str, Dict[str, Any]],
                 initial_condition: Dict[str, Any], exact_solution: Dict[str, Any],
                 dimension: int = 1, device: Optional[torch.device] = None):
        """
        Initialize the Black-Scholes Equation.
        
        :param sigma: Volatility
        :param r: Risk-free rate
        :param domain: Spatial domain (tuple for 1D, list of tuples for higher dimensions)
        :param time_domain: Temporal domain
        :param boundary_conditions: Dictionary of boundary conditions
        :param initial_condition: Dictionary of initial condition parameters
        :param exact_solution: Dictionary of exact solution parameters
        :param dimension: Problem dimension (1 for 1D, 2 for 2D, etc.)
        :param device: Device to use for computations
        """
        config = PDEConfig(
            name="Black-Scholes Equation",
            domain=domain,
            time_domain=time_domain,
            parameters={'sigma': sigma, 'r': r},
            boundary_conditions=boundary_conditions,
            initial_condition=initial_condition,
            exact_solution=exact_solution,
            dimension=dimension,
            device=device
        )
        super().__init__(config)
        self.sigma = sigma
        self.r = r
    
    def compute_residual(self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute the Black-Scholes equation residual.
        
        :param model: Neural network model
        :param x: Spatial coordinates (stock prices)
        :param t: Time coordinates
        :return: Residual tensor
        """
        xt = torch.cat([x, t], dim=1)
        xt.requires_grad_(True)
        
        # Compute derivatives
        V = model(xt)
        V_t = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V),
                                create_graph=True)[0]
        
        # Compute first and second derivatives with respect to S
        if self.dimension == 1:
            V_S = torch.autograd.grad(V, x, grad_outputs=torch.ones_like(V),
                                    create_graph=True)[0]
            V_SS = torch.autograd.grad(V_S, x, grad_outputs=torch.ones_like(V_S),
                                     create_graph=True)[0]
        else:
            # For higher dimensions, compute derivatives for each dimension
            V_S = torch.zeros_like(V)
            V_SS = torch.zeros_like(V)
            for dim in range(self.dimension):
                V_S_dim = torch.autograd.grad(V, x[:, dim:dim+1], grad_outputs=torch.ones_like(V),
                                            create_graph=True)[0]
                V_S += V_S_dim
                V_SS_dim = torch.autograd.grad(V_S_dim, x[:, dim:dim+1], grad_outputs=torch.ones_like(V_S_dim),
                                             create_graph=True)[0]
                V_SS += V_SS_dim
        
        # Compute residual
        return V_t + 0.5 * self.sigma**2 * x**2 * V_SS + self.r * x * V_S - self.r * V
    
    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute exact analytical solution (European call option).
        
        :param x: Spatial coordinates (stock prices)
        :param t: Time coordinates
        :return: Exact solution tensor
        """
        if self.dimension == 1:
            # 1D Black-Scholes solution (European call option)
            K = 1.0  # Strike price
            d1 = (torch.log(x/K) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * torch.sqrt(t))
            d2 = d1 - self.sigma * torch.sqrt(t)
            return x * torch.erf(d1) - K * torch.exp(-self.r * t) * torch.erf(d2)
        else:
            # For higher dimensions, use product of 1D solutions
            solution = torch.ones_like(x[:, 0:1])
            for dim in range(self.dimension):
                K = 1.0  # Strike price
                d1 = (torch.log(x[:, dim:dim+1]/K) + (self.r + 0.5 * self.sigma**2) * t) / (self.sigma * torch.sqrt(t))
                d2 = d1 - self.sigma * torch.sqrt(t)
                solution *= x[:, dim:dim+1] * torch.erf(d1) - K * torch.exp(-self.r * t) * torch.erf(d2)
            return solution
    
    def _create_boundary_condition(self, bc_type: str, params: Dict[str, Any]) -> callable:
        """
        Create boundary condition function from parameters.
        
        :param bc_type: Type of boundary condition
        :param params: Parameters for the boundary condition
        :return: Boundary condition function
        """
        if bc_type == 'initial':
            ic_type = params.get('type', 'call_option')
            if ic_type == 'call_option':
                K = params.get('strike_price', 1.0)
                if self.dimension == 1:
                    return lambda x, t: torch.maximum(x - K, torch.zeros_like(x))
                else:
                    return lambda x, t: torch.maximum(torch.sum(x, dim=1, keepdim=True) - K, torch.zeros_like(x[:, 0:1]))
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
        V_pred = model(torch.cat([x, t], dim=1))
        V_exact = self.exact_solution(x, t)
        error = torch.abs(V_pred - V_exact)
        return {
            'l2_error': torch.mean(error**2).item(),
            'max_error': torch.max(error).item(),
            'mean_error': torch.mean(error).item()
        }
