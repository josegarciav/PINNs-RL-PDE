# Convection equation
# Application domains: Fluid flow, biological transport, industrial processing.
# Complexity: Simple, 1st-order linear


import torch
from .pde_base import PDEBase

class ConvectionEquation(PDEBase):
    """1D Convection equation: ∂u/∂t + c ∂u/∂x = 0"""

    def __init__(self, c=1.0, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.c = c  # Convection speed

    def compute_residual(self, model, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]

        return u_t + self.c * u_x

    def boundary_conditions(self, x):
        x = x.to(self.device)
        return torch.sin(torch.pi * x)  # Example boundary condition

    def exact_solution(self, x, t):
        """Exact analytical solution for Convection equation: u(x,t) = sin(pi(x - c*t))"""
        return None
