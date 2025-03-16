# Burgers' equation
# Application domains: Turbulence, traffic modeling, shock waves.
# Complexity: Nonlinear, 2nd-order


import torch
from .pde_base import PDEBase

class BurgersEquation(PDEBase):
    """1D Burgers' equation: ∂u/∂t + u ∂u/∂x - ν ∂²u/∂x² = 0"""

    def __init__(self, nu=0.01, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.nu = nu  # Viscosity coefficient

    def compute_residual(self, model, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

        return u_t + u * u_x - self.nu * u_xx

    def boundary_conditions(self, x):
        x = x.to(self.device)
        return torch.sin(torch.pi * x)  # Example boundary condition
