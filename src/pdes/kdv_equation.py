# Korteweg–de Vries (KdV) equation
# Application domains: Water waves, solitons, fiber optics, plasma physics.
# Complexity: Higher-order (3rd derivative)


import torch
from .pde_base import PDEBase

class KdVEquation(PDEBase):
    """Korteweg–de Vries (KdV) equation: ∂u/∂t + 6u ∂u/∂x + ∂³u/∂x³ = 0"""

    def __init__(self, domain=(0, 1), device=None):
        super().__init__(domain, device)

    def compute_residual(self, model, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xxx = torch.autograd.grad(torch.autograd.grad(torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0]

        return u_t + 6 * u * u_x + u_xxx

    def boundary_conditions(self, x):
        x = x.to(self.device)
        return torch.sin(2 * torch.pi * x)  # Example boundary condition
