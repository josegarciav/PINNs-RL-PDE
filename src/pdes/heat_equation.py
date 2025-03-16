# Heat equation

import torch
from .pde_base import PDEBase

class HeatEquation(PDEBase):
    """1D Heat equation: ∂u/∂t - α ∂²u/∂x² = 0"""

    def __init__(self, alpha=0.01, domain=(0, 1)):
        super().__init__(domain)
        self.alpha = alpha  # Thermal diffusivity

    def compute_residual(self, model, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0]

        return u_t - self.alpha * u_xx

    def boundary_conditions(self, x):
        return torch.sin(torch.pi * x)  # Example BC: u(0, t) = 0
