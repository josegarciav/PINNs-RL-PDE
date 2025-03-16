# Wave equation

import torch
from .pde_base import PDEBase

class WaveEquation(PDEBase):
    """1D Wave equation: ∂²u/∂t² - c² ∂²u/∂x² = 0"""

    def __init__(self, c=1.0, domain=(0, 1)):
        super().__init__(domain)
        self.c = c  # Wave speed

    def compute_residual(self, model, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))

        u_tt = torch.autograd.grad(torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0], t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0]

        return u_tt - self.c**2 * u_xx

    def boundary_conditions(self, x):
        return torch.sin(torch.pi * x)  # Example BC

