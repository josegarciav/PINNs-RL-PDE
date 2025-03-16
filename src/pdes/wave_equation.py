# Wave equation
# Application domains: Acoustics, electromagnetics, seismology, structural mechanics.
# Complexity: Simple, 2nd-order linear


import torch
from .pde_base import PDEBase

class WaveEquation(PDEBase):
    """1D Wave equation: ∂²u/∂t² - c² ∂²u/∂x² = 0"""

    def __init__(self, c=1.0, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.c = c  # Wave speed

    def compute_residual(self, model, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))

        u_tt = torch.autograd.grad(torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0], t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0]

        return u_tt - self.c**2 * u_xx

    def boundary_conditions(self, x):
        x = x.to(self.device)
        return torch.sin(torch.pi * x)  # Example BC
