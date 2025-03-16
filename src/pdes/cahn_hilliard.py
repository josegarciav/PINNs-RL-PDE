# Cahn-Hilliard equation
# Application domains: Phase separation, 3D printing, image processing.
# Complexity: Higher-order (4th derivative)


import torch
from .pde_base import PDEBase

class CahnHilliardEquation(PDEBase):
    """Cahn-Hilliard equation: ∂u/∂t = D ∂²/∂x² (u³ - u - ε² ∂²u/∂x²)"""

    def __init__(self, D=1.0, epsilon=0.01, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.D = D
        self.epsilon = epsilon

    def compute_residual(self, model, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0]
        u_xxxx = torch.autograd.grad(torch.autograd.grad(u_xx, x, torch.ones_like(u_xx), create_graph=True)[0], x, torch.ones_like(u_xx), create_graph=True)[0]

        return u_t - self.D * (u**3 - u - self.epsilon**2 * u_xxxx)

    def boundary_conditions(self, x):
        x = x.to(self.device)
        return torch.sin(torch.pi * x)

    def exact_solution(self, x, t):
        """Exact analytical solution for Cahn-Hilliard equation: u(x,t) = sin(pi*x) * exp(-D*pi^2*t)"""
        return None
