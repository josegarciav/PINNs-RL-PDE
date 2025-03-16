# Allen-Cahn equation
# Application domains: Materials science, phase transitions, biological morphogenesis.
# Complexity: Nonlinear, 2nd-order


import torch
from .pde_base import PDEBase

class AllenCahnEquation(PDEBase):
    """Allen-Cahn equation: ∂u/∂t - D ∂²u/∂x² + u - u³ = 0"""

    def __init__(self, D=0.01, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.D = D  # Diffusion coefficient

    def compute_residual(self, model, x, t):
        x = x.to(self.device)
        t = t.to(self.device)
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((x, t), dim=1))
        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0], x, torch.ones_like(u), create_graph=True)[0]

        return u_t - self.D * u_xx + u - u**3

    def boundary_conditions(self, x):
        x = x.to(self.device)
        return torch.tanh(5 * (x - 0.5))

    def exact_solution(self, x, t):
        """Exact analytical solution for Allen-Cahn equation: u(x,t) = tanh(5(x - 0.5))"""
        return None
