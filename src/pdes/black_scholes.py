# Black-Scholes equation

import torch
from .pde_base import PDEBase

class BlackScholesEquation(PDEBase):
    """Black-Scholes equation for option pricing."""

    def __init__(self, r=0.05, sigma=0.2, domain=(0, 1)):
        super().__init__(domain)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility

    def compute_residual(self, model, S, t):
        S.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((S, t), dim=1))

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_S = torch.autograd.grad(u, S, torch.ones_like(u), create_graph=True)[0]
        u_SS = torch.autograd.grad(u_S, S, torch.ones_like(u_S), create_graph=True)[0]

        return u_t + 0.5 * self.sigma**2 * S**2 * u_SS + self.r * S * u_S - self.r * u

    def boundary_conditions(self, S):
        return torch.maximum(S - 1, torch.zeros_like(S))  # Call option boundary condition
