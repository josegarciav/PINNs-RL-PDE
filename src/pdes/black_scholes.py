# Black-Scholes equation
# Application domains: Finance, risk management, derivatives pricing.
# Complexity: Simple, 2nd-order linear


import torch
from .pde_base import PDEBase

class BlackScholesEquation(PDEBase):
    """Black-Scholes equation for option pricing: ∂u/∂t + (1/2) σ² S² ∂²u/∂S² + r S ∂u/∂S - r u = 0"""

    def __init__(self, r=0.05, sigma=0.2, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.r = r  # Risk-free rate
        self.sigma = sigma  # Volatility

    def compute_residual(self, model, S, t):
        S = S.to(self.device)
        t = t.to(self.device)
        S.requires_grad_(True)
        t.requires_grad_(True)

        u = model(torch.cat((S, t), dim=1))

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_S = torch.autograd.grad(u, S, torch.ones_like(u), create_graph=True)[0]
        u_SS = torch.autograd.grad(u_S, S, torch.ones_like(u_S), create_graph=True)[0]

        return u_t + 0.5 * self.sigma**2 * S**2 * u_SS + self.r * S * u_S - self.r * u

    def boundary_conditions(self, S):
        S = S.to(self.device)
        return torch.maximum(S - 1, torch.zeros_like(S))  # Call option boundary condition
