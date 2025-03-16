# Heat equation
# Application domains: Heat transfer, climate modeling, biomedical engineering.
# Complexity: Simple, 2nd-order linear

import torch
import numpy as np
from .pde_base import PDEBase


class HeatEquation(PDEBase):
    def __init__(self, alpha=0.01, domain=(0, 1), device=None):
        super().__init__(domain, device)
        self.alpha = alpha

    def compute_residual(self, model, x, t):
        x.requires_grad_(True)
        t.requires_grad_(True)
        inputs = torch.cat([x, t], dim=1).to(self.device)
        u = model(inputs)

        u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u), create_graph=True)[0]

        return u_t - self.alpha * u_xx

    def boundary_conditions(self, x, t):
        x, t = x.to(self.device), t.to(self.device)

        # Initial condition: u(x,0) = sin(pi*x)
        initial_condition = torch.sin(np.pi * x) * (t == 0).float()

        # Boundary condition: u(0,t)=u(1,t)=0
        boundary_condition = torch.zeros_like(x)

        # Combine initial and boundary conditions logically:
        return torch.where(t == 0, initial_condition, boundary_condition)
