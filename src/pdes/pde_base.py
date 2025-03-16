# Base class for PDEs

import torch

class PDEBase:
    """Base class for defining PDEs for PINNs."""

    def __init__(self, domain, device=None):
        """
        Initialize PDE with its domain.

        :param domain: Tuple (min, max) indicating the range of the PDE.
        :param device: Torch device (CPU/GPU/MPS).
        """
        self.domain = domain
        self.device = device if device else torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    def compute_residual(self, model, x, t=None):
        """
        Compute the PDE residual for a given neural network model.
        This method must be implemented by subclasses.

        :param model: The PINN model.
        :param x: Collocation points in space.
        :param t: Optional time variable.
        :return: Residual loss tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_residual()")

    def boundary_conditions(self, x, t=None):
        """
        Define boundary conditions for the PDE.
        This method must be implemented by subclasses.
        
        :param x: Collocation points in space.
        :param t: Optional time variable.
        :return: Boundary condition enforcement tensor.
        """
        raise NotImplementedError("Subclasses must implement boundary_conditions()")

    def enforce_boundary_conditions(self, model, x, t=None):
        """
        Computes and applies boundary conditions during training.
        
        :param model: The PINN model.
        :param x: Boundary points in space.
        :param t: Optional time variable.
        :return: Loss from boundary conditions.
        """
        bc = self.boundary_conditions(x, t)
        u_pred = model(torch.cat([x, t], dim=1) if t is not None else x)
        return torch.mean((u_pred - bc) ** 2)  # MSE loss for BCs
