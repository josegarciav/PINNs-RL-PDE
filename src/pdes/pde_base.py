# Base class for PDEs


import torch

class PDEBase:
    """Base class for defining PDEs for PINNs."""

    def __init__(self, domain, device=None):
        """
        Initialize PDE with its domain and device.

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
        :param x: Collocation points in space (tensor).
        :param t: Optional time variable (tensor).
        :return: Residual loss tensor.
        """
        raise NotImplementedError("Subclasses must implement compute_residual()")

    def boundary_conditions(self, x, t=None):
        """
        Define boundary conditions for the PDE.
        This method must be implemented by subclasses.

        :param x: Collocation points in space (tensor).
        :param t: Optional time variable (tensor).
        :return: Expected boundary values (tensor).
        """
        raise NotImplementedError("Subclasses must implement boundary_conditions()")

    def enforce_boundary_conditions(self, model, x, t=None):
        """
        Computes and applies boundary conditions during training.
        """
        x = x.to(self.device)
        if t is not None:
            t = t.to(self.device)

        inputs = torch.cat([x, t], dim=1) if t is not None else x
        u_pred = model(inputs)

        bc = self.boundary_conditions(x, t)
        return torch.mean((u_pred - bc) ** 2)

    def exact_solution(self, x, t=None):
        """
        Compute the exact analytical solution for the PDE.
        This method must be implemented by subclasses.

        :param x: Collocation points in space (tensor).
        :param t: Optional time variable (tensor).
        :return: Exact solution tensor.
        """
        raise NotImplementedError("Subclasses must implement exact_solution()")
