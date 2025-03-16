# Base class for PDEs

import torch

class PDEBase:
    """Base class for defining PDEs for PINNs."""

    def __init__(self, domain):
        """
        Initialize PDE with its domain.
        :param domain: Tuple (min, max) indicating the range of the PDE.
        """
        self.domain = domain

    def compute_residual(self, model, x):
        """
        Compute the PDE residual for a given neural network model.
        This method must be implemented by subclasses.
        :param model: The PINN model.
        :param x: Collocation points.
        :return: Residual loss.
        """
        raise NotImplementedError("Subclasses must implement compute_residual()")

    def boundary_conditions(self, x):
        """
        Define boundary conditions for the PDE.
        This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement boundary_conditions()")
