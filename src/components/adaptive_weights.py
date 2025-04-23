import torch
import numpy as np
import logging


class AdaptiveLossWeights:
    """Handles adaptive weighting of different loss components in PINNs."""

    def __init__(self, strategy="rbw", alpha=0.9, eps=1e-5, initial_weights=None):
        """
        Initialize adaptive loss weights handler.

        Args:
            strategy (str): Weight adaptation strategy ('lrw' or 'rbw')
            alpha (float): Moving average factor for weight updates
            eps (float): Small constant for numerical stability
            initial_weights (list): Initial weights for [pde, boundary, initial] components
        """
        self.strategy = strategy.lower()
        self.alpha = alpha
        # Ensure eps is a float, not a string
        self.eps = float(eps) if isinstance(eps, (str, int)) else eps
        self.initial_weights = torch.tensor(initial_weights) if initial_weights is not None else None
        self.weights = None
        self.running_losses = None
        self.running_grads = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"AdaptiveLossWeights initialized with strategy={strategy}, alpha={alpha}, eps={self.eps}, initial_weights={initial_weights}"
        )
        self.prev_weights = None

    def update_weights_lrw(self, gradients):
        """
        Update weights using Learning Rate based Weighting (LRW).

        Args:
            gradients (list): List of gradient norms for each loss component
        """
        if self.running_grads is None:
            self.running_grads = gradients
            self.weights = self.initial_weights.to(gradients.device) if self.initial_weights is not None else torch.ones_like(gradients)
            self.logger.info(f"LRW - Initialized weights: {self.weights}")
            return self.weights

        # Update running average of gradients
        self.running_grads = (
            self.alpha * self.running_grads + (1 - self.alpha) * gradients
        )

        # Ensure eps is a tensor on the same device as the gradients
        eps_tensor = torch.tensor(self.eps, device=self.running_grads.device)

        # Compute weights inversely proportional to gradient magnitudes
        inv_grads = 1.0 / (self.running_grads + eps_tensor)
        self.weights = inv_grads / torch.sum(inv_grads)

        self.logger.info(
            f"LRW - Updated weights: gradients={gradients.detach().cpu().numpy()}, running_grads={self.running_grads.detach().cpu().numpy()}, weights={self.weights.detach().cpu().numpy()}"
        )

        return self.weights

    def update_weights_rbw(self, losses):
        """
        Update weights using Relative Error based Weighting (RBW).
        Gives more weight to terms with bigger losses to balance them.

        Args:
            losses (list): List of individual loss components
        """
        if self.running_losses is None:
            self.running_losses = losses
            self.weights = self.initial_weights.to(losses.device) if self.initial_weights is not None else torch.ones_like(losses)
            self.logger.info(f"RBW - Initialized weights: {self.weights}")
            return self.weights

        # Update running average of losses
        self.running_losses = (
            self.alpha * self.running_losses + (1 - self.alpha) * losses
        )

        # Ensure eps is a tensor on the same device as the losses
        eps_tensor = torch.tensor(self.eps, device=self.running_losses.device)

        # Normalize the running losses to get weights
        # Give more weight to higher losses to focus optimization on problematic terms
        normalized_losses = self.running_losses / (self.running_losses.sum() + eps_tensor)
        self.weights = normalized_losses  # Higher loss -> Higher weight

        # Update weights with exponential moving average
        if self.prev_weights is not None:
            self.weights = self.alpha * self.prev_weights + (1 - self.alpha) * self.weights
        
        self.prev_weights = self.weights.clone()

        self.logger.info(
            f"RBW - Updated weights: losses={losses.detach().cpu().numpy()}, running_losses={self.running_losses.detach().cpu().numpy()}, weights={self.weights.detach().cpu().numpy()}"
        )

        return self.weights

    def update(self, losses=None, gradients=None):
        """
        Update weights based on chosen strategy.

        Args:
            losses (list): List of individual loss components
            gradients (list): List of gradient norms for each loss component

        Returns:
            torch.Tensor: Updated loss weights
        """
        if self.strategy == "lrw" and gradients is not None:
            return self.update_weights_lrw(gradients)
        elif self.strategy == "rbw" and losses is not None:
            return self.update_weights_rbw(losses)
        else:
            raise ValueError(
                f"Invalid combination of strategy ({self.strategy}) and inputs"
            )

    def get_weights(self):
        """Return current weights."""
        if self.weights is not None:
            return self.weights
        elif self.initial_weights is not None:
            return self.initial_weights
        else:
            return torch.ones(3) / 3.0  # Default to equal weights for 3 components
