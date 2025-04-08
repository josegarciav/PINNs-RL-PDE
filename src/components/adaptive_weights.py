import torch
import numpy as np


class AdaptiveLossWeights:
    """Handles adaptive weighting of different loss components in PINNs."""

    def __init__(self, strategy="rbw", alpha=0.9, eps=1e-5):
        """
        Initialize adaptive loss weights handler.

        Args:
            strategy (str): Weight adaptation strategy ('lrw' or 'rbw')
            alpha (float): Moving average factor for weight updates
            eps (float): Small constant for numerical stability
        """
        self.strategy = strategy.lower()
        self.alpha = alpha
        self.eps = eps
        self.weights = None
        self.running_losses = None
        self.running_grads = None

    def update_weights_lrw(self, gradients):
        """
        Update weights using Learning Rate based Weighting (LRW).

        Args:
            gradients (list): List of gradient norms for each loss component
        """
        if self.running_grads is None:
            self.running_grads = gradients
            self.weights = torch.ones_like(gradients)
            return self.weights

        # Update running average of gradients
        self.running_grads = (
            self.alpha * self.running_grads + (1 - self.alpha) * gradients
        )

        # Compute weights inversely proportional to gradient magnitudes
        inv_grads = 1.0 / (self.running_grads + self.eps)
        self.weights = inv_grads / torch.sum(inv_grads)

        print(
            f"LRW - Updated weights: gradients={gradients}, running_grads={self.running_grads}, weights={self.weights}"
        )

        return self.weights

    def update_weights_rbw(self, losses):
        """
        Update weights using Relative Error based Weighting (RBW).

        Args:
            losses (list): List of individual loss components
        """
        if self.running_losses is None:
            self.running_losses = losses
            self.weights = torch.ones_like(losses)
            return self.weights

        # Update running average of losses
        self.running_losses = (
            self.alpha * self.running_losses + (1 - self.alpha) * losses
        )

        # Compute weights based on relative error magnitudes
        max_loss = torch.max(self.running_losses)
        relative_losses = self.running_losses / (max_loss + self.eps)
        self.weights = relative_losses / torch.sum(relative_losses)

        print(
            f"RBW - Updated weights: losses={losses}, running_losses={self.running_losses}, weights={self.weights}"
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
        return (
            self.weights if self.weights is not None else torch.ones(3) / 3.0
        )  # Default to equal weights for 3 components
