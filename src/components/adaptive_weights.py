import torch
import numpy as np
import logging


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
        # Ensure eps is a float, not a string
        self.eps = float(eps) if isinstance(eps, (str, int)) else eps
        self.weights = None
        self.running_losses = None
        self.running_grads = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"AdaptiveLossWeights initialized with strategy={strategy}, alpha={alpha}, eps={self.eps}")

    def update_weights_lrw(self, gradients):
        """
        Update weights using Learning Rate based Weighting (LRW).

        Args:
            gradients (list): List of gradient norms for each loss component
        """
        if self.running_grads is None:
            self.running_grads = gradients
            self.weights = torch.ones_like(gradients)
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

        Args:
            losses (list): List of individual loss components
        """
        if self.running_losses is None:
            self.running_losses = losses
            self.weights = torch.ones_like(losses)
            self.logger.info(f"RBW - Initialized weights: {self.weights}")
            return self.weights

        # Update running average of losses
        self.running_losses = (
            self.alpha * self.running_losses + (1 - self.alpha) * losses
        )

        # Ensure eps is a tensor on the same device as the losses
        eps_tensor = torch.tensor(self.eps, device=self.running_losses.device)
        
        # Compute weights based on relative error magnitudes
        max_loss = torch.max(self.running_losses)
        relative_losses = self.running_losses / (max_loss + eps_tensor)
        self.weights = relative_losses / torch.sum(relative_losses)

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
        return (
            self.weights if self.weights is not None else torch.ones(3) / 3.0
        )  # Default to equal weights for 3 components
