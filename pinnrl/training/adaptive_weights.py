"""Adaptive loss weighting: gradient-norm (``lrw``) and relative-loss (``rbw``) balancing."""

import logging

import torch


class AdaptiveLossWeights:
    """Adaptive weights for the [residual, boundary, initial] loss components.

    The first :meth:`update` call returns ``initial_weights`` (or ones) and
    seeds the running average; later calls return weights that sum to 1.
    """

    def __init__(self, strategy="rbw", alpha=0.9, eps=1e-5, initial_weights=None):
        """Initialize adaptive loss weights handler.

        Args:
            strategy (str): Weight adaptation strategy ('lrw' or 'rbw').
            alpha (float): EMA retention factor: ``new = alpha * old + (1 - alpha) * current``.
            eps (float): Small constant for numerical stability.
            initial_weights (list | None): Weights for [residual, boundary, initial]
                returned by the first update; ``None`` means ones.
        """
        self.strategy = strategy.lower()
        self.alpha = alpha
        # Ensure eps is a float, not a string
        self.eps = float(eps) if isinstance(eps, (str, int)) else eps
        self.initial_weights = (
            torch.tensor(initial_weights) if initial_weights is not None else None
        )
        self.weights = None
        self.running_losses = None
        self.running_grads = None
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            f"AdaptiveLossWeights initialized with strategy={strategy}, alpha={alpha}, eps={self.eps}, initial_weights={initial_weights}"
        )
        self.prev_weights = None

    def update_weights_lrw(self, gradients):
        """Update weights using Learning Rate based Weighting (LRW).

        After the first call (which only seeds ``g_bar``)::

            g_bar_i = alpha * g_bar_i + (1 - alpha) * g_i
            w_i = (1 / (g_bar_i + eps)) / sum_{j active} (1 / (g_bar_j + eps))

        over the *active* components (``g_bar_i > 0``), so components with smaller
        gradient norms get larger weights. A component whose gradient norm is zero
        (a constant loss, e.g. a PDE with no boundary conditions) gets weight 0:
        it cannot be minimised, and ``1 / eps`` would otherwise hand it almost all
        the weight and switch the other terms off. If no component is active the
        weights are uniform.

        Args:
            gradients (torch.Tensor): Gradient norm of each loss component, shape ``(n,)``.

        Returns:
            torch.Tensor: The weights, shape ``(n,)``.
        """
        if self.running_grads is None:
            self.running_grads = gradients
            self.weights = (
                self.initial_weights.to(gradients.device)
                if self.initial_weights is not None
                else torch.ones_like(gradients)
            )
            self.logger.debug(f"LRW - Initialized weights: {self.weights}")
            return self.weights

        # Update running average of gradients
        self.running_grads = self.alpha * self.running_grads + (1 - self.alpha) * gradients

        # Ensure eps is a tensor on the same device as the gradients
        eps_tensor = torch.tensor(self.eps, device=self.running_grads.device)

        # Inverse-gradient weights over the components that have a gradient at all.
        active = self.running_grads > 0
        if not torch.any(active):
            self.weights = torch.full_like(self.running_grads, 1.0 / len(self.running_grads))
        else:
            inv_grads = torch.where(
                active,
                1.0 / (self.running_grads + eps_tensor),
                torch.zeros_like(self.running_grads),
            )
            self.weights = inv_grads / torch.sum(inv_grads)

        self.logger.debug(
            f"LRW - Updated weights: gradients={gradients.detach().cpu().numpy()}, running_grads={self.running_grads.detach().cpu().numpy()}, weights={self.weights.detach().cpu().numpy()}"
        )

        return self.weights

    def update_weights_rbw(self, losses):
        """Update weights using Relative Error based Weighting (RBW).

        Gives more weight to terms with bigger losses. After the first call
        (which only seeds ``L_bar``)::

            L_bar_i = alpha * L_bar_i + (1 - alpha) * L_i
            w_i = alpha * w_prev_i + (1 - alpha) * L_bar_i / (sum_j L_bar_j + eps)

        where ``w_prev`` is the previous RBW result; on the second call there is
        none yet and ``w_i`` is the normalised term alone.

        Args:
            losses (torch.Tensor): Detached value of each loss component, shape ``(n,)``.

        Returns:
            torch.Tensor: The weights, shape ``(n,)``.
        """
        if self.running_losses is None:
            self.running_losses = losses
            self.weights = (
                self.initial_weights.to(losses.device)
                if self.initial_weights is not None
                else torch.ones_like(losses)
            )
            self.logger.info(f"RBW - Initialized weights: {self.weights}")
            return self.weights

        # Update running average of losses
        self.running_losses = self.alpha * self.running_losses + (1 - self.alpha) * losses

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

        self.logger.debug(
            f"RBW - Updated weights: losses={losses.detach().cpu().numpy()}, running_losses={self.running_losses.detach().cpu().numpy()}, weights={self.weights.detach().cpu().numpy()}"
        )

        return self.weights

    def update(self, losses=None, gradients=None):
        """Update weights based on chosen strategy.

        Args:
            losses (torch.Tensor | None): Loss component values (required for ``rbw``).
            gradients (torch.Tensor | None): Gradient norm per component (required for ``lrw``).

        Returns:
            torch.Tensor: Updated loss weights.

        Raises:
            ValueError: If the input the strategy needs is missing, or the strategy is unknown.
        """
        if self.strategy == "lrw" and gradients is not None:
            return self.update_weights_lrw(gradients)
        elif self.strategy == "rbw" and losses is not None:
            return self.update_weights_rbw(losses)
        else:
            raise ValueError(f"Invalid combination of strategy ({self.strategy}) and inputs")
