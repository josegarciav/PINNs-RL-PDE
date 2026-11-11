"""Base class for neural network architectures."""

from typing import Any

import numpy as np
import torch
import torch.nn as nn

# Type aliases for better readability
InputType = torch.Tensor | np.ndarray | list
OutputType = torch.Tensor | np.ndarray | list
NetworkConfig = dict[str, Any]


class ExactLayerNorm(nn.LayerNorm):
    """Layer normalization over the last dimension, written out in elementary operations.

    ``torch.nn.LayerNorm`` returns the right values and the right first and
    second derivatives, but its third-order derivative is wrong (PyTorch
    2.10, checked against finite differences): the weight gradient of a loss
    that contains second derivatives of the network with respect to its
    inputs, which is every PINN residual with a ``u_xx``, comes out 2 to 3 %
    off in norm on a 4 x 64 network. Composed of elementary operations the
    same function differentiates correctly to any order. The parameters and
    their names are those of ``nn.LayerNorm``, so checkpoints load either way.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``weight * (x - mean) / sqrt(var + eps) + bias`` over the last dimension."""
        if len(self.normalized_shape) != 1:
            return super().forward(x)
        c = x - x.mean(dim=-1, keepdim=True)
        y = c * torch.rsqrt(c.square().mean(dim=-1, keepdim=True) + self.eps)
        if not self.elementwise_affine:
            return y
        if self.bias is None:
            return y * self.weight
        return torch.addcmul(self.bias, self.weight, y)  # as fast as the built-in under autograd


class BaseNetwork(nn.Module):
    """Base class for all neural network architectures."""

    def __init__(self, config: dict) -> None:
        """Initialize the base network.

        Args:
            config: Configuration mapping with ``.get`` (a dict or a ``ModelConfig``)
        """
        super().__init__()
        self.config: Any = config
        self.device = config.get("device", torch.device("cpu"))
        self.to(self.device)

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        raise NotImplementedError

    def _prepare_input(self, x: InputType) -> torch.Tensor:
        """Convert input to tensor and move to correct device.

        Args:
            x: Input tensor or array

        Returns:
            Tensor on correct device
        """
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)

        if x.device != self.device:
            x = x.to(self.device)

        return x

    def save_state(self, path: str) -> None:
        """Save model state with architecture information.

        Args:
            path: Path to save the model
        """
        state = {"model_state_dict": self.state_dict(), "config": self.config}
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load model state with architecture information.

        Args:
            path: Path to load the model from
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_state_dict(state["model_state_dict"])
        self.config = state["config"]

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")
