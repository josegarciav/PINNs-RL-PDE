"""Neural network with Fourier feature encoding for better fitting of high-frequency functions."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork, InputType, NetworkConfig, OutputType


def _fourier_feature_transform(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Fourier feature map ``[sin(x B), cos(x B)]``."""
    x_proj = x @ B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatures(nn.Module):
    """Random Fourier feature map ``[sin(xB), cos(xB)]`` with fixed ``B ~ N(0, scale^2)``.

    ``B`` has shape ``(input_dim, mapping_size)`` and is a buffer, not trained;
    there is no ``2π`` factor.
    """

    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        scale: float = 10.0,
        device: torch.device | None = None,
    ):
        """Initialize Fourier features.

        Args:
            input_dim: Input dimension
            mapping_size: Number of frequencies (output has ``2 * mapping_size`` features)
            scale: Standard deviation of the entries of ``B``
            device: Device to place the features on (default: MPS if available, else CPU)
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        # Fixed random frequencies B ~ N(0, scale^2)
        self.register_buffer("B", torch.randn(input_dim, mapping_size, device=self.device) * scale)

        # Pre-compute output dimension
        self.output_dim = mapping_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with Fourier feature mapping.

        Args:
            x: Input tensor

        Returns:
            ``[sin(xB), cos(xB)]``, shape ``(..., 2 * mapping_size)``
        """
        if self.B.device != x.device:
            self.B = self.B.to(x.device)
        return _fourier_feature_transform(x, self.B)


class FourierNetwork(BaseNetwork):
    """MLP on random Fourier features: ``[sin(xB), cos(xB)]`` -> hidden layers -> output."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - mapping_size: Size of the Fourier feature mapping (default 32)
                - hidden_dims: Width of each hidden layer (one layer per entry,
                  like the feedforward network); falls back to
                  ``[hidden_dim] * num_layers`` (``num_layers`` default 4)
                - output_dim: Output dimension
                - activation: Activation function name (default ``relu``)
                - scale: Standard deviation of ``B`` (default 10.0)
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.mapping_size = config.get("mapping_size", 32)  # Default mapping size
        self.hidden_dims = list(
            config.get("hidden_dims") or [config["hidden_dim"]] * config.get("num_layers", 4)
        )
        self.output_dim = config["output_dim"]
        activation_name = config.get("activation", "relu")
        self.activation_fn = self._get_activation_module(activation_name)
        self.scale = config.get("scale", 10.0)

        # Build layers
        self.fourier = FourierFeatures(
            self.input_dim, self.mapping_size, self.scale, device=self.device
        )
        self.layers = nn.ModuleList()
        prev_dim = 2 * self.mapping_size  # Sine and cosine features

        # One hidden layer per width, then the output layer
        for width in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, width))
            prev_dim = width

        # Add output layer
        self.layers.append(nn.Linear(prev_dim, self.output_dim))

        # Move all layers to device
        self.to(self.device)

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        x = self.fourier(x)
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)
