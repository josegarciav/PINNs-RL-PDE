"""Neural network with Fourier feature encoding for better fitting of high-frequency functions."""

from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.jit import script

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig


@script
def _fourier_feature_transform(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Optimized Fourier feature transformation using TorchScript."""
    x_proj = x @ B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatures(nn.Module):
    """Optimized Fourier feature embedding with TorchScript support."""

    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        scale: float = 10.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize Fourier features.

        Args:
            input_dim: Input dimension
            mapping_size: Size of the feature mapping
            scale: Scale factor for the random Fourier features
            device: Device to place the features on
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Initialize random Fourier features with better initialization
        self.register_buffer(
            "B", torch.randn(input_dim, mapping_size, device=self.device) * scale
        )

        # Pre-compute output dimension
        self.output_dim = mapping_size * 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Fourier feature mapping.

        Args:
            x: Input tensor
            
        Returns:
            Mapped features
        """
        if self.B.device != x.device:
            self.B = self.B.to(x.device)
        return _fourier_feature_transform(x, self.B)


class FourierNetwork(BaseNetwork):
    """Neural network with Fourier feature embedding."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - mapping_size: Size of the Fourier feature mapping
                - hidden_dims: List of hidden layer dimensions
                - output_dim: Output dimension
                - activation: Activation function name
                - scale: Scale of the random Fourier features
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.mapping_size = config.get("mapping_size", 32)  # Default mapping size
        self.hidden_dims = config.get(
            "hidden_dims", [128, 128]
        )  # Default hidden dimensions
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
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, self.output_dim))

        # Move all layers to device
        self.to(self.device)

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

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