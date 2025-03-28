"""SIREN architecture with sine activation functions for better representation of signals and implicit functions."""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig


class SIRENLayer(nn.Module):
    """SIREN layer with sinusoidal activation."""

    def __init__(
        self, in_features: int, out_features: int, omega_0: float = 30.0
    ) -> None:
        """
        Initialize the layer.

        Args:
            in_features: Input dimension
            out_features: Output dimension
            omega_0: Frequency parameter
        """
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize the weights using SIREN initialization."""
        with torch.no_grad():
            self.linear.weight.uniform_(
                -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                np.sqrt(6 / self.linear.in_features) / self.omega_0,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(BaseNetwork):
    """Sinusoidal Representation Network."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - hidden_dims: List of hidden layer dimensions
                - output_dim: Output dimension
                - omega_0: Frequency parameter
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.omega_0 = config.get("omega_0", 30.0)

        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(SIRENLayer(prev_dim, hidden_dim, omega_0=self.omega_0))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, self.output_dim))

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x) 