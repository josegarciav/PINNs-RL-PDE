"""ResNet architecture for improved gradient flow in deep networks."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig


class ResNetBlock(nn.Module):
    """Base component for ResNet architecture - a single residual block with skip connection."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        activation: str = "relu",
        dropout: float = 0.1,
    ):
        """
        Initialize a single residual block component.

        Args:
            in_dim: Input dimension
            hidden_dim: Hidden dimension
            activation: Activation function name
            dropout: Dropout rate
        """
        super().__init__()
        
        # Get activation function
        if activation == "relu":
            self.activation_fn = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation_fn = nn.LeakyReLU()
        elif activation == "tanh":
            self.activation_fn = nn.Tanh()
        elif activation == "sigmoid":
            self.activation_fn = nn.Sigmoid()
        elif activation == "gelu":
            self.activation_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Build the residual block
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation_fn,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.activation_fn(x + self.layers(x))


class ResNet(BaseNetwork):
    """Residual Neural Network architecture built from multiple ResNetBlock components."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the ResNet architecture.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - hidden_dim: Hidden layer dimension
                - num_blocks: Number of ResNetBlock components
                - output_dim: Output dimension
                - activation: Activation function name
                - dropout: Dropout rate
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.num_blocks = config["num_blocks"]
        self.output_dim = config["output_dim"]
        activation_name = config.get("activation", "relu")
        self.activation_fn = self._get_activation_module(activation_name)
        self.dropout = config.get("dropout", 0.1)

        # Build the ResNet architecture
        self.input_layer = nn.Linear(self.input_dim, self.hidden_dim)
        # Stack multiple ResNetBlock components
        self.blocks = nn.ModuleList(
            [
                ResNetBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    config.get("activation", "relu"),
                    self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        x = self.activation_fn(self.input_layer(x))
        for block in self.blocks:
            x = block(x)
        return self.output_layer(x) 