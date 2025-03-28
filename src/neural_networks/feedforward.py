"""Feed-forward neural network architecture."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig


class FeedForwardNetwork(BaseNetwork):
    """Simple feed-forward neural network with configurable hidden layers."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the feed-forward network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - hidden_dims: List of hidden layer dimensions
                - output_dim: Output dimension
                - activation: Activation function name ('relu', 'tanh', etc.)
                - dropout: Dropout rate
                - layer_norm: Whether to use layer normalization
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.dropout_rate = config.get("dropout", 0.1)
        self.use_layer_norm = config.get("layer_norm", True)
        
        # Get activation function
        activation_name = config.get("activation", "relu")
        self.activation = self._get_activation_module(activation_name)
        
        # Build layers
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(self.activation)
            layers.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
            
        # Output layer
        layers.append(nn.Linear(prev_dim, self.output_dim))
        
        self.layers = nn.Sequential(*layers)

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

        Args:
            x: Input tensor or array
            
        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        return self.layers(x) 