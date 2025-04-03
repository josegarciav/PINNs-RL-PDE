"""Autoencoder neural network architecture for dimensionality reduction and feature learning."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig


class AutoEncoder(BaseNetwork):
    """Autoencoder neural network."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - latent_dim: Latent space dimension
                - hidden_dims: List of hidden layer dimensions
                - activation: Activation function name
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.latent_dim = config.get("latent_dim", 16)
        self.hidden_dims = config.get("hidden_dims", [32, 64])
        activation_name = config.get("activation", "relu")
        self.activation_fn = self._get_activation_module(activation_name)
        self.dropout_rate = config.get("dropout", 0.1)
        self.use_layer_norm = config.get("layer_norm", True)

        # Build encoder
        self.encoder = nn.ModuleList()
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.encoder.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_layer_norm:
                self.encoder.append(nn.LayerNorm(hidden_dim))
            self.encoder.append(self.activation_fn)
            self.encoder.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        self.encoder.append(nn.Linear(prev_dim, self.latent_dim))

        # Build decoder
        self.decoder = nn.ModuleList()
        prev_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            self.decoder.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_layer_norm:
                self.decoder.append(nn.LayerNorm(hidden_dim))
            self.decoder.append(self.activation_fn)
            self.decoder.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        self.decoder.append(nn.Linear(prev_dim, self.input_dim))

    def encode(self, x: InputType) -> torch.Tensor:
        """
        Encode input to latent space.

        Args:
            x: Input tensor or array

        Returns:
            Latent space representation
        """
        x = self._prepare_input(x)
        for layer in self.encoder:
            x = layer(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent space to output.

        Args:
            z: Latent space tensor

        Returns:
            Decoded output
        """
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Reconstructed output
        """
        z = self.encode(x)
        return self.decode(z)
