"""Encoder-decoder MLP with a latent bottleneck, used as a PINN function approximator."""

import torch
import torch.nn as nn

from .base_network import BaseNetwork, InputType, NetworkConfig, OutputType


class AutoEncoder(BaseNetwork):
    """Encoder-decoder MLP: ``input -> hidden_dims -> latent -> reversed(hidden_dims) -> output``.

    It maps coordinates to the solution; it does not reconstruct its input.
    """

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - latent_dim: Latent space dimension
                - hidden_dims: Encoder hidden widths; the decoder uses them reversed
                - output_dim: Output dimension (default 1)
                - activation: Activation function name
                - dropout: Dropout rate (default 0.0)
                - layer_norm: LayerNorm after each hidden linear layer (default True)
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.latent_dim = config.get("latent_dim", 16)
        self.hidden_dims = config.get("hidden_dims", [32, 64])
        activation_name = config.get("activation", "relu")
        self.dropout_rate = config.get("dropout", 0.0)
        self.use_layer_norm = config.get("layer_norm", True)

        self.output_dim = config.get("output_dim", 1)

        # Encoder maps input -> latent; the decoder maps latent -> output_dim
        # for PINN function approximation.
        self.encoder = nn.ModuleList()
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.encoder.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_layer_norm:
                self.encoder.append(nn.LayerNorm(hidden_dim))
            self.encoder.append(self._get_activation_module(activation_name))
            if self.dropout_rate > 0.0:
                self.encoder.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        self.encoder.append(nn.Linear(prev_dim, self.latent_dim))

        # Decoder: latent -> output_dim (solution value)
        self.decoder = nn.ModuleList()
        prev_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            self.decoder.append(nn.Linear(prev_dim, hidden_dim))
            if self.use_layer_norm:
                self.decoder.append(nn.LayerNorm(hidden_dim))
            self.decoder.append(self._get_activation_module(activation_name))
            if self.dropout_rate > 0.0:
                self.decoder.append(nn.Dropout(self.dropout_rate))
            prev_dim = hidden_dim
        self.decoder.append(nn.Linear(prev_dim, self.output_dim))

    def encode(self, x: InputType) -> torch.Tensor:
        """Encode input to latent space.

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
        """Decode latent space to output.

        Args:
            z: Latent space tensor

        Returns:
            Decoded output
        """
        for layer in self.decoder:
            z = layer(z)
        return z

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Predicted solution, shape ``(batch, output_dim)``
        """
        z = self.encode(x)
        return self.decode(z)
