"""Neural network architectures for Physics-Informed Neural Networks (PINNs)."""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import sys
import os

from src.config import Config

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig
from .feedforward import FeedForwardNetwork
from .resnet import ResNet, ResNetBlock
from .siren import SIREN, SIRENLayer
from .fourier import FourierNetwork, FourierFeatures
from .autoencoder import AutoEncoder

# Import optional architectures if available
try:
    from .attention import AttentionNetwork, SelfAttention

    __all__ = [
        "BaseNetwork",
        "InputType",
        "OutputType",
        "NetworkConfig",
        "FeedForwardNetwork",
        "ResNet",
        "ResNetBlock",
        "SIREN",
        "SIRENLayer",
        "FourierNetwork",
        "FourierFeatures",
        "AutoEncoder",
        "AttentionNetwork",
        "SelfAttention",
        "PINNModel",
    ]
except ImportError:
    __all__ = [
        "BaseNetwork",
        "InputType",
        "OutputType",
        "NetworkConfig",
        "FeedForwardNetwork",
        "ResNet",
        "ResNetBlock",
        "SIREN",
        "SIRENLayer",
        "FourierNetwork",
        "FourierFeatures",
        "AutoEncoder",
        "PINNModel",
    ]


class PINNModel(BaseNetwork):
    """
    Physics-Informed Neural Network model factory.

    This class provides a unified interface to create different neural network
    architectures suitable for Physics-Informed Neural Networks.
    """

    def __init__(self, config: Config, **kwargs):
        """
        Initialize the PINN model.

        Args:
            config: Configuration object with model settings
            **kwargs: Additional architecture-specific parameters
        """
        # Store configuration
        self.config = config

        # Create base NetworkConfig
        super().__init__(config.model)

        # Store architecture type
        self.architecture = config.model.architecture

        # Create model based on architecture
        if self.architecture == "fourier":
            self.model = FourierNetwork(config.model)
        elif self.architecture == "resnet":
            self.model = ResNet(config.model)
        elif self.architecture == "siren":
            self.model = SIREN(config.model)
        elif self.architecture == "attention":
            self.model = AttentionNetwork(config.model)
        elif self.architecture == "autoencoder":
            self.model = AutoEncoder(config.model)
            # Additional output projection for AutoEncoder to ensure correct output dimension
            self.output_proj = FeedForwardNetwork(config.model)
        else:  # Default to feedforward
            self.model = FeedForwardNetwork(config.model)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x: Input tensor of shape (batch_size, input_dim)

        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        if self.architecture == "autoencoder" and hasattr(self, "output_proj"):
            # For autoencoder, we need to project from input_dim to output_dim
            return self.output_proj(self.model(x))
        return self.model(x)


def create_network(config: Config) -> nn.Module:
    """
    Create a neural network based on the configuration.

    Args:
        config: Configuration object

    Returns:
        Neural network module
    """
    architecture = config.model.architecture

    if architecture == "feedforward":
        from .feedforward import FeedForwardNetwork

        return FeedForwardNetwork(config)
    elif architecture == "fourier":
        from .fourier import FourierNetwork

        return FourierNetwork(config)
    elif architecture == "siren":
        from .siren import SirenNetwork

        return SirenNetwork(config)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
