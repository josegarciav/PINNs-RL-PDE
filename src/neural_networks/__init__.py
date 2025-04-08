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

    def __init__(self, config: Config, device=None, **kwargs):
        """
        Initialize the PINN model.

        Args:
            config: Configuration object with model settings
            device: Optional device to override config device
            **kwargs: Additional architecture-specific parameters
        """
        # Store configuration
        self.config = config
        
        # Use provided device or get from config
        self.device = device if device is not None else config.device
        
        # Override device in config with the determined device
        config_with_device = config.model
        config_with_device.device = self.device
        
        # Create base NetworkConfig
        super().__init__(config_with_device)

        # Store architecture type
        self.architecture = config.model.architecture
        self.architecture_name = (
            config.model.architecture
        )  # More accessible name for metadata

        # Create model based on architecture
        if self.architecture == "fourier":
            self.model = FourierNetwork(config_with_device)
        elif self.architecture == "resnet":
            # Create a specific configuration dictionary for ResNet
            resnet_config = {
                "input_dim": config.model.input_dim,
                "hidden_dim": config.model.hidden_dim,
                "output_dim": config.model.output_dim,
                "activation": config.model.activation,
                "dropout": config.model.dropout,
                "device": self.device,
            }
            
            # Make sure num_blocks is defined, using num_layers as fallback
            if hasattr(config.model, "num_blocks") and config.model.num_blocks is not None:
                resnet_config["num_blocks"] = config.model.num_blocks
            else:
                resnet_config["num_blocks"] = config.model.num_layers
                
            # Add hidden_dims if defined
            if hasattr(config.model, "hidden_dims") and config.model.hidden_dims is not None:
                resnet_config["hidden_dims"] = config.model.hidden_dims
                
            self.model = ResNet(resnet_config)
        elif self.architecture == "siren":
            self.model = SIREN(config_with_device)
        elif self.architecture == "attention":
            self.model = AttentionNetwork(config_with_device)
        elif self.architecture == "autoencoder":
            self.model = AutoEncoder(config_with_device)
            # Additional output projection for AutoEncoder to ensure correct output dimension
            self.output_proj = FeedForwardNetwork(config_with_device)
        else:  # Default to feedforward
            self.model = FeedForwardNetwork(config_with_device)
        
        # Explicitly move the model to the specified device
        self.model = self.model.to(self.device)
        self.to(self.device)

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
