"""Neural network architectures for Physics-Informed Neural Networks (PINNs)."""

import torch
from typing import Dict, Any, Optional, Union, List
import sys
import os

# Add project root to Python path to import config.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config import load_config, get_architecture_config, merge_configs

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
        'BaseNetwork', 'InputType', 'OutputType', 'NetworkConfig',
        'FeedForwardNetwork', 'ResNet', 'ResNetBlock', 'SIREN', 'SIRENLayer',
        'FourierNetwork', 'FourierFeatures', 'AutoEncoder',
        'AttentionNetwork', 'SelfAttention', 'PINNModel'
    ]
except ImportError:
    __all__ = [
        'BaseNetwork', 'InputType', 'OutputType', 'NetworkConfig',
        'FeedForwardNetwork', 'ResNet', 'ResNetBlock', 'SIREN', 'SIRENLayer',
        'FourierNetwork', 'FourierFeatures', 'AutoEncoder', 'PINNModel'
    ]


class PINNModel(BaseNetwork):
    """
    Physics-Informed Neural Network model factory.
    
    This class provides a unified interface to create different neural network
    architectures suitable for Physics-Informed Neural Networks.
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        activation: Optional[str] = None,
        fourier_features: Optional[bool] = None,
        fourier_scale: Optional[float] = None,
        dropout: Optional[float] = None,
        layer_norm: Optional[bool] = None,
        architecture: Optional[str] = None,
        device=None,
        config_path: Optional[str] = None,
        pde_type: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the PINN model.

        Args:
            input_dim: Input dimension (x, t)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (u)
            num_layers: Number of layers
            activation: Activation function name ('tanh', 'relu', 'gelu')
            fourier_features: Whether to use Fourier features
            fourier_scale: Scale factor for Fourier features
            dropout: Dropout rate
            layer_norm: Whether to use layer normalization
            architecture: Neural network architecture to use
            device: Device to place the model on
            config_path: Path to the configuration file
            pde_type: Type of PDE to use for configuration (e.g., 'heat', 'wave')
            **kwargs: Additional architecture-specific parameters
        """
        # Load default configuration
        self.full_config = load_config(config_path)
        
        # If PDE type is specified, use its configuration
        if pde_type and pde_type in self.full_config.get("pde_configs", {}):
            pde_config = self.full_config["pde_configs"][pde_type]
            # Override architecture and dimensions from PDE config if not provided
            if architecture is None:
                architecture = pde_config.get("architecture")
            if input_dim is None:
                input_dim = pde_config.get("input_dim")
            if output_dim is None:
                output_dim = pde_config.get("output_dim")
        
        # Get default architecture if not specified by PDE or args
        if architecture is None:
            architecture = self.full_config.get("model", {}).get("architecture", "fourier")
        
        # Get architecture-specific configuration
        arch_config = get_architecture_config(self.full_config, architecture)
        
        # Set default dimensions from model section if not specified
        if input_dim is None:
            input_dim = self.full_config.get("model", {}).get("input_dim", 2)
        if output_dim is None:
            output_dim = self.full_config.get("model", {}).get("output_dim", 1)
        
        # Override with constructor arguments
        override_config = {
            "input_dim": input_dim,
            "output_dim": output_dim
        }
        
        if hidden_dim is not None:
            override_config["hidden_dim"] = hidden_dim
        if num_layers is not None:
            override_config["num_layers"] = num_layers
        if activation is not None:
            override_config["activation"] = activation
        if fourier_features is not None:
            override_config["fourier_features"] = fourier_features
        if fourier_scale is not None:
            override_config["fourier_scale"] = fourier_scale
        if dropout is not None:
            override_config["dropout"] = dropout
        if layer_norm is not None:
            override_config["layer_norm"] = layer_norm
        if device is not None:
            override_config["device"] = device
            
        # Add any additional kwargs
        override_config.update(kwargs)
        
        # Merge configurations
        config = merge_configs(arch_config, override_config)
        
        # Create base NetworkConfig
        super().__init__(config)
        
        # Store architecture type
        self.architecture = architecture
        
        # Create model based on architecture
        if architecture == "fourier":
            self.model = FourierNetwork(config)
        elif architecture == "resnet":
            self.model = ResNet(config)
        elif architecture == "siren":
            self.model = SIREN(config)
        elif architecture == "attention":
            try:
                self.model = AttentionNetwork(config)
            except NameError:
                raise ImportError("AttentionNetwork architecture is not available. Please install the required dependencies.")
        elif architecture == "autoencoder":
            self.model = AutoEncoder(config)
            # Additional output projection for AutoEncoder to ensure correct output dimension
            self.output_proj = FeedForwardNetwork({
                "input_dim": config["input_dim"],
                "hidden_dims": [config.get("hidden_dim", 64)], 
                "output_dim": config["output_dim"],
                "activation": config.get("activation", "relu"),
                "dropout": config.get("dropout", 0.1),
                "device": config.get("device")
            })
        else:  # Default to feedforward
            self.model = FeedForwardNetwork(config)

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