import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Tuple, Any, Union
import numpy as np
from torch.jit import script

@script
def _fourier_feature_transform(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Optimized Fourier feature transformation using TorchScript."""
    x_proj = x @ B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow with optimized implementation."""
    
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize residual block.
        
        :param in_dim: Input dimension
        :param hidden_dim: Hidden dimension
        :param dropout: Dropout rate
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.
        
        :param x: Input tensor
        :return: Output tensor
        """
        return x + self.layers(x)

class FourierFeatures(nn.Module):
    """Optimized Fourier feature embedding with TorchScript support."""
    
    def __init__(self, input_dim: int, mapping_size: int, scale: float = 10.0, device: Optional[torch.device] = None):
        """
        Initialize Fourier features.
        
        :param input_dim: Input dimension
        :param mapping_size: Size of the feature mapping
        :param scale: Scale factor for the random Fourier features
        :param device: Device to place the features on
        """
        super().__init__()
        self.input_dim = input_dim
        self.mapping_size = mapping_size
        self.scale = scale
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Initialize random Fourier features with better initialization
        self.register_buffer('B', torch.randn(input_dim, mapping_size, device=self.device) * scale)
        
        # Pre-compute output dimension
        self.output_dim = mapping_size * 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Fourier feature mapping.
        
        :param x: Input tensor
        :return: Mapped features
        """
        if self.B.device != x.device:
            self.B = self.B.to(x.device)
        return _fourier_feature_transform(x, self.B)

class PINNModel(nn.Module):
    """Optimized Physics-Informed Neural Network model with improved performance."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = 'tanh',
        fourier_features: bool = True,
        fourier_scale: float = 2.0,
        dropout: float = 0.1,
        layer_norm: bool = True,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the PINN model.
        
        :param input_dim: Input dimension (x, t)
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Output dimension (u)
        :param num_layers: Number of layers
        :param activation: Activation function ('tanh', 'relu', 'gelu')
        :param fourier_features: Whether to use Fourier features
        :param fourier_scale: Scale factor for Fourier features
        :param dropout: Dropout rate
        :param layer_norm: Whether to use layer normalization
        :param device: Device to place the model on
        """
        super().__init__()
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Store architecture parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.fourier_features = fourier_features
        self.fourier_scale = fourier_scale
        self.dropout = dropout
        self.layer_norm = layer_norm
        
        # Optimized activation function selection
        self.act = {
            'tanh': nn.Tanh(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }.get(activation, nn.Tanh())
        
        # Fourier feature embedding with pre-computed dimensions
        if fourier_features:
            self.fourier = FourierFeatures(input_dim, hidden_dim, fourier_scale, device=self.device)
            input_dim = self.fourier.output_dim
        
        # Build network with optimized layer structure
        self.layers = nn.ModuleList()
        
        # Input layer with optimized initialization
        self.layers.append(nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if layer_norm else nn.Identity(),
            self.act,
            nn.Dropout(dropout)
        ))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):
            self.layers.append(ResidualBlock(hidden_dim, hidden_dim, dropout))
        
        # Output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        
        # Initialize weights with improved initialization
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
        
        # Enable TorchScript optimization
        self.traced = False
    
    def _init_weights(self):
        """Improved weight initialization using Kaiming initialization for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='tanh')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized forward pass with optional TorchScript tracing."""
        if not self.traced:
            self.traced = True
            self.forward = torch.jit.trace(self.forward, torch.randn(1, self.input_dim, device=self.device))
        
        if self.fourier_features:
            x = self.fourier(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_intermediate_activations(self) -> torch.Tensor:
        """Optimized intermediate activations computation."""
        activations = []
        x = self.fourier(self.last_input) if self.fourier_features else self.last_input
        
        for layer in self.layers[:-1]:
            x = layer(x)
            activations.append(x)
        
        return torch.cat(activations, dim=-1)
    
    def save_state(self, path: str):
        """Optimized model state saving with compression."""
        state = {
            'model_state_dict': self.state_dict(),
            'architecture': {
                'input_dim': self.input_dim,
                'hidden_dim': self.hidden_dim,
                'output_dim': self.output_dim,
                'num_layers': self.num_layers,
                'activation': self.activation,
                'fourier_features': self.fourier_features,
                'fourier_scale': self.fourier_scale,
                'dropout': self.dropout,
                'layer_norm': self.layer_norm
            }
        }
        torch.save(state, path, _use_new_zipfile_serialization=True)
    
    @classmethod
    def load_state(cls, path: str, device: Optional[torch.device] = None) -> 'PINNModel':
        """Optimized model state loading with device handling."""
        state = torch.load(path, map_location=device)
        architecture = state['architecture']
        
        model = cls(
            input_dim=architecture['input_dim'],
            hidden_dim=architecture['hidden_dim'],
            output_dim=architecture['output_dim'],
            num_layers=architecture['num_layers'],
            activation=architecture['activation'],
            fourier_features=architecture['fourier_features'],
            fourier_scale=architecture['fourier_scale'],
            dropout=architecture['dropout'],
            layer_norm=architecture['layer_norm'],
            device=device
        )
        
        model.load_state_dict(state['model_state_dict'])
        return model

    def count_parameters(self) -> int:
        """Optimized parameter counting."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        """Enhanced model summary with memory usage."""
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'activation': self.activation,
            'fourier_features': self.fourier_features,
            'fourier_scale': self.fourier_scale,
            'dropout': self.dropout,
            'layer_norm': self.layer_norm,
            'num_parameters': self.count_parameters(),
            'device': str(self.device),
            'memory_usage': f"{sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2:.2f} MB"
        }

# Example usage with optimized model creation
if __name__ == "__main__":
    model = PINNModel(
        input_dim=2,
        hidden_dim=128,
        output_dim=1,
        num_layers=4,
        activation='tanh',
        fourier_features=True,
        fourier_scale=2.0,
        dropout=0.1,
        layer_norm=True
    )
    
    summary = model.get_model_summary()
    print(f"Model Summary:\n{summary}")
