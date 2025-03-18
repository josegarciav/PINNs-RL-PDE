import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Dict, Tuple, Any, Union
import numpy as np

class ResidualBlock(nn.Module):
    """Residual block for better gradient flow."""
    
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
    """Fourier feature embedding for better approximation of periodic functions."""
    
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
        
        # Initialize random Fourier features
        self.register_buffer('B', torch.randn(input_dim, mapping_size, device=self.device) * scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with Fourier feature mapping.
        
        :param x: Input tensor
        :return: Mapped features
        """
        x_proj = x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class PINNModel(nn.Module):
    """Physics-Informed Neural Network model."""
    
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
        
        # Setup activation function
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        # Fourier feature embedding
        if fourier_features:
            self.fourier = FourierFeatures(input_dim, hidden_dim, fourier_scale, device=self.device)
            input_dim = hidden_dim * 2  # Sine and cosine features
        
        # Build network
        self.layers = nn.ModuleList()
        
        # Input layer
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
        
        # Initialize weights
        self._init_weights()
        
        # Move model to device
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize network weights using Xavier/Glorot initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        :param x: Input tensor
        :return: Output tensor
        """
        # Apply Fourier features if enabled
        if self.fourier_features:
            x = self.fourier(x)
        
        # Forward through layers
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def get_intermediate_activations(self) -> torch.Tensor:
        """
        Get intermediate activations for RL state representation.
        
        :return: Tensor of intermediate activations
        """
        activations = []
        x = self.fourier(self.last_input) if self.fourier_features else self.last_input
        
        for layer in self.layers[:-1]:  # Exclude output layer
            x = layer(x)
            activations.append(x)
        
        return torch.cat(activations, dim=-1)
    
    def save_state(self, path: str):
        """
        Save model state.
        
        :param path: Path to save the model
        """
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
        torch.save(state, path)
    
    @classmethod
    def load_state(cls, path: str, device: Optional[torch.device] = None) -> 'PINNModel':
        """
        Load model state.
        
        :param path: Path to load the model from
        :param device: Device to load the model to
        :return: Loaded model instance
        """
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
        """
        Count the number of trainable parameters.
        
        :return: Number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model architecture.

        :return: Dictionary containing model summary
        """
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
            'device': str(self.device)
        }

# Create model
model = PINNModel(
    input_dim=2,  # (x, t)
    hidden_dim=128,
    output_dim=1,  # u(x,t)
    num_layers=4,
    activation='tanh',
    fourier_features=True,
    fourier_scale=2.0,
    dropout=0.1,
    layer_norm=True
)

# Get model summary
summary = model.get_model_summary()
print(f"Model has {summary['num_parameters']} parameters")

# Save model
model.save_state('model.pth')

# Load model
loaded_model = PINNModel.load_state('model.pth')
