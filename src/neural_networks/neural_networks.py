"""Neural network architectures for PINNs."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.jit import script

from ..utils.types import ArrayLike

# Type aliases for better readability
InputType = Union[Tensor, ArrayLike]
OutputType = Union[Tensor, ArrayLike]
LayerConfig = Dict[str, Any]
NetworkConfig = Dict[str, Any]


@script
def _fourier_feature_transform(x: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Optimized Fourier feature transformation using TorchScript."""
    x_proj = x @ B
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class FourierFeatures(nn.Module):
    """Optimized Fourier feature embedding with TorchScript support."""

    def __init__(
        self,
        input_dim: int,
        mapping_size: int,
        scale: float = 10.0,
        device: Optional[torch.device] = None,
    ):
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
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )

        # Initialize random Fourier features with better initialization
        self.register_buffer(
            "B", torch.randn(input_dim, mapping_size, device=self.device) * scale
        )

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


class BaseNetwork(nn.Module):
    """Base class for all neural network architectures."""

    def __init__(self, config: Dict) -> None:
        """
        Initialize the base network.

        Args:
            config: Configuration dictionary
        """
        super().__init__()
        self.config = config
        self.device = config.get("device", torch.device("cpu"))
        self.to(self.device)

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        raise NotImplementedError

    def _prepare_input(self, x: InputType) -> torch.Tensor:
        """Convert input to tensor and move to correct device.

        Args:
            x: Input tensor or array

        Returns:
            Tensor on correct device
        """
        if isinstance(x, (np.ndarray, list)):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        elif not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        return x

    def save_state(self, path: str) -> None:
        """Save model state with architecture information.

        Args:
            path: Path to save the model
        """
        state = {"model_state_dict": self.state_dict(), "config": self.config}
        torch.save(state, path)

    def load_state(self, path: str) -> None:
        """Load model state with architecture information.

        Args:
            path: Path to load the model from
        """
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state["model_state_dict"])
        self.config = state["config"]

    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_model_summary(self) -> Dict:
        """Get a summary of the model architecture."""
        return {
            "num_parameters": self.count_parameters(),
            "device": str(self.device),
            "memory_usage": f"{sum(p.numel() * p.element_size() for p in self.parameters()) / 1024**2:.2f} MB",
        }


class FeedForwardNetwork(BaseNetwork):
    """Simple feed-forward neural network with configurable hidden layers."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the feed-forward network.

        :param config: Network configuration dictionary containing:
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

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

        :param x: Input tensor or array
        :return: Output tensor or array
        """
        x = self._prepare_input(x)
        return self.layers(x)


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

        :param in_dim: Input dimension
        :param hidden_dim: Hidden dimension
        :param activation: Activation function name
        :param dropout: Dropout rate
        """
        super().__init__()
        self.activation_fn = self._get_activation_module(activation)
        
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

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.activation_fn(x + self.layers(x))


class ResNet(BaseNetwork):
    """Residual Neural Network architecture built from multiple ResNetBlock components."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the ResNet architecture.

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

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

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


class FourierNetwork(BaseNetwork):
    """Neural network with Fourier feature embedding."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - mapping_size: Size of the Fourier feature mapping
                - hidden_dims: List of hidden layer dimensions
                - output_dim: Output dimension
                - activation: Activation function name
                - scale: Scale of the random Fourier features
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.mapping_size = config.get("mapping_size", 32)  # Default mapping size
        self.hidden_dims = config.get(
            "hidden_dims", [128, 128]
        )  # Default hidden dimensions
        self.output_dim = config["output_dim"]
        activation_name = config.get("activation", "relu")
        self.activation_fn = self._get_activation_module(activation_name)
        self.scale = config.get("scale", 10.0)

        # Build layers
        self.fourier = FourierFeatures(
            self.input_dim, self.mapping_size, self.scale, device=self.device
        )
        self.layers = nn.ModuleList()
        prev_dim = 2 * self.mapping_size  # Sine and cosine features
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, self.output_dim))

        # Move all layers to device
        self.to(self.device)

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        x = self.fourier(x)
        for layer in self.layers[:-1]:
            x = self.activation_fn(layer(x))
        return self.layers[-1](x)


class SIREN(BaseNetwork):
    """Sinusoidal Representation Network."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

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
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        for layer in self.layers[:-1]:
            x = layer(x)
        return self.layers[-1](x)


class SIRENLayer(nn.Module):
    """SIREN layer with sinusoidal activation."""

    def __init__(
        self, in_features: int, out_features: int, omega_0: float = 30.0
    ) -> None:
        """Initialize the layer.

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
        """Forward pass of the layer.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return torch.sin(self.omega_0 * self.linear(x))


class SelfAttention(nn.Module):
    """Self-attention layer."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        """Initialize the attention layer.

        Args:
            dim: Input dimension
            heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        assert self.head_dim * heads == dim, "Dimension must be divisible by heads"

        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the layer.

        Args:
            x: Input tensor of shape (batch_size, dim)

        Returns:
            Output tensor of shape (batch_size, dim)
        """
        residual = x
        batch_size = x.shape[0]

        # Use unsqueeze to add sequence length dimension of 1
        # Shape: (batch_size, 1, dim)
        x = x.unsqueeze(1)
        
        # Linear projections
        q = self.query(x).view(batch_size, 1, self.heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch_size, 1, self.heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch_size, 1, self.heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, self.dim)
        out = self.proj(out).squeeze(1)  # Remove sequence dimension
        
        # Add residual connection and apply layer norm
        out = self.layer_norm(out + residual)
        
        return out


class FeedForwardBlock(nn.Module):
    """Feed-forward block for transformer architecture."""

    def __init__(self, dim: int, expansion: int = 4, dropout: float = 0.1) -> None:
        """Initialize the feed-forward block.

        Args:
            dim: Input dimension
            expansion: Dimension expansion factor
            dropout: Dropout rate
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion, dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.layer_norm(x + self.net(x))


class AttentionNetwork(BaseNetwork):
    """Neural network with self-attention layers."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - hidden_dim: Hidden dimension
                - output_dim: Output dimension
                - num_layers: Number of attention layers
                - num_heads: Number of attention heads
                - dropout: Dropout rate
                - activation: Activation function name
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dim = config["hidden_dim"]
        self.output_dim = config["output_dim"]
        self.num_layers = config.get("num_layers", 4)
        self.num_heads = config.get("num_heads", 4)
        self.dropout = config.get("dropout", 0.1)
        activation_name = config.get("activation", "gelu")
        self.activation_fn = self._get_activation_module(activation_name)

        # Input projection
        self.input_proj = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Attention layers
        self.layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.layers.append(nn.ModuleList([
                SelfAttention(self.hidden_dim, self.num_heads, self.dropout),
                FeedForwardBlock(self.hidden_dim, dropout=self.dropout)
            ]))
        
        # Output projection
        self.output_proj = nn.Linear(self.hidden_dim, self.output_dim)
        
        # Initialize weights
        self.apply(self._init_weights)

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def _init_weights(self, module):
        """Initialize weights with small values to prevent exploding gradients."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        x = self.activation_fn(self.input_proj(x))
        
        # Apply attention layers
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
            
        return self.output_proj(x)


class AutoEncoder(BaseNetwork):
    """Autoencoder neural network."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - latent_dim: Latent space dimension
                - hidden_dims: List of hidden layer dimensions
                - activation: Activation function name
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.latent_dim = config["latent_dim"]
        self.hidden_dims = config["hidden_dims"]
        activation_name = config.get("activation", "relu")
        self.activation_fn = self._get_activation_module(activation_name)

        # Build encoder
        self.encoder = nn.ModuleList()
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.encoder.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.encoder.append(nn.Linear(prev_dim, self.latent_dim))

        # Build decoder
        self.decoder = nn.ModuleList()
        prev_dim = self.latent_dim
        for hidden_dim in reversed(self.hidden_dims):
            self.decoder.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        self.decoder.append(nn.Linear(prev_dim, self.input_dim))

    def _get_activation_module(self, activation_name: str) -> nn.Module:
        """Convert activation function name to module."""
        if activation_name == "relu":
            return nn.ReLU()
        elif activation_name == "leaky_relu":
            return nn.LeakyReLU()
        elif activation_name == "tanh":
            return nn.Tanh()
        elif activation_name == "sigmoid":
            return nn.Sigmoid()
        elif activation_name == "gelu":
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation_name}")

    def encode(self, x: InputType) -> OutputType:
        """Encode input to latent space.

        Args:
            x: Input tensor or array

        Returns:
            Latent space representation
        """
        x = self._prepare_input(x)
        for layer in self.encoder[:-1]:
            x = self.activation_fn(layer(x))
        return self.encoder[-1](x)

    def decode(self, z: InputType) -> OutputType:
        """Decode latent space to output.

        Args:
            z: Latent space tensor or array

        Returns:
            Decoded output
        """
        z = self._prepare_input(z)
        for layer in self.decoder[:-1]:
            z = self.activation_fn(layer(z))
        return self.decoder[-1](z)

    def forward(self, x: InputType) -> OutputType:
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Reconstructed output
        """
        z = self.encode(x)
        return self.decode(z)


class PINNModel(BaseNetwork):
    """Physics-Informed Neural Network model with multiple architecture options."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str = "tanh",
        fourier_features: bool = True,
        fourier_scale: float = 2.0,
        dropout: float = 0.1,
        layer_norm: bool = True,
        architecture: str = "fourier",
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the PINN model.

        :param input_dim: Input dimension (x, t)
        :param hidden_dim: Hidden layer dimension
        :param output_dim: Output dimension (u)
        :param num_layers: Number of layers
        :param activation: Activation function name ('tanh', 'relu', 'gelu')
        :param fourier_features: Whether to use Fourier features
        :param fourier_scale: Scale factor for Fourier features
        :param dropout: Dropout rate
        :param layer_norm: Whether to use layer normalization
        :param architecture: Neural network architecture to use
        :param device: Device to place the model on
        """
        # Create base config for all architectures
        config = {
            "input_dim": input_dim,
            "output_dim": output_dim,
            "activation": activation,
            "dropout": dropout,
            "device": device,
            "architecture": architecture,
            "fourier_features": fourier_features,
            "fourier_scale": fourier_scale,
            "layer_norm": layer_norm,
        }

        # Add architecture-specific parameters
        if architecture == "fourier":
            config.update(
                {
                    "mapping_size": hidden_dim,
                    "hidden_dims": [hidden_dim] * (num_layers - 1),
                    "scale": fourier_scale,
                }
            )
            super().__init__(config)
            self.model = FourierNetwork(config)
        elif architecture == "resnet":
            config.update({"hidden_dim": hidden_dim, "num_blocks": num_layers - 1})
            super().__init__(config)
            self.model = ResNet(config)
        elif architecture == "siren":
            config.update(
                {
                    "hidden_dims": [hidden_dim] * (num_layers - 1),
                    "omega_0": fourier_scale,
                }
            )
            super().__init__(config)
            self.model = SIREN(config)
        elif architecture == "attention":
            config.update({
                "hidden_dim": hidden_dim,
                "num_layers": num_layers - 1,
                "num_heads": 4,  # Default number of heads
                "dropout": dropout
            })
            super().__init__(config)
            self.model = AttentionNetwork(config)
        elif architecture == "autoencoder":
            config.update(
                {
                    "latent_dim": hidden_dim,
                    "hidden_dims": [hidden_dim * 2] * (num_layers - 1),
                }
            )
            super().__init__(config)
            self.model = AutoEncoder(config)
            # Additional output projection for AutoEncoder to ensure correct output dimension
            self.output_proj = nn.Linear(input_dim, output_dim)
        else:
            config.update({"hidden_dims": [hidden_dim] * (num_layers - 1)})
            super().__init__(config)
            self.model = FeedForwardNetwork(config)

        # Store architecture parameters
        self.architecture = architecture
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.fourier_features = fourier_features
        self.fourier_scale = fourier_scale
        self.dropout = dropout
        self.layer_norm = layer_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Output tensor of shape (batch_size, output_dim)
        """
        if self.architecture == "autoencoder":
            # For autoencoder, we need to project from input_dim to output_dim
            return self.output_proj(self.model(x))
        return self.model(x)

    def save_state(self, path: str):
        """Save model state with architecture information."""
        state = {
            "model_state_dict": self.state_dict(),
            "config": self.config,
            "architecture": {
                "input_dim": self.input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim,
                "num_layers": self.num_layers,
                "activation": self.activation,
                "fourier_features": self.fourier_features,
                "fourier_scale": self.fourier_scale,
                "dropout": self.dropout,
                "layer_norm": self.layer_norm,
                "architecture": self.architecture,
            },
        }
        torch.save(state, path)

    def load_state(self, path: str):
        """Load model state with architecture information."""
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state["model_state_dict"])
        self.config = state["config"]

        # Update instance variables from both config and architecture info
        if "architecture" in state:
            arch_state = state["architecture"]
            self.architecture = arch_state["architecture"]
            self.input_dim = arch_state["input_dim"]
            self.hidden_dim = arch_state["hidden_dim"]
            self.output_dim = arch_state["output_dim"]
            self.num_layers = arch_state["num_layers"]
            self.activation = arch_state["activation"]
            self.fourier_features = arch_state["fourier_features"]
            self.fourier_scale = arch_state["fourier_scale"]
            self.dropout = arch_state["dropout"]
            self.layer_norm = arch_state["layer_norm"]
        else:
            # Fallback to config if architecture info is not present
            self.architecture = self.config["architecture"]
            self.input_dim = self.config["input_dim"]
            self.hidden_dim = self.config.get("hidden_dim", self.input_dim)
            self.output_dim = self.config["output_dim"]
            self.activation = self.config["activation"]
            self.fourier_features = self.config["fourier_features"]
            self.fourier_scale = self.config["fourier_scale"]
            self.dropout = self.config["dropout"]
            self.layer_norm = self.config["layer_norm"]
