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
    """Feed-forward neural network."""

    def __init__(self, config: NetworkConfig) -> None:
        """Initialize the network.

        Args:
            config: Network configuration dictionary containing:
                - input_dim: Input dimension
                - hidden_dims: List of hidden layer dimensions
                - output_dim: Output dimension
                - activation: Activation function name
                - dropout: Dropout rate (optional)
        """
        super().__init__(config)
        self.input_dim = config["input_dim"]
        self.hidden_dims = config["hidden_dims"]
        self.output_dim = config["output_dim"]
        self.activation = getattr(F, config.get("activation", "relu"))
        self.dropout = config.get("dropout", 0.0)

        # Build layers
        self.layers = nn.ModuleList()
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            if self.dropout > 0:
                self.layers.append(nn.Dropout(self.dropout))
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
            if isinstance(layer, nn.Linear):
                x = self.activation(layer(x))
            else:
                x = layer(x)
        return self.layers[-1](x)


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
        self.activation = getattr(F, activation)
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, in_dim),
            nn.LayerNorm(in_dim),
            self.activation,
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with residual connection.

        :param x: Input tensor
        :return: Output tensor
        """
        return self.activation(x + self.layers(x))


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
        self.activation = getattr(F, config.get("activation", "relu"))
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
        """Forward pass of the network.

        Args:
            x: Input tensor or array

        Returns:
            Output tensor or array
        """
        x = self._prepare_input(x)
        x = self.activation(self.input_layer(x))
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
        self.activation = getattr(F, config.get("activation", "relu"))
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
            x = self.activation(layer(x))
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
        self.activation = getattr(F, config.get("activation", "relu"))

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

    def encode(self, x: InputType) -> OutputType:
        """Encode input to latent space.

        Args:
            x: Input tensor or array

        Returns:
            Latent space representation
        """
        x = self._prepare_input(x)
        for layer in self.encoder[:-1]:
            x = self.activation(layer(x))
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
            z = self.activation(layer(z))
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
        :param activation: Activation function ('tanh', 'relu', 'gelu')
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
        elif architecture == "autoencoder":
            config.update(
                {
                    "latent_dim": hidden_dim,
                    "hidden_dims": [hidden_dim * 2] * (num_layers - 1),
                }
            )
            super().__init__(config)
            self.model = AutoEncoder(config)
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
