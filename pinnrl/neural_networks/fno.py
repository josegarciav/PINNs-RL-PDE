"""Fourier Neural Operator (FNO) architecture for Physics-Informed Neural Networks.

Implements a point-wise FNO suitable for PINN training, where inputs are coordinate
pairs (x, t) rather than gridded fields. The architecture lifts inputs to a
higher-dimensional channel space, applies spectral convolution layers (FFT along
the feature dimension -> learnable frequency-domain weights -> IFFT, with a linear
bypass), and projects back to the solution dimension.

Reference: Li et al., "Fourier Neural Operator for Parametric Partial Differential
Equations" (2021), arXiv:2010.08895.
"""

import torch
import torch.nn as nn

from .base_network import BaseNetwork, InputType, NetworkConfig, OutputType


class SpectralConv1d(nn.Module):
    """1D spectral convolution layer operating in Fourier space.

    Treats the feature/channel dimension as a 1D signal:
    1. FFT along the feature dimension
    2. Multiply low-frequency modes by learnable complex weights
    3. IFFT back to feature space
    """

    def __init__(self, channels: int, modes: int):
        super().__init__()
        self.channels = channels
        self.modes = min(modes, channels // 2 + 1)

        # Learnable complex weights: (modes, modes) in complex = (modes, modes, 2) real
        scale = 1.0 / (channels * self.modes)
        self.weights = nn.Parameter(scale * torch.randn(self.modes, self.modes, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spectral convolution along the feature dimension.

        Args:
            x: (batch, channels) tensor

        Returns:
            (batch, channels) tensor
        """
        # FFT along feature dimension
        x_ft = torch.fft.rfft(x, dim=-1)  # (batch, channels//2+1) complex

        # Truncate to kept modes
        modes = self.modes
        x_ft_trunc = x_ft[:, :modes]  # (batch, modes)

        # Complex weight multiplication: (batch, modes) x (modes, modes) -> (batch, modes)
        w = torch.view_as_complex(self.weights)  # (modes, modes)
        out_ft = torch.einsum("bm,mn->bn", x_ft_trunc, w)

        # Pad back to full frequency size and IFFT
        out_full = torch.zeros_like(x_ft)
        out_full[:, :modes] = out_ft
        return torch.fft.irfft(out_full, n=self.channels, dim=-1)


class FNOBlock(nn.Module):
    """Single FNO layer: spectral convolution + linear bypass + activation."""

    def __init__(self, channels: int, modes: int, activation: str = "gelu"):
        super().__init__()
        self.spectral = SpectralConv1d(channels, modes)
        self.linear = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(),
            "sigmoid": nn.Sigmoid(),
        }
        self.activation = activations.get(activation, nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: spectral path + linear bypass with residual connection.

        Args:
            x: (batch, channels) tensor

        Returns:
            (batch, channels) tensor
        """
        spectral_out = self.spectral(x)
        linear_out = self.linear(x)
        return self.activation(self.norm(spectral_out + linear_out + x))


class FNONetwork(BaseNetwork):
    """Fourier Neural Operator network for point-wise PINN training.

    Architecture:
        Input(input_dim) -> Lift(hidden_dim) ->
        [FNOBlock(hidden_dim)] x num_blocks ->
        Project(output_dim)
    """

    def __init__(self, config: NetworkConfig) -> None:
        super().__init__(config)

        self.input_dim = config["input_dim"]
        self.output_dim = config["output_dim"]
        self.hidden_dim = config.get("hidden_dim", 128)
        self.num_blocks = config.get("num_blocks", 4)
        self.modes = config.get("modes", 16)
        activation = config.get("activation", "gelu")

        # Lifting layer: project (x, t) to higher-dimensional channel space
        self.lift = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Spectral convolution blocks
        self.blocks = nn.ModuleList(
            [FNOBlock(self.hidden_dim, self.modes, activation) for _ in range(self.num_blocks)]
        )

        # Projection layers: channel space -> solution
        self.project = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

        self.to(self.device)

    def forward(self, x: InputType) -> OutputType:
        x = self._prepare_input(x)

        # Lift to channel space
        h = self.lift(x)

        # Apply FNO blocks
        for block in self.blocks:
            h = block(h)

        # Project to output
        return self.project(h)
