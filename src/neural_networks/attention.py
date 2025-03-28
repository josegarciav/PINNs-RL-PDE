"""Transformer-based architecture with self-attention mechanisms."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base_network import BaseNetwork, InputType, OutputType, NetworkConfig


class SelfAttention(nn.Module):
    """Self-attention layer."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1) -> None:
        """
        Initialize the attention layer.

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
        """
        Forward pass of the layer.

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
        """
        Initialize the feed-forward block.

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
        """
        Forward pass of the block.

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        return self.layer_norm(x + self.net(x))


class AttentionNetwork(BaseNetwork):
    """Neural network with self-attention layers."""

    def __init__(self, config: NetworkConfig) -> None:
        """
        Initialize the network.

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

    def _init_weights(self, module):
        """Initialize weights with small values to prevent exploding gradients."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x: InputType) -> OutputType:
        """
        Forward pass of the network.

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