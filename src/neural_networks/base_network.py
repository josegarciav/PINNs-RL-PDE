"""Base class for neural network architectures."""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn

# Type aliases for better readability
InputType = Union[torch.Tensor, Union[np.ndarray, List]]
OutputType = Union[torch.Tensor, Union[np.ndarray, List]]
NetworkConfig = Dict[str, Any]


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

        if x.device != self.device:
            x = x.to(self.device)

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
