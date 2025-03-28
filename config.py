"""Configuration utilities for the PINN framework."""

import os
import yaml
from typing import Dict, Any, Optional, Union


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the configuration from a YAML file.
    
    Args:
        config_path: Path to the configuration file. If None, uses the default path.
        
    Returns:
        Dictionary with configuration parameters
    """
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_architecture_config(config: Dict[str, Any], architecture: str) -> Dict[str, Any]:
    """
    Get the specific configuration for a given architecture.
    
    Args:
        config: Full configuration dictionary
        architecture: Name of the architecture
        
    Returns:
        Dictionary with architecture-specific configuration
    """
    # Start with the base model config
    model_config = config.get("model", {}).copy()
    
    # Override with architecture-specific config
    arch_config = config.get("architectures", {}).get(architecture, {})
    model_config.update(arch_config)
    
    return model_config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries, with override_config taking precedence.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to override values
        
    Returns:
        Merged configuration dictionary
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if (
            key in base_config 
            and isinstance(base_config[key], dict) 
            and isinstance(value, dict)
        ):
            merged[key] = merge_configs(base_config[key], value)
        else:
            merged[key] = value
    
    return merged 