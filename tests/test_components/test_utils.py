import os
import torch
import yaml
import numpy as np
from src.pdes.pde_base import PDEConfig
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.convection_equation import ConvectionEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pdes.pendulum_equation import PendulumEquation


def load_pde_config(pde_type, device=None):
    """
    Load a specific PDE configuration from config.yaml.
    
    Args:
        pde_type (str): The type of PDE to load ('heat', 'wave', 'burgers', etc.)
        device (torch.device, optional): Device to use for tensor computations. 
                                         Defaults to CPU if None.
                                         
    Returns:
        tuple: (PDEConfig object, PDE Class) for creating PDE instances
    """
    # Default device is CPU if not provided
    if device is None:
        device = torch.device("cpu")
    
    # Load config file
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Get PDE-specific configuration
    if "pde_configs" not in config or pde_type not in config["pde_configs"]:
        raise ValueError(f"PDE type '{pde_type}' not found in config.yaml")
    
    pde_config = config["pde_configs"][pde_type]
    
    # Add device to config
    pde_config["device"] = device
    
    # Create a PDEConfig object
    pde_config_obj = PDEConfig(
        name=pde_config.get("name", f"{pde_type.capitalize()} Equation"),
        domain=pde_config.get("domain", [(0.0, 1.0)]),
        time_domain=pde_config.get("time_domain", (0.0, 1.0)),
        parameters=pde_config.get("parameters", {}),
        boundary_conditions=pde_config.get("boundary_conditions", {}),
        initial_condition=pde_config.get("initial_condition", {}),
        exact_solution=pde_config.get("exact_solution", {}),
        dimension=pde_config.get("dimension", 1),
        input_dim=pde_config.get("input_dim"),
        output_dim=pde_config.get("output_dim"),
        architecture=pde_config.get("architecture"),
        device=device
    )
    
    # Return the appropriate PDE class based on type
    pde_class_map = {
        "heat": HeatEquation,
        "wave": WaveEquation,
        "burgers": BurgersEquation,
        "kdv": KdVEquation,
        "convection": ConvectionEquation,
        "allen_cahn": AllenCahnEquation,
        "cahn_hilliard": CahnHilliardEquation,
        "black_scholes": BlackScholesEquation,
        "pendulum": PendulumEquation
    }
    
    if pde_type not in pde_class_map:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    return pde_config_obj, pde_class_map[pde_type]

def create_pde_from_config(pde_type, device=None, dimension=None):
    """
    Create a PDE instance from the configuration in config.yaml.
    
    Args:
        pde_type (str): The type of PDE to create ('heat', 'wave', 'burgers', etc.)
        device (torch.device, optional): Device to use for tensor computations.
        dimension (int, optional): Override the dimension in the config.
        
    Returns:
        PDEBase: An instance of the appropriate PDE class
    """
    config, pde_class = load_pde_config(pde_type, device)
    
    # Override dimension if provided
    if dimension is not None:
        config.dimension = dimension
        
        # For multi-dimensional PDEs, update domain structure if needed
        if dimension > 1 and len(config.domain) == 1:
            # Extend the domain to multiple dimensions
            base_domain = config.domain[0]
            config.domain = [base_domain] * dimension
        
        # Update input_dim based on the new dimension
        # For spatial dimensions + time, input_dim = dimension + 1
        config.input_dim = dimension + 1
    
    # Create the PDE instance with the config
    return pde_class(config=config)
