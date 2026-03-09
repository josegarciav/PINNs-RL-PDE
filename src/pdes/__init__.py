"""
Partial differential equations (PDEs).
"""

from .allen_cahn import AllenCahnEquation
from .black_scholes import BlackScholesEquation
from .burgers_equation import BurgersEquation
from .cahn_hilliard import CahnHilliardEquation
from .convection_equation import ConvectionEquation
from .heat_equation import HeatEquation
from .kdv_equation import KdVEquation
from .pde_base import PDEBase, PDEConfig  # noqa: F401
from .pendulum_equation import PendulumEquation
from .wave_equation import WaveEquation


def create_pde(config):
    """
    Create a PDE instance based on the configuration.

    Args:
        config: PDE configuration

    Returns:
        PDE instance
    """
    # Default to heat equation if type is not specified
    pde_type = getattr(config, "type", "heat").lower() if hasattr(config, "type") else "heat"

    if pde_type == "heat":
        return HeatEquation(config)
    elif pde_type == "wave":
        return WaveEquation(config)
    elif pde_type == "burgers":
        return BurgersEquation(config)
    elif pde_type == "kdv":
        return KdVEquation(config)
    elif pde_type == "cahn_hilliard":
        return CahnHilliardEquation(config)
    elif pde_type == "allen_cahn":
        return AllenCahnEquation(config)
    elif pde_type == "black_scholes":
        return BlackScholesEquation(config)
    elif pde_type == "convection":
        return ConvectionEquation(config)
    elif pde_type == "pendulum":
        return PendulumEquation(config)
    else:
        raise ValueError(f"PDE type not supported: {pde_type}")
