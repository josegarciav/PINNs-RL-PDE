"""Partial differential equations (PDEs)."""

from .allen_cahn import AllenCahnEquation
from .black_scholes import BlackScholesEquation
from .burgers_equation import BurgersEquation
from .cahn_hilliard import CahnHilliardEquation
from .convection_equation import ConvectionEquation
from .custom import CustomPDE, Fields
from .heat_equation import HeatEquation
from .kdv_equation import KdVEquation
from .pde_base import PDEBase, PDEConfig
from .pendulum_equation import PendulumEquation
from .registry import PDE_SPECS, PDESpec, create_pde, get_pde_spec
from .wave_equation import WaveEquation

__all__ = [
    "AllenCahnEquation",
    "BlackScholesEquation",
    "BurgersEquation",
    "CahnHilliardEquation",
    "ConvectionEquation",
    "CustomPDE",
    "Fields",
    "HeatEquation",
    "KdVEquation",
    "PDEBase",
    "PDEConfig",
    "PendulumEquation",
    "WaveEquation",
    "PDE_SPECS",
    "PDESpec",
    "create_pde",
    "get_pde_spec",
]
