"""Partial differential equations (PDEs)."""

from .allen_cahn import AllenCahnEquation
from .black_scholes import BlackScholesEquation
from .burgers_equation import BurgersEquation
from .cahn_hilliard import CahnHilliardEquation
from .convection_equation import ConvectionEquation
from .custom import CustomPDE, Fields
from .heat_equation import HeatEquation
from .inputs import TimeSeries
from .kdv_equation import KdVEquation
from .pde_base import PDEBase, PDEConfig
from .pendulum_equation import PendulumEquation
from .registry import PDE_SPECS, PDESpec, create_pde, get_pde_spec
from .sensors import BoundarySensor, IntegralSensor, PointSensor, Sensor, sensor_jacobian
from .unknowns import UnknownFunction
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
    "PointSensor",
    "BoundarySensor",
    "IntegralSensor",
    "Sensor",
    "sensor_jacobian",
    "TimeSeries",
    "UnknownFunction",
    "WaveEquation",
    "PDE_SPECS",
    "PDESpec",
    "create_pde",
    "get_pde_spec",
]
