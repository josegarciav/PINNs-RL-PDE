"""Single source of truth for the PDEs that ship with pinnrl.

Every entry point (``pinnrl-train``, the dashboard, the dataset registry)
resolves PDE names through :func:`get_pde_spec`, which accepts either the
config key used under ``pde_configs`` in ``config.yaml`` (``"heat"``,
``"allen_cahn"``) or the display name (``"Heat Equation"``).
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Type

from .allen_cahn import AllenCahnEquation
from .black_scholes import BlackScholesEquation
from .burgers_equation import BurgersEquation
from .cahn_hilliard import CahnHilliardEquation
from .convection_equation import ConvectionEquation
from .heat_equation import HeatEquation
from .kdv_equation import KdVEquation
from .pde_base import PDEBase, PDEConfig
from .pendulum_equation import PendulumEquation
from .wave_equation import WaveEquation


@dataclass(frozen=True)
class PDESpec:
    """A registered PDE.

    Attributes:
        key: Config key under ``pde_configs`` in ``config.yaml``.
        display_name: Human-readable name shown by the dashboard and CLI.
        cls: The :class:`PDEBase` subclass implementing it.
    """

    key: str
    display_name: str
    cls: Type[PDEBase]


PDE_SPECS: Tuple[PDESpec, ...] = (
    PDESpec("heat", "Heat Equation", HeatEquation),
    PDESpec("heat_2d", "Heat Equation 2D", HeatEquation),
    PDESpec("wave", "Wave Equation", WaveEquation),
    PDESpec("burgers", "Burgers Equation", BurgersEquation),
    PDESpec("convection", "Convection Equation", ConvectionEquation),
    PDESpec("kdv", "KdV Equation", KdVEquation),
    PDESpec("allen_cahn", "Allen-Cahn Equation", AllenCahnEquation),
    PDESpec("cahn_hilliard", "Cahn-Hilliard Equation", CahnHilliardEquation),
    PDESpec("black_scholes", "Black-Scholes Equation", BlackScholesEquation),
    PDESpec("pendulum", "Pendulum Equation", PendulumEquation),
)

_LOOKUP: Dict[str, PDESpec] = {}
for _spec in PDE_SPECS:
    _LOOKUP[_spec.key] = _spec
    _LOOKUP[_spec.display_name.lower()] = _spec


def get_pde_spec(name: str) -> PDESpec:
    """Resolve a config key or display name (case-insensitive) to its spec.

    Raises:
        KeyError: If ``name`` matches no registered PDE.
    """
    spec = _LOOKUP.get(name.strip().lower())
    if spec is None:
        available = ", ".join(s.key for s in PDE_SPECS)
        raise KeyError(f"Unknown PDE {name!r}. Available: {available}")
    return spec


def create_pde(name: str, config: PDEConfig) -> PDEBase:
    """Instantiate the PDE registered under ``name`` with ``config``."""
    return get_pde_spec(name).cls(config=config)
