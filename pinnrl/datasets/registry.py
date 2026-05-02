"""Static metadata registry for `The Well` benchmark datasets.

Each entry hand-curates the defaults the dashboard uses to populate the New
Training form when a user picks a dataset: spatial dimensionality, channel
fields, recommended PDE/architecture, and the loss recipe that makes sense
for that dataset (``data_augmented`` when pinnrl already has an analytical
solver, ``data_only`` otherwise).

Names match the strings accepted by ``the_well.data.WellDataset`` as
``well_dataset_name``. Domain and time extents come from the Well dataset
specification page; values that vary per-trajectory are left as the
canonical normalised range and refined at load time from the HDF5
``dimensions`` group when available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class WellEntry:
    """Defaults pinnrl uses when a Well dataset is selected.

    Attributes:
        name: WellDataset identifier — passed straight to ``well_dataset_name``.
        description: One-line human-readable summary for the dashboard pill.
        n_spatial_dims: 2 or 3 (Well does not ship 1D datasets).
        fields: Channel names in the order returned by the loader.
        domain: Per-axis ``(low, high)`` bounds for spatial coordinates.
        time_domain: ``(t_min, t_max)`` over a single trajectory.
        default_pde_key: Key into ``PDE_REGISTRY`` when a matched analytical
            solver exists, else ``None`` (use ``data_only`` mode).
        default_architecture: Architecture preset most appropriate for the
            field shape (FNO for periodic 2D fluids, MLP for the rest).
        default_input_dim: Network input dim — ``len(domain) + 1`` for time.
        default_output_dim: Network output dim — ``len(fields)``.
        recommended_mode: ``data_augmented`` if matched, else ``data_only``.
    """

    name: str
    description: str
    n_spatial_dims: int
    fields: Tuple[str, ...]
    domain: Tuple[Tuple[float, float], ...]
    time_domain: Tuple[float, float]
    default_pde_key: Optional[str]
    default_architecture: str
    default_input_dim: int
    default_output_dim: int
    recommended_mode: str

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.n_spatial_dims not in (2, 3):
            raise ValueError(
                f"WellEntry {self.name!r}: n_spatial_dims must be 2 or 3, "
                f"got {self.n_spatial_dims}"
            )
        if len(self.domain) != self.n_spatial_dims:
            raise ValueError(
                f"WellEntry {self.name!r}: domain has {len(self.domain)} axes "
                f"but n_spatial_dims is {self.n_spatial_dims}"
            )
        if self.recommended_mode not in ("data_only", "data_augmented"):
            raise ValueError(
                f"WellEntry {self.name!r}: recommended_mode must be one of "
                f"'data_only', 'data_augmented'"
            )
        if self.recommended_mode == "data_augmented" and self.default_pde_key is None:
            raise ValueError(
                f"WellEntry {self.name!r}: data_augmented mode requires a " f"default_pde_key"
            )


# Curated subset of The Well datasets. Names are the canonical
# ``well_dataset_name`` strings from PolymathicAI/the_well. Extend as
# needed — every entry here automatically appears in the dashboard.
_PERIODIC_2D_UNIT: Tuple[Tuple[float, float], ...] = ((0.0, 1.0), (0.0, 1.0))
_PERIODIC_3D_UNIT: Tuple[Tuple[float, float], ...] = (
    (0.0, 1.0),
    (0.0, 1.0),
    (0.0, 1.0),
)


WELL_REGISTRY: Dict[str, WellEntry] = {
    "active_matter": WellEntry(
        name="active_matter",
        description="2D active nematic fluid — concentration, velocity, orientation tensor, strain rate.",
        n_spatial_dims=2,
        fields=(
            "concentration",
            "velocity_x",
            "velocity_y",
            "orientation_xx",
            "orientation_xy",
            "orientation_yx",
            "orientation_yy",
            "strain_rate_xx",
            "strain_rate_xy",
            "strain_rate_yx",
            "strain_rate_yy",
        ),
        domain=_PERIODIC_2D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=11,
        recommended_mode="data_only",
    ),
    "rayleigh_benard": WellEntry(
        name="rayleigh_benard",
        description="2D thermal convection between hot and cold plates.",
        n_spatial_dims=2,
        fields=("buoyancy", "pressure", "velocity_x", "velocity_y"),
        domain=((0.0, 4.0), (0.0, 1.0)),
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=4,
        recommended_mode="data_only",
    ),
    "shear_flow": WellEntry(
        name="shear_flow",
        description="2D incompressible shear flow with Kelvin-Helmholtz roll-up.",
        n_spatial_dims=2,
        fields=("tracer", "pressure", "velocity_x", "velocity_y"),
        domain=((0.0, 1.0), (0.0, 2.0)),
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=4,
        recommended_mode="data_only",
    ),
    "rayleigh_taylor_instability": WellEntry(
        name="rayleigh_taylor_instability",
        description="3D buoyancy-driven mixing of two density layers.",
        n_spatial_dims=3,
        fields=("density", "pressure", "velocity_x", "velocity_y", "velocity_z"),
        domain=_PERIODIC_3D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="mlp",
        default_input_dim=4,
        default_output_dim=5,
        recommended_mode="data_only",
    ),
    "turbulent_radiative_layer_2D": WellEntry(
        name="turbulent_radiative_layer_2D",
        description="2D radiatively cooling shear layer (astrophysical).",
        n_spatial_dims=2,
        fields=("density", "pressure", "velocity_x", "velocity_y"),
        domain=((0.0, 1.0), (0.0, 0.5)),
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=4,
        recommended_mode="data_only",
    ),
    "MHD_64": WellEntry(
        name="MHD_64",
        description="3D magneto-hydrodynamic turbulence at 64^3 resolution.",
        n_spatial_dims=3,
        fields=(
            "density",
            "velocity_x",
            "velocity_y",
            "velocity_z",
            "magnetic_x",
            "magnetic_y",
            "magnetic_z",
        ),
        domain=_PERIODIC_3D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="mlp",
        default_input_dim=4,
        default_output_dim=7,
        recommended_mode="data_only",
    ),
    "gray_scott_reaction_diffusion": WellEntry(
        name="gray_scott_reaction_diffusion",
        description="2D Gray-Scott reaction-diffusion (pattern formation).",
        n_spatial_dims=2,
        fields=("A", "B"),
        domain=_PERIODIC_2D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=2,
        recommended_mode="data_only",
    ),
    "euler_multi_quadrants_periodicBC": WellEntry(
        name="euler_multi_quadrants_periodicBC",
        description="2D compressible Euler — Riemann-style multi-quadrant ICs (periodic).",
        n_spatial_dims=2,
        fields=("density", "pressure", "velocity_x", "velocity_y"),
        domain=_PERIODIC_2D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=4,
        recommended_mode="data_only",
    ),
    "viscoelastic_instability": WellEntry(
        name="viscoelastic_instability",
        description="2D viscoelastic flow showing elastic turbulence.",
        n_spatial_dims=2,
        fields=("c_xx", "c_xy", "c_yy", "pressure", "velocity_x", "velocity_y"),
        domain=_PERIODIC_2D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=6,
        recommended_mode="data_only",
    ),
    "acoustic_scattering_maze": WellEntry(
        name="acoustic_scattering_maze",
        description="2D acoustic wave scattering through a maze geometry.",
        n_spatial_dims=2,
        fields=("pressure", "velocity_x", "velocity_y"),
        domain=_PERIODIC_2D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key="wave",
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=3,
        recommended_mode="data_augmented",
    ),
    "helmholtz_staircase": WellEntry(
        name="helmholtz_staircase",
        description="2D Helmholtz scattering on a staircase domain.",
        n_spatial_dims=2,
        fields=("real", "imaginary"),
        domain=_PERIODIC_2D_UNIT,
        time_domain=(0.0, 1.0),
        default_pde_key="wave",
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=2,
        recommended_mode="data_augmented",
    ),
    "planetswe": WellEntry(
        name="planetswe",
        description="2D shallow-water on a sphere (planetary scale).",
        n_spatial_dims=2,
        fields=("height", "velocity_lon", "velocity_lat"),
        domain=((0.0, 360.0), (-90.0, 90.0)),
        time_domain=(0.0, 1.0),
        default_pde_key=None,
        default_architecture="fno",
        default_input_dim=3,
        default_output_dim=3,
        recommended_mode="data_only",
    ),
}


def list_dataset_names() -> List[str]:
    """Names of every Well dataset known to the registry, sorted."""
    return sorted(WELL_REGISTRY)


def get_entry(name: str) -> WellEntry:
    """Look up a dataset by name. Raises ``KeyError`` with a helpful list."""
    if name not in WELL_REGISTRY:
        raise KeyError(
            f"Unknown Well dataset {name!r}. Known datasets: " f"{', '.join(list_dataset_names())}"
        )
    return WELL_REGISTRY[name]
