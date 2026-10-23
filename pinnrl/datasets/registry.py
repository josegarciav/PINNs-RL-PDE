"""Static metadata registry for `The Well` benchmark datasets.

Each entry records what pinnrl needs to train on a dataset without opening
it: the spatial grid extents, the time step and trajectory length, the
channel names in the order :func:`pinnrl.datasets.well_loader.load_well_slice`
returns them, and the recommended architecture and loss recipe.

The grids, time steps, trajectory lengths and channel layouts were read from
the ``dimensions`` and ``t{0,1,2}_fields`` groups of each dataset's first
training file on the Hugging Face mirror (``hf://datasets/polymathic-ai/``).
Times are relative to the start of a trajectory (``the_well`` normalises them
that way by default). Vector fields expand to one channel per spatial axis and
rank-2 tensors to ``d * d`` channels (``xx, xy, yx, yy`` in 2-D).

``domain`` is the same in every file, but ``dt``, ``n_steps`` and
``time_domain`` describe the first training file only, and five datasets'
other files differ: rayleigh_taylor_instability (one file per Atwood number)
runs for 29.5 to 118 time units, helmholtz_staircase (one file per frequency)
for 2.5 to 100, viscoelastic_instability stores 20 or 60 steps spanning 19 to
76, and the steps of rayleigh_benard and planetswe vary by up to 1.6%. So a
PDE fed by a Well dataset spans the time range of the observations it loads,
and the loader asks for a number of steps, which ``the_well`` caps at each
trajectory's own length.

``euler_multi_quadrants_openBC`` is not registered: its files record negative
trajectory counts, so ``the_well`` 1.2 cannot open it, and the periodic-boundary
variant is not on the mirror.

None of these datasets is governed by one of pinnrl's built-in PDEs (the
acoustic maze has a spatially varying sound speed, the Helmholtz staircase is
a frequency-domain problem), so every entry defaults to ``data_only``; use
``data_augmented`` with your own observations of a pinnrl PDE instead.
"""

from __future__ import annotations

from dataclasses import dataclass

# ``the_well`` reads the first step plus a ``max_rollout_steps``-step rollout
# (default 100), so a full-trajectory sample holds at most this many steps.
DEFAULT_LOADED_STEPS = 101
# A 3-D step is tens of MB (128^3 nodes x 4 float32 channels is 34 MB for
# rayleigh_taylor_instability), so 3-D datasets read a short window by default.
DEFAULT_LOADED_STEPS_3D = 11


@dataclass(frozen=True)
class WellEntry:
    """Defaults pinnrl uses when a Well dataset is selected.

    Attributes:
        name: WellDataset identifier — passed straight to ``well_dataset_name``.
        description: One-line human-readable summary for the dashboard's dataset info box.
        fields: Channel names in the order returned by the loader.
        domain: Per-axis ``(low, high)`` coordinate extents of the spatial grid.
        dt: Mean time between stored steps in the first training file.
        n_steps: Stored steps per trajectory in the first training file.
        default_architecture: Architecture preset most appropriate for the
            field shape (FNO for 2-D grids, feedforward for 3-D).
        default_pde_key: PDE key accepted by
            :func:`pinnrl.pdes.registry.get_pde_spec` when the dataset is
            governed by that PDE, else ``None``.
        recommended_mode: ``data_augmented`` if ``default_pde_key`` is set,
            else ``data_only``.
        default_loaded_steps: Steps :func:`load_well_slice` asks for per
            trajectory when the caller passes no ``n_steps``.
    """

    name: str
    description: str
    fields: tuple[str, ...]
    domain: tuple[tuple[float, float], ...]
    dt: float
    n_steps: int
    default_architecture: str
    default_pde_key: str | None = None
    recommended_mode: str = "data_only"
    default_loaded_steps: int = DEFAULT_LOADED_STEPS

    def __post_init__(self) -> None:  # pragma: no cover - dataclass guard
        if self.n_spatial_dims not in (2, 3):
            raise ValueError(
                f"WellEntry {self.name!r}: n_spatial_dims must be 2 or 3, got {self.n_spatial_dims}"
            )
        if self.recommended_mode not in ("data_only", "data_augmented"):
            raise ValueError(
                f"WellEntry {self.name!r}: recommended_mode must be one of "
                f"'data_only', 'data_augmented'"
            )
        if (self.recommended_mode == "data_augmented") != (self.default_pde_key is not None):
            raise ValueError(
                f"WellEntry {self.name!r}: data_augmented mode goes with a default_pde_key"
            )

    @property
    def n_spatial_dims(self) -> int:
        """Number of spatial axes (2 or 3)."""
        return len(self.domain)

    @property
    def default_input_dim(self) -> int:
        """Network input size: the spatial axes plus time."""
        return self.n_spatial_dims + 1

    @property
    def default_output_dim(self) -> int:
        """Network output size: one per channel."""
        return len(self.fields)

    @property
    def time_domain(self) -> tuple[float, float]:
        """``(0, dt * (n_steps - 1))``: a whole trajectory of the first training file."""
        return (0.0, self.dt * (self.n_steps - 1))

    def steps_to_read(self, n_steps: int | None = None) -> int:
        """Steps :func:`load_well_slice` asks for per trajectory: ``n_steps``, else the default.

        Not capped at ``n_steps`` of the entry: files can hold longer
        trajectories, and ``the_well`` caps each read at its trajectory's length.

        Raises:
            ValueError: If ``n_steps`` is below 1.
        """
        if n_steps is None:
            n_steps = self.default_loaded_steps
        if int(n_steps) < 1:
            raise ValueError(f"n_steps must be at least 1, got {n_steps}")
        return int(n_steps)


def _components(name: str, axes: str, rank: int) -> tuple[str, ...]:
    """Channel names of a vector (rank 1) or rank-2 tensor field."""
    if rank == 1:
        return tuple(f"{name}_{a}" for a in axes)
    return tuple(f"{name}_{a}{b}" for a in axes for b in axes)


WELL_REGISTRY: dict[str, WellEntry] = {
    entry.name: entry
    for entry in (
        WellEntry(
            name="active_matter",
            description="2D active nematic fluid — concentration, velocity, orientation tensor, strain rate.",
            fields=(
                "concentration",
                *_components("velocity", "xy", 1),
                *_components("orientation", "xy", 2),
                *_components("strain_rate", "xy", 2),
            ),
            domain=((0.0, 10.0), (0.0, 10.0)),
            dt=0.25,
            n_steps=81,
            default_architecture="fno",
        ),
        WellEntry(
            name="rayleigh_benard",
            description="2D thermal convection between hot and cold plates.",
            fields=("buoyancy", "pressure", *_components("velocity", "xy", 1)),
            domain=((0.0, 4.0), (0.0, 1.0)),
            dt=0.25,
            n_steps=200,
            default_architecture="fno",
        ),
        WellEntry(
            name="shear_flow",
            description="2D incompressible shear flow with Kelvin-Helmholtz roll-up.",
            fields=("tracer", "pressure", *_components("velocity", "xy", 1)),
            domain=((0.0, 1.0), (0.0, 1.0)),
            dt=0.1,
            n_steps=200,
            default_architecture="fno",
        ),
        WellEntry(
            name="rayleigh_taylor_instability",
            description="3D buoyancy-driven mixing of two density layers.",
            fields=("density", *_components("velocity", "xyz", 1)),
            domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            dt=118.004 / 118,
            n_steps=119,
            default_architecture="feedforward",
            default_loaded_steps=DEFAULT_LOADED_STEPS_3D,
        ),
        WellEntry(
            name="turbulent_radiative_layer_2D",
            description="2D radiatively cooling shear layer (astrophysical).",
            fields=("density", "pressure", *_components("velocity", "xy", 1)),
            domain=((-0.5, 0.5), (-1.0, 2.0)),
            dt=1.597033,
            n_steps=101,
            default_architecture="fno",
        ),
        WellEntry(
            name="MHD_64",
            description="3D magneto-hydrodynamic turbulence at 64^3 resolution.",
            fields=(
                "density",
                *_components("magnetic", "xyz", 1),
                *_components("velocity", "xyz", 1),
            ),
            domain=((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)),
            dt=0.01,
            n_steps=100,
            default_architecture="feedforward",
            default_loaded_steps=DEFAULT_LOADED_STEPS_3D,
        ),
        WellEntry(
            name="gray_scott_reaction_diffusion",
            description="2D Gray-Scott reaction-diffusion (pattern formation).",
            fields=("A", "B"),
            domain=((-1.0, 1.0), (-1.0, 1.0)),
            dt=10.0,
            n_steps=1001,
            default_architecture="fno",
        ),
        WellEntry(
            name="viscoelastic_instability",
            description="2D viscoelastic flow showing elastic turbulence.",
            fields=(
                "pressure",
                "c_zz",
                *_components("velocity", "xy", 1),
                *_components("C", "xy", 2),
            ),
            domain=((0.0, 6.2831855), (-1.0, 1.0)),
            dt=1.0,
            n_steps=20,
            default_architecture="fno",
        ),
        WellEntry(
            name="acoustic_scattering_maze",
            description="2D acoustic scattering through a maze (spatially varying density and sound speed).",
            fields=("pressure", *_components("velocity", "xy", 1)),
            domain=((0.0, 1.0), (0.0, 1.0)),
            dt=4.0 / 201,
            n_steps=202,
            default_architecture="fno",
        ),
        WellEntry(
            name="helmholtz_staircase",
            description="2D Helmholtz (frequency-domain) scattering over a staircase boundary.",
            fields=("pressure_re", "pressure_im"),
            domain=((-8.0, 8.0), (-0.5, 3.5)),
            dt=100.00244 / 49,
            n_steps=50,
            default_architecture="fno",
        ),
        WellEntry(
            name="planetswe",
            description="2D shallow water on a sphere (colatitude theta, longitude phi in radians).",
            fields=("height", *_components("velocity", ("theta", "phi"), 1)),
            domain=((0.0093755, 3.1322172), (0.0, 6.2709136)),
            dt=1007.0045 / 1007,
            n_steps=1008,
            default_architecture="fno",
        ),
    )
}


def list_dataset_names() -> list[str]:
    """Names of every Well dataset known to the registry, sorted."""
    return sorted(WELL_REGISTRY)


def get_entry(name: str) -> WellEntry:
    """Look up a dataset by name. Raises ``KeyError`` with a helpful list."""
    if name not in WELL_REGISTRY:
        raise KeyError(
            f"Unknown Well dataset {name!r}. Known datasets: {', '.join(list_dataset_names())}"
        )
    return WELL_REGISTRY[name]
