"""Base class for PDEs: derivatives, boundary/initial conditions, sampling, loss and validation."""

import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn as nn


@dataclass
class PDEConfig:
    """Configuration dataclass for PDE parameters."""

    name: str
    domain: (
        tuple[float, float] | list[tuple[float, float]]
    )  # For 1D, list of tuples for higher dimensions
    time_domain: tuple[float, float] | None  # None: the observations' time range
    parameters: dict[str, float]
    boundary_conditions: dict[str, dict[str, Any]]
    initial_condition: dict[str, Any]
    exact_solution: dict[str, Any]
    dimension: int = 1  # Default to 1D
    input_dim: int | None = None  # Input dimensions for the NN (spatial + temporal)
    output_dim: int | None = None  # Output dimensions for the NN (solution components)
    architecture: str | None = None  # Neural network architecture for this PDE
    device: torch.device | None = None
    training: dict[str, Any] | None = None  # A TrainingConfig or a plain dict
    # Inverse-problem fields. Empty in the default forward setup.
    # Names listed in ``trainable_parameters`` are wrapped as ``nn.Parameter``s
    # initialized from ``parameter_initial_guesses`` (or ``parameters[name]``).
    trainable_parameters: list[str] = field(default_factory=list)
    parameter_initial_guesses: dict[str, float] = field(default_factory=dict)
    # An in-memory dict {"x", "t", "u"}, {"path": "<file.npz>"}, or a Well
    # spec {"source": "well", "name": ...} (see ``_load_observation_data``).
    observation_data: dict[str, Any] | None = None


_BOUNDARY_KINDS = ("dirichlet", "neumann", "periodic")


def _as_scalar(name: str, value: Any) -> float | None:
    """A trainable parameter value as a float; one-element lists are unwrapped."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(
                f"Parameter {name!r} has {len(value)} components; only scalar "
                "parameters can be identified in inverse mode."
            )
        value = value[0]
    return float(value)


class PDEBase:
    """Base class for PDEs solved by PINNs.

    A subclass provides the physics:

    * :meth:`compute_residual` — the PDE operator applied to the network.
    * :meth:`_create_initial_condition` — ``u(x, t0)`` for its IC types.
    * :meth:`_exact_solution` — a closed-form solution, when one exists.
    * ``time_order = 2`` for second-order-in-time PDEs; the initial loss then
      also enforces ``u_t(x, t0) = initial_condition["velocity"]`` (default 0).

    The base class owns everything else: derivatives, collocation sampling,
    boundary conditions and the composite loss. ``boundary_conditions`` in the
    config maps a kind to its options; every spatial axis gets the condition
    on both faces:

    * ``dirichlet``: ``{"value": v}`` or ``{"left": a, "right": b}`` (the lower /
      upper face), or ``{"type": "exact"}`` to take values from the exact solution;
    * ``neumann``: ``{"value": g}`` — ``du/dx_axis = g`` on both faces;
    * ``periodic``: ``{}`` — ``u`` and ``du/dx_axis`` agree on opposite faces.

    ``exact_solution: {}`` in the config means the problem has no closed-form
    solution; :meth:`exact_solution` then returns ``None``.
    """

    time_order = 1

    def __init__(self, config: PDEConfig, bandit_sampler=None):
        """Initialize PDE with given configuration.

        Args:
            config: Configuration dataclass.
            bandit_sampler: Optional :class:`ResidualBanditSampler` for the
                ``adaptive`` strategy; built lazily with defaults when omitted.
        """
        self.config = config
        self.domain = config.domain
        self.bandit_sampler = bandit_sampler

        # One (lo, hi) pair per spatial axis; a bare (lo, hi) means one axis.
        if not self.domain:
            self.domain = [(0.0, 1.0)]
        elif isinstance(self.domain[0], (list, tuple)):
            self.domain = [(float(lo), float(hi)) for lo, hi in self.domain]
        else:
            self.domain = [(float(self.domain[0]), float(self.domain[1]))]

        # Update config with normalized domain format
        self.config.domain = self.domain

        # Handle time_domain/t_domain property
        if hasattr(config, "time_domain"):
            self.time_domain = config.time_domain
        elif hasattr(config, "t_domain"):
            self.time_domain = config.t_domain
        else:
            self.time_domain = [0.0, 1.0]

        # Convert time_domain to tuple if it's a list
        if isinstance(self.time_domain, list):
            self.time_domain = tuple(self.time_domain)

        # Setup device - Handle device explicitly and carefully
        if hasattr(config, "device") and config.device is not None:
            # If config.device is already a torch.device object, use it directly
            if isinstance(config.device, torch.device):
                self.device = config.device
            else:
                # Try to convert string to torch.device
                try:
                    self.device = torch.device(str(config.device))
                except Exception:
                    print(f"Warning: Invalid device '{config.device}', falling back to CPU")
                    self.device = torch.device("cpu")
        else:
            # Fallback to CPU if no device specified
            self.device = torch.device("cpu")

        # Ensure config has the same device
        self.config.device = self.device

        # Store dimensionality
        self.dimension = config.dimension

        # Make sure parameters exist but preserve any existing parameters
        if not hasattr(config, "parameters"):
            config.parameters = {}
        elif config.parameters is None:
            config.parameters = {}

        # Register trainable parameters as nn.Parameter (inverse-problem mode).
        # In forward mode this dict stays empty and adds no behaviour.
        self._trainable_params: nn.ParameterDict = nn.ParameterDict()
        trainable_names = list(getattr(config, "trainable_parameters", []) or [])
        initial_guesses = dict(getattr(config, "parameter_initial_guesses", {}) or {})
        # Snapshot the original (true) values BEFORE the parameters become
        # trainable, so we can plot a reference line in the dashboard.
        self._true_parameters: dict[str, float] = {}
        for name in trainable_names:
            true_val = _as_scalar(name, config.parameters.get(name))
            if true_val is not None:
                self._true_parameters[name] = true_val
            init_val = _as_scalar(
                name, initial_guesses.get(name, true_val if true_val is not None else 1.0)
            )
            self._trainable_params[name] = nn.Parameter(torch.tensor(init_val, device=self.device))

        # Load observation data (used by the data-fitting loss in inverse mode).
        self.observation_data: dict[str, torch.Tensor] | None = self._load_observation_data(
            getattr(config, "observation_data", None)
        )
        # No time window of its own (a Well dataset, whose trajectory length can
        # differ by file): span the time range of the observations, else [0, 1].
        if self.time_domain is None:
            if self.observation_data is not None:
                t = self.observation_data["t"]
                self.time_domain = (float(t.min()), float(t.max()))
            else:
                self.time_domain = (0.0, 1.0)
            self.config.time_domain = self.time_domain

        self.initial_condition = self._create_initial_condition(
            dict(config.initial_condition or {})
        )
        self.boundary_conditions = self._parse_boundary_conditions(config.boundary_conditions or {})
        self._fixed_points: dict[tuple[str, int], Any] = {}

        # Setup neural network parameters if not specified
        if self.config.input_dim is None:
            self.config.input_dim = self.dimension + 1  # Spatial dimensions + time

        if self.config.output_dim is None:
            self.config.output_dim = 1  # Default to single output (u)

    def get_parameter(self, name: str, default=None, required: bool = False):
        """Safely get a parameter value with validation.

        When ``name`` has been registered as a trainable parameter (inverse
        problem mode), the live ``nn.Parameter`` tensor is returned so that
        downstream residual computations build the autograd graph through it.
        Otherwise the plain float from ``config.parameters`` is returned.

        Args:
            name: Parameter name
            default: Default value if parameter not found
            required: Whether parameter is required (raises error if missing)

        Returns:
            Parameter value (``nn.Parameter`` if trainable, else float)

        Raises:
            ValueError: If required parameter is missing
        """
        # Trainable param takes precedence over the static config value.
        trainable = getattr(self, "_trainable_params", None)
        if trainable is not None and name in trainable:
            return trainable[name]

        if not hasattr(self.config, "parameters") or self.config.parameters is None:
            if required:
                raise ValueError(f"Required parameter '{name}' not found in config")
            return default

        value = self.config.parameters.get(name, default)
        if value is None and required:
            raise ValueError(f"Required parameter '{name}' not found in config")
        return value

    def _compute_data_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Reduction loss between the model and the attached observations.

        Returns a zero tensor (no autograd connection) when no observation_data
        is configured, so PDEs that call this in forward mode see a no-op.
        """
        obs = getattr(self, "observation_data", None)
        if not obs:
            return torch.tensor(0.0, device=self.device)
        u_pred_obs = model(torch.cat([obs["x"], obs["t"]], dim=1))
        return self._apply_loss_fn(u_pred_obs - obs["u"])

    def _training_option(self, name: str, default=None):
        """Read ``name`` from ``config.training`` (a TrainingConfig, a dict, or absent)."""
        training = getattr(self.config, "training", None)
        if training is None:
            return default
        if isinstance(training, dict):
            return training.get(name, default)
        return getattr(training, name, default)

    def _apply_loss_fn(self, error: torch.Tensor) -> torch.Tensor:
        """Reduce a per-sample residual/error tensor into a scalar loss.

        The reduction is selected by ``training.loss_function`` (default mse).
        Unknown names fall back to mse so existing test trajectories are
        preserved.
        """
        name = self._training_option("loss_function", "mse")
        if name == "mae":
            return torch.mean(torch.abs(error))
        if name == "huber":
            return torch.nn.functional.huber_loss(
                error,
                torch.zeros_like(error),
                reduction="mean",
                delta=float(self._training_option("huber_delta", 1.0)),
            )
        return torch.mean(error**2)

    def trainable_parameters_iter(self):
        """Iterator over registered trainable PDE parameters (for the optimizer)."""
        if hasattr(self, "_trainable_params"):
            return self._trainable_params.parameters()
        return iter(())

    def get_trainable_parameter_values(self) -> dict[str, float]:
        """Snapshot of current trainable parameter values as plain floats."""
        if not hasattr(self, "_trainable_params"):
            return {}
        return {name: float(p.detach().cpu().item()) for name, p in self._trainable_params.items()}

    def _load_observation_data(
        self, obs_cfg: dict[str, Any] | None
    ) -> dict[str, torch.Tensor] | None:
        """Convert raw observation-data spec into device tensors.

        Accepts:

        * A path-based spec ``{"path": "/some/file.npz"}`` (the npz must
          contain ``x``, ``t``, ``u``).
        * An inline spec carrying numpy/list arrays under those same keys.
        * A Well-dataset spec ``{"source": "well", "name": ..., "split": ...,
          "n_traj": ..., "n_points": ..., "seed": ..., "base": ...}``,
          which is delegated to :func:`pinnrl.datasets.load_well_slice`.

        Returns ``None`` when ``obs_cfg`` is empty.
        """
        if not obs_cfg:
            return None
        device = self.device

        if obs_cfg.get("source") == "well":
            from pinnrl.datasets import load_well_slice

            kwargs = {k: v for k, v in obs_cfg.items() if k != "source"}
            kwargs.setdefault("device", device)
            return load_well_slice(**kwargs)

        if "path" in obs_cfg and obs_cfg["path"]:
            path = obs_cfg["path"]
            if not os.path.exists(path):
                raise FileNotFoundError(f"Observation data file not found: {path}")
            data = np.load(path)
            x = np.asarray(data["x"], dtype=np.float32)
            t = np.asarray(data["t"], dtype=np.float32)
            u = np.asarray(data["u"], dtype=np.float32)
        elif all(k in obs_cfg for k in ("x", "t", "u")):
            x_raw, t_raw, u_raw = obs_cfg["x"], obs_cfg["t"], obs_cfg["u"]
            # Already-tensor case is short-circuited so we don't double-wrap.
            if all(isinstance(v, torch.Tensor) for v in (x_raw, t_raw, u_raw)):
                return {
                    "x": x_raw.to(device),
                    "t": t_raw.to(device),
                    "u": u_raw.to(device),
                }
            x = np.asarray(x_raw, dtype=np.float32)
            t = np.asarray(t_raw, dtype=np.float32)
            u = np.asarray(u_raw, dtype=np.float32)
        else:
            return None

        if x.ndim == 1:
            x = x.reshape(-1, 1)
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        if u.ndim == 1:
            u = u.reshape(-1, 1)

        return {
            "x": torch.tensor(x, device=device),
            "t": torch.tensor(t, device=device),
            "u": torch.tensor(u, device=device),
        }

    def generate_synthetic_observations(
        self,
        n_points: int = 200,
        noise_std: float = 0.0,
        seed: int | None = 0,
    ) -> dict[str, torch.Tensor]:
        """Sample noisy observations from the analytical solution.

        Useful for inverse-problem demos and CI tests: pick uniform points in
        the domain, evaluate ``self.exact_solution``, optionally add Gaussian
        noise, and store the result on ``self.observation_data`` so the next
        ``compute_loss`` call automatically picks it up.

        Args:
            n_points: Number of observations.
            noise_std: Standard deviation of the additive Gaussian noise.
            seed: Seed for the points and the noise; ``None`` leaves it unseeded.

        Returns:
            ``{"x", "t", "u"}`` tensors, also stored on ``self.observation_data``.

        Raises:
            ValueError: If the PDE has no exact solution configured.
        """
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(int(seed))

        spatial_dims = max(int(self.dimension), 1)
        x_cols = []
        for d in range(spatial_dims):
            lo, hi = self.domain[d]
            x_cols.append(torch.rand(n_points, 1, generator=gen) * (hi - lo) + lo)
        x = (
            torch.cat(x_cols, dim=1).to(self.device)
            if spatial_dims > 1
            else x_cols[0].to(self.device)
        )

        t_lo, t_hi = self.time_domain[0], self.time_domain[1]
        t = (torch.rand(n_points, 1, generator=gen) * (t_hi - t_lo) + t_lo).to(self.device)

        # Temporarily unregister the trainable params so ``get_parameter``
        # returns the TRUE values from ``config.parameters`` (the ones
        # snapshotted in ``_true_parameters``) rather than the current guess.
        # Without this, synthetic obs generated in inverse mode would already
        # encode the wrong α and the trainer would converge to the guess
        # rather than recovering the true value.
        saved_params = self._trainable_params if hasattr(self, "_trainable_params") else None
        try:
            if saved_params is not None and self._true_parameters:
                self._trainable_params = nn.ParameterDict()
            if not self.has_exact_solution:
                raise ValueError(
                    f"{self.config.name}: synthetic observations need a configured exact_solution"
                )
            with torch.no_grad():
                u = self.exact_solution(x, t)
                if noise_std and noise_std > 0:
                    noise = torch.randn(u.shape, generator=gen).to(self.device) * float(noise_std)
                    u = u + noise
        finally:
            if saved_params is not None:
                self._trainable_params = saved_params

        self.observation_data = {"x": x, "t": t, "u": u}
        return self.observation_data

    def _create_initial_condition(self, params: dict[str, Any]) -> Callable:
        """Return ``u0(x, t)`` for the generic IC types; subclasses add their own.

        Generic types (all functions of the first spatial coordinate ``x_1``):

        * ``fixed`` — ``value``;
        * ``gaussian`` — ``exp(-(x_1 - mean)² / (2 std²))``;
        * ``sine`` — ``amplitude · sin(frequency · π · x_1)``;
        * ``tanh`` — ``tanh(x_1 / epsilon)``;
        * ``random`` — a smooth, reproducible perturbation: ``amplitude`` times a
          normalised sum of five cosine modes over the domain, with weights and
          phases drawn from ``seed`` (default 0). Bounded by ``amplitude``.

        An empty ``initial_condition`` returns ``None``: no initial-condition term.

        Raises:
            ValueError: For an unknown type.
        """
        ic_type = params.get("type")
        if not params:
            return None
        if ic_type == "tanh":
            epsilon = float(params.get("epsilon", 0.1))
            return lambda x, t: torch.tanh(x[:, 0:1] / epsilon)
        if ic_type == "random":
            rng = np.random.default_rng(params.get("seed", 0))
            weights = rng.uniform(-1.0, 1.0, size=5)
            phases = rng.uniform(0.0, 2 * np.pi, size=5)
            amplitude = float(params.get("amplitude", 0.1)) / np.abs(weights).sum()
            lo, hi = (float(b) for b in self.domain[0])

            def perturbation(x, t):
                s = (x[:, 0:1] - lo) / (hi - lo)
                return amplitude * sum(
                    float(w) * torch.cos(2 * np.pi * (m + 1) * s + float(p))
                    for m, (w, p) in enumerate(zip(weights, phases, strict=False))
                )

            return perturbation
        if ic_type == "fixed":
            value = float(params.get("value", 0.0))
            return lambda x, t: torch.full_like(x[:, 0:1], value)
        if ic_type == "gaussian":
            mean, std = float(params.get("mean", 0.0)), float(params.get("std", 0.1))
            return lambda x, t: torch.exp(-((x[:, 0:1] - mean) ** 2) / (2 * std**2))
        if ic_type == "sine":
            a, k = float(params.get("amplitude", 1.0)), float(params.get("frequency", 1.0))
            return lambda x, t: a * torch.sin(k * torch.pi * x[:, 0:1])
        raise ValueError(f"{type(self).__name__}: unsupported initial condition type {ic_type!r}")

    def _parse_boundary_conditions(self, config: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Validate the ``boundary_conditions`` block (see the class docstring)."""
        parsed = {}
        for kind, options in config.items():
            options = dict(options or {})
            if kind not in _BOUNDARY_KINDS:
                raise ValueError(
                    f"Unknown boundary condition {kind!r}; expected one of {_BOUNDARY_KINDS}"
                )
            bc_type = options.get("type")
            if kind == "dirichlet":
                if bc_type not in (None, "fixed", "exact"):
                    raise ValueError(f"Unknown dirichlet type {bc_type!r}; use 'fixed' or 'exact'")
                if bc_type == "exact" and not self.config.exact_solution:
                    raise ValueError("dirichlet type 'exact' needs a configured exact_solution")
            elif kind == "neumann" and bc_type not in (None, "zero"):
                raise ValueError(f"Unknown neumann type {bc_type!r}; use 'zero' or a 'value'")
            parsed[kind] = options
        return parsed

    def _seeded_points(self, kind: str, n: int):
        """Fixed, seeded points for the boundary and initial losses.

        They are drawn once per instance, so the loss is deterministic across
        calls (L-BFGS line searches need that).
        """
        key = (kind, n)
        if key not in self._fixed_points:
            gen = torch.Generator().manual_seed(0 if kind == "initial" else 1)
            lo = torch.tensor([float(b[0]) for b in self.domain])
            hi = torch.tensor([float(b[1]) for b in self.domain])
            x = lo + (hi - lo) * torch.rand(n, self.dimension, generator=gen)
            t0, t1 = float(self.time_domain[0]), float(self.time_domain[1])
            if kind == "initial":
                points = (x.to(self.device), torch.full((n, 1), t0, device=self.device))
            else:
                t = (t0 + (t1 - t0) * torch.rand(n, 1, generator=gen)).to(self.device)
                points = []
                for axis in range(self.dimension):
                    x_lo, x_hi = x.clone(), x.clone()
                    x_lo[:, axis], x_hi[:, axis] = lo[axis], hi[axis]
                    points.append((axis, x_lo.to(self.device), x_hi.to(self.device), t))
            self._fixed_points[key] = points
        return self._fixed_points[key]

    def _dirichlet_values(self, options, x_lo, x_hi, t):
        """Target values on the lower and upper face for a dirichlet condition."""
        if options.get("type") == "exact":
            with torch.no_grad():
                return self.exact_solution(x_lo, t), self.exact_solution(x_hi, t)
        value = float(options.get("value", 0.0))
        left = float(options.get("left", value))
        right = float(options.get("right", value))
        return torch.full_like(t, left), torch.full_like(t, right)

    def _boundary_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Mean loss over every configured condition on every face."""
        if not self.boundary_conditions:
            return torch.tensor(0.0, device=self.device)
        n = int(self._training_option("num_boundary_points", 100))
        needs_grad = "neumann" in self.boundary_conditions or "periodic" in self.boundary_conditions
        terms = []
        for axis, x_lo, x_hi, t in self._seeded_points("boundary", n):
            x_lo = x_lo.clone().requires_grad_(needs_grad)
            x_hi = x_hi.clone().requires_grad_(needs_grad)
            u_lo = model(torch.cat([x_lo, t], dim=1))
            u_hi = model(torch.cat([x_hi, t], dim=1))
            if needs_grad:
                du_lo = self._grad(u_lo, x_lo)[:, axis : axis + 1]
                du_hi = self._grad(u_hi, x_hi)[:, axis : axis + 1]
            for kind, options in self.boundary_conditions.items():
                if kind == "dirichlet":
                    g_lo, g_hi = self._dirichlet_values(options, x_lo.detach(), x_hi.detach(), t)
                    terms += [u_lo - g_lo, u_hi - g_hi]
                elif kind == "neumann":
                    g = float(options.get("value", 0.0))
                    terms += [du_lo - g, du_hi - g]
                else:  # periodic
                    terms += [u_lo - u_hi, du_lo - du_hi]
        return torch.stack([self._apply_loss_fn(term) for term in terms]).mean()

    def _initial_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """``u(x, t0) = u0(x)``, plus ``u_t(x, t0) = velocity`` when ``time_order == 2``."""
        if self.initial_condition is None:
            return torch.tensor(0.0, device=self.device)
        n = int(self._training_option("num_initial_points", 100))
        x, t = self._seeded_points("initial", n)
        t = t.clone().requires_grad_(self.time_order == 2)
        u = model(torch.cat([x, t], dim=1))
        loss = self._apply_loss_fn(u - self.initial_condition(x, t.detach()))
        if self.time_order == 2:
            velocity = float(self.config.initial_condition.get("velocity", 0.0))
            loss = loss + self._apply_loss_fn(self._grad(u, t) - velocity)
        return loss

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Compute the PDE residual. Must be implemented by subclasses.

        Args:
            model: Neural network model.
            x: Spatial coordinates, shape ``(N, dimension)``.
            t: Time coordinates, shape ``(N, 1)``.

        Returns:
            Residual tensor, shape ``(N, 1)``.
        """
        raise NotImplementedError("Subclasses must implement compute_residual")

    @staticmethod
    def _grad(f: torch.Tensor, wrt: torch.Tensor) -> torch.Tensor:
        """Pointwise ``df/dwrt`` (shape of ``wrt``); zeros where ``f`` does not depend on it."""
        g = torch.autograd.grad(
            f, wrt, grad_outputs=torch.ones_like(f), create_graph=True, allow_unused=True
        )[0]
        return torch.zeros_like(wrt) if g is None else g

    def laplacian(self, f: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Sum of the pure second derivatives of ``f`` along each column of ``x``.

        ``x`` must be the leaf tensor ``f`` was computed from (``requires_grad``).
        Differentiating with respect to the whole tensor and then taking a column
        is what makes this correct in more than one dimension; a column slice
        ``x[:, d:d+1]`` is not part of the graph.
        """
        grad = self._grad(f, x)
        lap = torch.zeros_like(f)
        for d in range(x.shape[1]):
            lap = lap + self._grad(grad[:, d : d + 1], x)[:, d : d + 1]
        return lap

    def _prepare_inputs(self, model, x: torch.Tensor, t: torch.Tensor):
        """Fresh leaf copies of ``x`` and ``t`` that require grad, and ``u = model([x, t])``."""
        x = x.detach().to(self.device).requires_grad_(True)
        t = t.detach().to(self.device).requires_grad_(True)
        return x, t, model(torch.cat([x, t], dim=1))

    def compute_derivatives(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        temporal_derivatives: list[int] = None,
        spatial_derivatives: set[int] = None,
    ) -> dict[str, torch.Tensor]:
        """Evaluate ``u = model([x, t])`` and the requested derivatives.

        Args:
            model: Neural network model.
            x: Spatial coordinates, shape ``(N, dimension)``.
            t: Time coordinates, shape ``(N, 1)``.
            temporal_derivatives: Orders of ``∂/∂t`` to return (1 and/or 2).
            spatial_derivatives: Orders of the pure ``∂/∂x_d`` derivatives to
                return (1 to 4).

        Returns a dict with ``"u"`` and, for the requested orders only:

        * time: ``"dt"``, ``"dt2"``;
        * space, 1-D: ``"dx"``, ``"dx2"``, ``"dx3"``, ``"dx4"``;
        * space, n-D: pure derivatives per axis, ``"dx1"``, ``"dx1x1"``, ... ``"dx2x2"``;
        * ``"laplacian"`` whenever order 2 is requested.

        Raises:
            ValueError: For temporal orders above 2 or spatial orders above 4.
        """
        temporal = sorted(set(temporal_derivatives or ()) - {0})
        spatial = sorted(set(spatial_derivatives or ()) - {0})
        if temporal and temporal[-1] > 2:
            raise ValueError(
                f"Temporal derivative order {temporal[-1]} is not supported. Maximum order is 2."
            )
        if spatial and spatial[-1] > 4:
            raise ValueError(
                f"Spatial derivative order {spatial[-1]} is not supported. Maximum order is 4."
            )

        x, t, u = self._prepare_inputs(model, x, t)
        derivatives = {"u": u}

        current = u
        for order in range(1, (temporal[-1] if temporal else 0) + 1):
            current = self._grad(current, t)
            if order in temporal:
                derivatives["dt" if order == 1 else f"dt{order}"] = current

        for dim in range(self.dimension):
            current = u
            for order in range(1, (spatial[-1] if spatial else 0) + 1):
                current = self._grad(current, x)[:, dim : dim + 1]
                if order not in spatial:
                    continue
                if self.dimension == 1:
                    key = "dx" if order == 1 else f"dx{order}"
                else:
                    key = "d" + f"x{dim + 1}" * order
                derivatives[key] = current

        if 2 in spatial:
            if self.dimension == 1:
                derivatives["laplacian"] = derivatives["dx2"]
            else:
                derivatives["laplacian"] = sum(
                    derivatives["d" + f"x{d + 1}" * 2] for d in range(self.dimension)
                )
        return derivatives

    @property
    def has_exact_solution(self) -> bool:
        """Whether the config names a closed-form solution."""
        return bool(self.config.exact_solution)

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
        """Closed-form ``u(x, t)``, or ``None`` when ``exact_solution`` is not configured."""
        if not self.has_exact_solution:
            return None
        # Exact solutions start from the IC at t = 0; shift so they hold for any t0.
        return self._exact_solution(x, t - float(self.time_domain[0]))

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            f"{type(self).__name__} has no exact solution of type "
            f"{self.config.exact_solution.get('type')!r}"
        )

    def _sample_uniform(self, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Exactly ``num_points`` points from a jittered space-time grid.

        A regular grid with at least ``num_points`` nodes is built over all
        ``dimension + 1`` axes, ``num_points`` nodes are drawn without
        replacement, and each coordinate gets Gaussian jitter with a standard
        deviation of 1% of its axis length (then is clipped to the domain) to
        break grid alignment.
        """
        bounds = list(self.domain) + [tuple(self.time_domain)]
        lo = torch.tensor([float(b[0]) for b in bounds], device=self.device)
        hi = torch.tensor([float(b[1]) for b in bounds], device=self.device)
        per_axis = max(2, math.ceil(num_points ** (1.0 / len(bounds))))
        axes = [torch.linspace(0.0, 1.0, per_axis, device=self.device)] * len(bounds)
        grid = torch.stack([g.reshape(-1) for g in torch.meshgrid(*axes, indexing="ij")], dim=1)
        unit = grid[torch.randperm(len(grid), device=self.device)[:num_points]]
        unit = torch.clamp(unit + 0.01 * torch.randn_like(unit), 0.0, 1.0)
        points = lo + (hi - lo) * unit
        return points[:, : self.dimension], points[:, -1:]

    def _sample_stratified(self, num_points: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Stratified random sampling (Latin Hypercube-style, pure PyTorch).

        Divides each dimension into num_points equal bins and places one random
        sample per bin, then shuffles across dimensions for independence.
        """
        total_dims = self.dimension + 1  # spatial dims + time

        # Build lower and upper bounds for all dims
        lowers = []
        uppers = []
        for dim in range(self.dimension):
            lowers.append(self.domain[dim][0])
            uppers.append(self.domain[dim][1])
        lowers.append(self.time_domain[0])
        uppers.append(self.time_domain[1])

        samples = torch.zeros(num_points, total_dims, device=self.device)
        for d in range(total_dims):
            lo, hi = lowers[d], uppers[d]
            bin_size = (hi - lo) / num_points
            # One random point inside each bin
            offsets = torch.rand(num_points, device=self.device)
            indices = torch.arange(num_points, dtype=torch.float32, device=self.device)
            samples[:, d] = lo + (indices + offsets) * bin_size
            # Shuffle this dimension independently for LHS property
            perm = torch.randperm(num_points, device=self.device)
            samples[:, d] = samples[perm, d]

        x = samples[:, : self.dimension]
        t = samples[:, -1].reshape(-1, 1)
        return x, t

    def _sample_residual_based(
        self,
        num_points: int,
        model: torch.nn.Module | None = None,
        k: float = 1.0,
        c: float = 0.0,
        pool_factor: int = 4,
        replacement: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Residual-based adaptive distribution (RAD) sampling.

        Draws ``num_points`` from a uniform candidate pool with probability

            p(x) ∝ |r(x)|^k / mean(|r|^k) + c

        (Wu et al., 2023). ``k`` sharpens the focus on high residuals and
        ``c`` mixes back a uniform floor; the defaults ``k=1, c=0`` are
        plain residual-proportional sampling.

        1. Draw a candidate pool of ``pool_factor * num_points`` points with
           :meth:`_sample_uniform`.
        2. Compute the PDE residual on the pool and draw ``num_points`` of
           them according to ``p``.
        3. Fall back to :meth:`_sample_uniform` when no model is given, the
           residual cannot be computed, or the weights are non-finite or all
           zero.

        Args:
            num_points: Number of points to return.
            model: Network whose residual drives the distribution.
            k: Residual exponent.
            c: Uniform floor added to the normalised weights.
            pool_factor: Candidate pool size as a multiple of ``num_points``.
            replacement: Draw with replacement; forced on when ``num_points``
                exceeds the pool.
        """
        if model is None:
            return self._sample_uniform(num_points)

        pool_size = num_points * pool_factor
        x_pool, t_pool = self._sample_uniform(pool_size)

        # Compute PDE residual at each candidate point
        x_pool = x_pool.detach().requires_grad_(True)
        t_pool = t_pool.detach().requires_grad_(True)

        try:
            # compute_residual builds an autograd graph, so we cannot wrap in no_grad.
            residuals = self.compute_residual(model, x_pool, t_pool)
            if isinstance(residuals, tuple):
                residuals = residuals[0]
            # One magnitude per point: the norm over the equations of a system.
            residual_mag = residuals.detach().reshape(x_pool.shape[0], -1).norm(dim=1)
        except Exception:
            return self._sample_uniform(num_points)

        # Convert residual magnitudes to sampling probabilities
        # Add small epsilon so zero-residual regions still have a chance
        weights = residual_mag**k
        if not torch.isfinite(weights).all() or weights.sum() <= 0:
            return self._sample_uniform(num_points)
        probs = weights / weights.mean() + c + 1e-8
        probs = probs / probs.sum()

        # Sample points weighted by residual magnitude
        replacement = replacement or num_points > len(probs)
        selected = torch.multinomial(probs, num_points, replacement=replacement)
        x = x_pool[selected].detach()
        t = t_pool[selected].detach()

        return x, t

    def generate_collocation_points(
        self, num_points: int, strategy: str = "uniform", **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate collocation points for training.

        Args:
            num_points: Number of points to generate.
            strategy: Sampling strategy, one of:

                * ``uniform``: jittered space-time grid (:meth:`_sample_uniform`);
                * ``stratified``: Latin-hypercube-style stratified random;
                * ``residual_based``: RAD (:meth:`_sample_residual_based`),
                  driven by the ``model`` kwarg and the optional ``k``, ``c``,
                  ``pool_factor`` and ``replacement`` kwargs;
                * ``adaptive``: the experimental residual-bandit sampler (1-D
                  only), driven by the ``model`` kwarg and an optional
                  ``progress`` in [0, 1].

                ``residual_based`` and ``adaptive`` fall back to ``uniform``
                when no ``model`` is passed.
            **kwargs: Strategy options listed above.

        Returns:
            ``(x, t)`` with shapes ``(num_points, dimension)`` and ``(num_points, 1)``.

        Raises:
            ValueError: For an unknown strategy, or ``adaptive`` with a model on
                a PDE with more than one spatial dimension.
        """
        if strategy == "uniform":
            x, t = self._sample_uniform(num_points)

        elif strategy == "stratified":
            x, t = self._sample_stratified(num_points)

        elif strategy == "residual_based":
            model = kwargs.get("model", None)
            rad_kwargs = {
                key: kwargs[key]
                for key in ("k", "c", "pool_factor", "replacement")
                if key in kwargs
            }
            x, t = self._sample_residual_based(num_points, model, **rad_kwargs)

        elif strategy == "adaptive":
            model = kwargs.get("model", None)
            if model is None:
                x, t = self._sample_uniform(num_points)
            else:
                if self.dimension != 1:
                    raise ValueError(
                        "The 'adaptive' (RL bandit) sampler supports 1-D PDEs only; "
                        "use 'residual_based' for higher dimensions."
                    )
                if self.bandit_sampler is None:
                    from pinnrl.config import RLConfig

                    self.bandit_sampler = RLConfig().build_sampler(self)
                x, t = self.bandit_sampler.sample(
                    self, model, num_points, progress=kwargs.get("progress", 0.0)
                )
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

        # Ensure the points are on the correct device before returning
        x = x.to(self.device)
        t = t.to(self.device)

        return x, t

    def compute_loss(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Loss components and their weighted total.

        Returns ``residual`` (at the collocation points ``x, t``), ``boundary``,
        ``initial``, ``data`` (observations; zero without any) and ``total``.
        ``training.mode`` selects the objective (see :meth:`weighted_total`):
        ``data_only`` keeps only the data term; every other mode adds the
        weighted physics terms to the data term. Weights come from
        ``training.loss_weights`` (``residual``/``pde``, ``boundary``,
        ``initial``, ``data``; 1, 10, 10 and 1 when a key is absent). With
        adaptive weighting enabled the physics weights here are 1 and the
        trainer re-weights the total.
        """
        losses = {
            "residual": self._apply_loss_fn(self.compute_residual(model, x, t)),
            "boundary": self._boundary_loss(model),
            "initial": self._initial_loss(model),
            "data": self._compute_data_loss(model),
        }

        weights = self._training_option("loss_weights") or {}
        adaptive = self._training_option("adaptive_weights")
        if adaptive is not None and getattr(adaptive, "enabled", False):
            physics_weights = (1.0, 1.0, 1.0)  # the trainer re-weights with its own
        else:
            physics_weights = (
                float(weights.get("pde", weights.get("residual", 1.0))),
                float(weights.get("boundary", 10.0)),
                float(weights.get("initial", 10.0)),
            )
        losses["total"] = self.weighted_total(losses, *physics_weights)
        return losses

    def weighted_total(self, losses, w_residual, w_boundary, w_initial) -> torch.Tensor:
        """Combine loss components under ``training.mode``.

        ``data_only`` drops the physics terms; outside ``forward`` the data term
        always counts (its weight is at least 1).
        """
        mode = self._training_option("mode", "forward")
        physics = 0.0 if mode == "data_only" else 1.0
        data_weight = float((self._training_option("loss_weights") or {}).get("data", 1.0))
        if mode != "forward" and data_weight <= 0.0:
            data_weight = 1.0
        return (
            physics
            * (
                w_residual * losses["residual"]
                + w_boundary * losses["boundary"]
                + w_initial * losses["initial"]
            )
            + data_weight * losses["data"]
        )

    def validate(self, model: torch.nn.Module, num_points: int = 1000) -> dict[str, float]:
        """Error of ``model`` against the exact solution on a jittered grid.

        Returns ``l2_error`` (relative L2, ``||u - u*|| / ||u*||``), ``mse``,
        ``max_error`` and ``mean_error``.

        Raises:
            ValueError: If the PDE has no exact solution configured.
        """
        if not self.has_exact_solution:
            raise ValueError(f"{self.config.name} has no exact solution configured")
        x, t = self.generate_collocation_points(num_points)
        u_exact = self.exact_solution(x, t)
        with torch.no_grad():
            # Every component the exact solution provides (one for scalar PDEs).
            u_pred = model(torch.cat([x, t], dim=1))[:, : u_exact.shape[1]]
        error = u_pred - u_exact
        return {
            "l2_error": (torch.linalg.norm(error) / torch.linalg.norm(u_exact)).item(),
            "mse": torch.mean(error**2).item(),
            "max_error": torch.max(torch.abs(error)).item(),
            "mean_error": torch.mean(torch.abs(error)).item(),
        }
