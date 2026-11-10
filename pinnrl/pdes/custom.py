"""User-defined equations: a PINN for any PDE or ODE written as text or as a Python function.

Example::

    from pinnrl import CustomPDE

    pde = CustomPDE(
        "u_t + u*u_x - nu*u_xx",
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"nu": 0.01 / 3.14159},
        initial_condition="-sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
    )
    trainer = pde.build_trainer("fourier", num_epochs=2000)
    trainer.train()

Systems take a list of equations and ``fields=["u", "v"]``; ODEs a domain with
only ``t``; steady problems a domain without ``t``. The residual can also be a
Python callable ``residual(f, p)`` over a :class:`Fields` accessor and the
parameter dict, for anything the expression language cannot say.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from torch import nn

from .inputs import TimeSeries
from .pde_base import PDEBase, PDEConfig
from .sensors import Sensor, SensorMisfit, sensor_misfit
from .symbolic import CompiledExpression, compile_expression, derivative_name, split_derivative
from .unknowns import UnknownFunction

_DUMMY_COORDINATE = "_x"  # the spatial input an ODE's network ignores


def resolve_input_paths(spec: Mapping[str, Any], base_dir: str | Path) -> dict[str, Any]:
    """A copy of ``spec`` whose relative ``inputs[*].path`` entries are taken from ``base_dir``."""
    spec = dict(spec)
    inputs = spec.get("inputs")
    if isinstance(inputs, Mapping):
        resolved = {}
        for name, entry in inputs.items():
            if (
                isinstance(entry, Mapping)
                and "path" in entry
                and not Path(entry["path"]).is_absolute()
            ):
                entry = {**entry, "path": str(Path(base_dir) / entry["path"])}
            resolved[name] = entry
        spec["inputs"] = resolved
    return spec


Equations = str | Sequence[str] | Callable[["Fields", dict[str, Any]], Any]


class Fields:
    """The network outputs and their derivatives at one batch of points, computed on demand.

    ``f.u`` is the field ``u`` (shape ``(N, 1)``), ``f.u_xx`` its second
    ``x``-derivative, ``f.x`` and ``f.t`` the coordinates; ``f.d("u", "x", "t")``
    is the same as ``f.u_xt``. Each derivative is computed by autograd once
    and cached, so a residual can use ``f.u_x`` and ``f.u_xx`` without paying
    for the first derivative twice.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        fields: Sequence[str],
        coordinates: Sequence[str],
        device: torch.device,
        unknowns: Mapping[str, UnknownFunction] | None = None,
    ):
        """Evaluate ``model`` at ``x, t`` (fresh leaf copies that require grad)."""
        self._x = x.detach().to(device).requires_grad_(True)
        self._t = t.detach().to(device).requires_grad_(True)
        self._fields = tuple(fields)
        self._coordinates = tuple(coordinates)
        self._unknowns: dict[str, UnknownFunction] = dict(unknowns or {})
        self._u = model(torch.cat([self._x, self._t], dim=1))
        self._cache: dict[tuple[str, tuple[str, ...]], torch.Tensor] = {}
        self._spatial = [c for c in self._coordinates if c != "t"]

    @property
    def fields(self) -> tuple[str, ...]:
        """Field names, in output order."""
        return self._fields

    @property
    def coordinates(self) -> tuple[str, ...]:
        """Coordinate names (spatial first, then ``t`` when present)."""
        return self._coordinates

    @property
    def unknowns(self) -> tuple[str, ...]:
        """Names of the unknown functions available as ``f.<name>``."""
        return tuple(self._unknowns)

    @property
    def n_points(self) -> int:
        """Number of points in the batch."""
        return self._x.shape[0]

    def coordinate(self, name: str) -> torch.Tensor:
        """The column of coordinate ``name``, shape ``(N, 1)``."""
        if name == "t":
            return self._t
        i = self._spatial.index(name)
        return self._x[:, i : i + 1]

    def value(self, field: str) -> torch.Tensor:
        """The field ``field`` as predicted by the network, or an unknown function, shape ``(N, 1)``."""
        if field in self._unknowns:
            key = (field, ())
            if key not in self._cache:
                fn = self._unknowns[field]
                self._cache[key] = fn(
                    {
                        name: self.coordinate(name)
                        if name in self._coordinates
                        else self.value(name)
                        for name in fn.inputs
                    }
                )
            return self._cache[key]
        j = self._fields.index(field)
        return self._u[:, j : j + 1]

    def d(self, field: str, *coordinates: str) -> torch.Tensor:
        """``∂^n field / ∂coordinates``, e.g. ``f.d("u", "x", "x")`` for ``u_xx``."""
        key = (field, tuple(sorted(coordinates)))
        if not coordinates:
            return self.value(field)
        if key not in self._cache:
            previous = self.d(field, *key[1][:-1])
            wrt = self._t if key[1][-1] == "t" else self._x
            grad = PDEBase._grad(previous, wrt)
            if key[1][-1] != "t":
                i = self._spatial.index(key[1][-1])
                grad = grad[:, i : i + 1]
            self._cache[key] = grad
        return self._cache[key]

    def laplacian(self, field: str) -> torch.Tensor:
        """Sum of the pure second spatial derivatives of ``field``."""
        return cast(torch.Tensor, sum(self.d(field, c, c) for c in self._spatial))

    def __getattr__(self, name: str) -> torch.Tensor:
        """``f.u``, ``f.u_xx``, ``f.x`` by attribute; anything else is an error."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._fields or name in self._unknowns:
            return self.value(name)
        if name in self._coordinates:
            return self.coordinate(name)
        split = split_derivative(name, self._fields + tuple(self._unknowns), self._coordinates)
        if split is not None:
            return self.d(split[0], *split[1])
        raise AttributeError(
            f"{name!r} is not a field {self._fields}, an unknown function {tuple(self._unknowns)}, "
            f"a coordinate {self._coordinates} or a derivative such as 'u_xx'"
        )

    __getitem__ = __getattr__


def _as_expression_map(spec: Any, fields: Sequence[str], what: str) -> dict[str, str] | None:
    """Normalise an expression spec to ``{field: text}``, or ``None`` when absent."""
    if spec is None:
        return None
    if isinstance(spec, str):
        if len(fields) != 1:
            raise ValueError(
                f"{what}: one expression given for {len(fields)} fields {tuple(fields)}"
            )
        return {fields[0]: spec}
    if isinstance(spec, Mapping):
        unknown = set(spec) - set(fields)
        if unknown:
            raise ValueError(
                f"{what}: unknown fields {sorted(unknown)}; fields are {tuple(fields)}"
            )
        return {f: str(spec[f]) for f in fields if f in spec}
    spec = list(spec)
    if len(spec) != len(fields):
        raise ValueError(f"{what}: {len(spec)} expressions for {len(fields)} fields")
    return dict(zip(fields, (str(s) for s in spec), strict=True))


class CustomPDE(PDEBase):
    """A PDE, ODE or system defined by its residual expressions.

    Args:
        equations: One expression (``"u_t - alpha*u_xx"``), a list of them for
            a system (one per field), or a callable ``residual(f, p)`` taking a
            :class:`Fields` accessor and the parameter dict and returning a
            tensor ``(N, k)`` or a sequence of ``(N, 1)`` tensors.
        domain: ``{coordinate: (lo, hi)}``. Spatial coordinates come first in
            the network input, in the order given; ``"t"`` is the time axis.
            A domain with only ``"t"`` is an ODE (the network gets a dummy
            spatial input it learns to ignore); one without ``"t"`` is a
            steady problem (no initial condition; ``t`` is fixed at 0).
        fields: Output names, default ``("u",)``.
        parameters: Name to value for every parameter the expressions use.
        initial_condition: ``u(x, t0)``: an expression in the spatial
            coordinates (and parameters), a list or ``{field: expression}``
            for systems, or a callable ``u0(x) -> (N, k)``. ``None`` means no
            initial term.
        initial_velocity: ``u_t(x, t0)`` in the same forms, used when an
            equation is second order in time (default 0).
        boundary_conditions: As for :class:`PDEBase` (``dirichlet``,
            ``neumann``, ``periodic``), applied to every field; or, for a
            system, ``{field: {kind: options}}`` with its own conditions per
            field (a field left out has none).
        exact_solution: ``u(x, t)`` in the same forms as the initial
            condition (expressions may use ``t``), or a callable
            ``u(x, t) -> (N, k)``; enables :meth:`validate`.
        name: Display name for runs and the dashboard.
        trainable_parameters: Parameters to identify in inverse mode.
        parameter_initial_guesses: Their starting values.
        inputs: Measured inputs, ``{name: TimeSeries | spec}`` (see
            :class:`~pinnrl.pdes.inputs.TimeSeries`): a signal of ``t`` that
            the equations and boundary values may use by name, read at the
            time of each point. Boundary values (``dirichlet`` ``value`` /
            ``left`` / ``right``, ``neumann`` ``value``) may also be
            expressions in the coordinates, parameters and inputs.
        parametric: ``{parameter: (lo, hi)}``: parameters the network takes
            as extra inputs, sampled uniformly over the range with the
            collocation points, so one training gives a surrogate
            ``u(x, t; theta)`` over the whole prior range (a parametric
            surrogate, K45). They are used in the expressions like
            coordinates; boundary conditions are not applied on their faces.
            :meth:`surrogate_misfit` evaluates the data misfit of the trained
            surrogate for a batch of parameter values, the forward model a
            swarm or a sampler needs.
        sensors: Observation operators (see :mod:`pinnrl.pdes.sensors`):
            :class:`~pinnrl.pdes.sensors.PointSensor`,
            :class:`~pinnrl.pdes.sensors.BoundarySensor`,
            :class:`~pinnrl.pdes.sensors.IntegralSensor` or their specs. Each
            sensor's records enter the data loss as its own weighted term;
            :meth:`observe` fills them from the exact solution.
        scales: ``{field: scale}`` or ``{field: (offset, scale)}``, the
            magnitude (and typical value) of each field in its physical units,
            e.g. ``{"T": (300.0, 50.0)}`` for a temperature around 300 K
            varying by tens of kelvin. With ``build_trainer(normalize=True)``
            the network predicts ``offset + scale * net(x̂)`` over inputs
            mapped to ``[-1, 1]``, so a problem written in metres, seconds and
            kelvin trains like one on the unit square.
        field_weights: ``{field: weight}`` for systems: the field's residual,
            initial and boundary errors enter the loss multiplied by this
            weight (default 1), so a slow or small component can be given
            more say than its scale would.
        unknown_functions: Field-valued unknowns, ``{name: spec}`` with the
            keys of :class:`~pinnrl.pdes.unknowns.UnknownFunction` (``inputs``,
            ``hidden``, ``positive``, ``bounds``, ``monotone``, ``initial``,
            ``truth``, ...) or ready-built instances. The name is used in the
            expressions like a parameter (``"u_t - alpha*u_xx"``, with
            ``alpha_x`` for its derivative). Their weights are trained in
            inverse mode; the ``truth`` serves synthetic data and grading.
        observation_data: Observations for the data term (see :class:`PDEConfig`).
        training: A ``TrainingConfig`` or dict of training options.
        device: Torch device.
        time_order: Override for the order in time (derived from the
            expressions when they are text; needed for a callable residual
            that is second order in time).
        architecture: Default architecture for :meth:`build_trainer`.

    Raises:
        ValueError: For a parameter the expressions use but ``parameters`` does
            not define, an expression that does not parse, or an inconsistent spec.
    """

    def __init__(
        self,
        equations: Equations,
        *,
        domain: Mapping[str, Sequence[float]],
        fields: Sequence[str] = ("u",),
        parameters: Mapping[str, float] | None = None,
        initial_condition: Any = None,
        initial_velocity: Any = None,
        boundary_conditions: Mapping[str, Mapping[str, Any]] | None = None,
        exact_solution: Any = None,
        name: str = "Custom PDE",
        trainable_parameters: Sequence[str] = (),
        parameter_initial_guesses: Mapping[str, float] | None = None,
        unknown_functions: Mapping[str, Mapping[str, Any] | UnknownFunction] | None = None,
        field_weights: Mapping[str, float] | None = None,
        scales: Mapping[str, float | Sequence[float]] | None = None,
        inputs: Mapping[str, Any] | None = None,
        sensors: Sequence[Sensor | Mapping[str, Any]] | None = None,
        parametric: Mapping[str, Sequence[float]] | None = None,
        observation_data: Mapping[str, Any] | None = None,
        training: Any = None,
        device: torch.device | str | None = None,
        time_order: int | None = None,
        architecture: str = "fourier",
        **_ignored: Any,
    ):
        """Build the problem; see the class docstring for the arguments."""
        self.fields = tuple(fields)
        if not self.fields:
            raise ValueError("fields must name at least one network output")
        domain = {str(k): (float(v[0]), float(v[1])) for k, v in domain.items()}
        self.parametric = tuple(str(k) for k in (parametric or {}))
        for pname, rng in dict(parametric or {}).items():
            if pname in domain or pname in (parameters or {}):
                raise ValueError(f"parametric {pname!r} is already a coordinate or a parameter")
            domain[str(pname)] = (float(rng[0]), float(rng[1]))
        if "t" in domain and self.parametric:  # keep time last in the network input
            domain["t"] = domain.pop("t")
        self._n_physical = len([k for k in domain if k != "t"]) - len(self.parametric)
        self.spatial_coordinates = tuple(k for k in domain if k != "t")
        self.time_dependent = "t" in domain
        self.is_ode = not self.spatial_coordinates
        if self.is_ode:
            self.spatial_coordinates = (_DUMMY_COORDINATE,)
            domain = {_DUMMY_COORDINATE: (0.0, 1.0), **domain}
        self.coordinates = self.spatial_coordinates + (("t",) if self.time_dependent else ())
        self.parameters = {str(k): float(v) for k, v in (parameters or {}).items()}
        self._architecture = architecture
        self.unknown_functions = nn.ModuleDict()
        for uname, spec in dict(unknown_functions or {}).items():
            uname = str(uname)
            if uname in self.parameters or uname in self.fields or uname in self.coordinates:
                raise ValueError(
                    f"unknown function {uname!r} clashes with a parameter, field or coordinate"
                )
            if isinstance(spec, UnknownFunction):
                fn = spec
            else:
                fn = UnknownFunction.from_spec(uname, spec, coordinate_ranges=domain)
            for inp in fn.inputs:
                if inp not in self.coordinates and inp not in self.fields:
                    raise ValueError(
                        f"input {inp!r} of unknown function {uname!r} is neither a coordinate "
                        f"{self.coordinates} nor a field {self.fields}"
                    )
            self.unknown_functions[uname] = fn
        self._unknown_names = tuple(self.unknown_functions.keys())
        self.inputs: dict[str, TimeSeries] = {}
        for iname, spec in dict(inputs or {}).items():
            iname = str(iname)
            taken = (
                set(self.parameters)
                | set(self.fields)
                | set(self.coordinates)
                | set(self._unknown_names)
            )
            if iname in taken:
                raise ValueError(
                    f"input {iname!r} clashes with a parameter, field, coordinate or unknown"
                )
            if not self.time_dependent:
                raise ValueError(f"input {iname!r}: a time series needs a time-dependent problem")
            self.inputs[iname] = TimeSeries.from_spec(iname, spec)
        self._bc_expressions: dict[str, CompiledExpression] = {}
        self._observables: dict[str, CompiledExpression] = {}
        self.sensors: list[Sensor] = [
            s if isinstance(s, Sensor) else Sensor.from_spec(dict(s)) for s in (sensors or [])
        ]
        self._spec = {
            "equations": equations if not callable(equations) else None,
            "domain": {
                k: list(v)
                for k, v in domain.items()
                if k != _DUMMY_COORDINATE and k not in self.parametric
            },
            "fields": list(self.fields),
            "parameters": dict(self.parameters),
            "initial_condition": None if callable(initial_condition) else initial_condition,
            "initial_velocity": None if callable(initial_velocity) else initial_velocity,
            "boundary_conditions": copy.deepcopy(dict(boundary_conditions or {})),
            "exact_solution": None if callable(exact_solution) else exact_solution,
            "name": name,
            "time_order": time_order,
            "architecture": architecture,
            "unknown_functions": {
                k: cast(UnknownFunction, v).to_spec() for k, v in self.unknown_functions.items()
            },
            "field_weights": dict(field_weights or {}),
            "parametric": {k: list(domain[k]) for k in self.parametric},
            "inputs": {k: v.to_spec() for k, v in self.inputs.items()},
            "sensors": [],  # filled by to_dict, so records written after construction are kept
        }
        self.field_scales: dict[str, tuple[float, float]] = {}
        for fname, value in dict(scales or {}).items():
            if fname not in self.fields:
                raise ValueError(f"scales for {fname!r}, which is not a field {self.fields}")
            offset, scale = (
                (0.0, float(value))
                if isinstance(value, (int, float))
                else (float(value[0]), float(value[1]))
            )
            if scale <= 0:
                raise ValueError(f"scale of {fname!r} must be positive")
            self.field_scales[fname] = (offset, scale)
        self._spec["scales"] = {k: list(v) for k, v in self.field_scales.items()}
        unknown_weights = sorted(set(field_weights or {}) - set(self.fields))
        if unknown_weights:
            raise ValueError(
                f"field_weights for {unknown_weights}, which are not fields {self.fields}"
            )
        self._field_weights = [float((field_weights or {}).get(f, 1.0)) for f in self.fields]
        self._per_field_bcs: dict[str, dict[str, Any]] | None = None
        bcs = dict(boundary_conditions or {})
        if bcs and all(k in self.fields for k in bcs):
            self._per_field_bcs = {f: dict(bcs[f] or {}) for f in bcs}
        elif any(k in self.fields for k in bcs):
            raise ValueError(
                "boundary_conditions mixes field names with condition kinds; give either "
                f"{{kind: options}} for every field or {{field: {{kind: options}}}}: {sorted(bcs)}"
            )

        # Residual: compiled expressions or the user's callable.
        self._residual_fn: Callable | None = None
        self._equations: list[CompiledExpression] = []
        if callable(equations):
            self._residual_fn = equations
            self._needed: set[str] = set()
        else:
            texts = [equations] if isinstance(equations, str) else list(equations)
            if len(texts) != len(self.fields):
                raise ValueError(
                    f"{len(texts)} equation(s) for {len(self.fields)} field(s) {self.fields}; "
                    "give one equation per field"
                )
            symbols = self.fields + self._unknown_names
            self._equations = [
                compile_expression(t, symbols, self.coordinates, self.parameters) for t in texts
            ]
            self._needed = {p for e in self._equations for p in e.parameters}
        self._ic = self._compile_map(initial_condition, "initial_condition", with_time=False)
        self._ic_fn = initial_condition if callable(initial_condition) else None
        self._velocity = self._compile_map(initial_velocity, "initial_velocity", with_time=False)
        self._velocity_fn = initial_velocity if callable(initial_velocity) else None
        self._exact = self._compile_map(exact_solution, "exact_solution", with_time=True)
        self._exact_fn = exact_solution if callable(exact_solution) else None
        for compiled in list((self._ic or {}).values()) + list((self._velocity or {}).values()):
            self._needed |= set(compiled.parameters)
        if self._exact:
            self._needed |= {p for e in self._exact.values() for p in e.parameters}
        missing = sorted(
            self._needed - set(self.parameters) - set(self._unknown_names) - set(self.inputs)
        )
        if missing:
            raise ValueError(
                f"Parameters {missing} appear in the equations but are not in parameters="
                f"{sorted(self.parameters)}; pass a value for each (or list it in "
                "trainable_parameters with an initial guess)"
            )

        if time_order is None:
            orders = [e.time_order for e in self._equations]
            time_order = max(orders, default=1) if self.time_dependent else 0
        self.time_order = max(int(time_order), 1) if self.time_dependent else 0

        has_ic = initial_condition is not None and self.time_dependent
        config = PDEConfig(
            name=name,
            domain=[domain[c] for c in self.spatial_coordinates],
            time_domain=domain["t"] if self.time_dependent else (0.0, 0.0),
            parameters=dict(self.parameters),
            boundary_conditions=copy.deepcopy(dict(boundary_conditions or {}))
            if self._per_field_bcs is None
            else {},
            initial_condition={"type": "custom"} if has_ic else {},
            exact_solution={"type": "custom"} if exact_solution is not None else {},
            dimension=len(self.spatial_coordinates),
            input_dim=len(self.spatial_coordinates) + 1,
            output_dim=len(self.fields),
            architecture=architecture,
            device=torch.device(device) if device is not None else None,
            training=training,
            trainable_parameters=list(trainable_parameters),
            parameter_initial_guesses=dict(parameter_initial_guesses or {}),
            observation_data=dict(observation_data) if observation_data else None,
        )
        super().__init__(config)
        self.unknown_functions.to(self.device)
        for sensor in self.sensors:
            sensor.bind(self)
            pending = getattr(sensor, "_pending_values", None)
            if pending is not None:
                sensor.set_values(pending)
                del sensor._pending_values
        if self._per_field_bcs is not None:
            self.config.boundary_conditions = copy.deepcopy(bcs)
            self.boundary_conditions = {
                f: self._parse_boundary_conditions(c) for f, c in self._per_field_bcs.items()
            }

    # ----------------------------------------------------------------- building
    def _compile_map(
        self, spec: Any, what: str, with_time: bool
    ) -> dict[str, CompiledExpression] | None:
        if spec is None or callable(spec):
            return None
        coords = self.coordinates if with_time else self.spatial_coordinates
        texts = _as_expression_map(spec, self.fields, what)
        assert texts is not None  # only a None spec maps to None
        compiled = {}
        for field, text in texts.items():
            expr = compile_expression(text, (), coords, self.parameters)
            if expr.parameters and set(expr.parameters) & set(self.fields):
                raise ValueError(f"{what} for {field!r} must not depend on the fields: {text!r}")
            for uname in set(expr.parameters) & set(self._unknown_names):
                fn = cast(UnknownFunction, self.unknown_functions[uname])
                if not fn.has_truth:
                    raise ValueError(
                        f"{what} for {field!r} uses the unknown function {uname!r}, which needs a "
                        f"truth to be evaluated there: {text!r}"
                    )
                if any(i not in coords for i in fn.inputs):
                    raise ValueError(
                        f"{what} for {field!r} uses {uname!r}, whose inputs {fn.inputs} are not all "
                        f"coordinates of that expression {tuple(coords)}"
                    )
            compiled[field] = expr
        return compiled

    @classmethod
    def from_dict(cls, spec: Mapping[str, Any], **overrides: Any) -> CustomPDE:
        """Build from a plain dict (the YAML layout of :meth:`to_dict`); ``overrides`` win."""
        spec = {**spec, **overrides}
        equations = spec.pop("equations", None) or spec.pop("equation", None)
        if equations is None:
            raise ValueError("A custom PDE spec needs an 'equations' entry")
        return cls(equations, **spec)

    @classmethod
    def from_yaml(cls, path: str, **overrides: Any) -> CustomPDE:
        """Build from a YAML file with the layout of :meth:`to_dict`."""
        with open(path) as f:
            spec = yaml.safe_load(f) or {}
        return cls.from_dict(resolve_input_paths(spec, Path(path).parent), **overrides)

    def to_dict(self) -> dict[str, Any]:
        """The textual spec that rebuilds this problem (callables are left out as ``None``)."""
        spec = copy.deepcopy(self._spec)
        spec["sensors"] = [s.to_spec() for s in self.sensors]
        return spec

    def build_trainer(
        self,
        architecture: str | None = None,
        config_path: str | None = None,
        device: torch.device | str | None = None,
        model: Mapping[str, Any] | None = None,
        normalize: bool = False,
        **training: Any,
    ):
        """A :class:`~pinnrl.training.trainer.PDETrainer` for this problem with packaged defaults.

        ``normalize=True`` non-dimensionalizes the network: its inputs are
        mapped from the domain to ``[-1, 1]`` and its outputs are
        ``offset + scale * net`` with the ``scales`` of the problem (1 for a
        field without one). The equations stay in physical units.

        ``architecture`` names an entry of the ``architectures:`` block of
        ``config.yaml`` (``config_path`` for your own file); ``model``
        overrides keys of that entry (``hidden_dims``, ``activation``,
        ``layer_norm``, ``scale``, ...). ``training`` overrides keys of the
        ``training:`` block (``num_epochs``, ``batch_size``,
        ``num_collocation_points``, ``learning_rate``, ``weight_decay``,
        ``mode``, ``loss_weights``, ...).
        """
        from pinnrl.config import DEFAULT_CONFIG_PATH, Config, ModelConfig, TrainingConfig
        from pinnrl.neural_networks import PINNModel
        from pinnrl.training.trainer import PDETrainer

        with open(config_path or DEFAULT_CONFIG_PATH) as f:
            defaults = yaml.safe_load(f)
        training_cfg = copy.deepcopy(defaults.get("training", {}))
        for key, value in training.items():
            if key in ("learning_rate", "weight_decay"):
                training_cfg.setdefault("optimizer_config", {})[key] = value
            elif isinstance(value, Mapping) and isinstance(training_cfg.get(key), Mapping):
                training_cfg[key] = {**training_cfg[key], **value}
            else:
                training_cfg[key] = value
        if (self.config.trainable_parameters or self._unknown_names) and "mode" not in training:
            training_cfg["mode"] = "inverse"
        architecture = architecture or self._architecture
        model = dict(model or {})
        self._normalized = bool(normalize)
        if normalize:
            model.setdefault("input_bounds", self.normalization_bounds())
            model.setdefault(
                "output_offset", [self.field_scales.get(f, (0.0, 1.0))[0] for f in self.fields]
            )
            model.setdefault(
                "output_scale", [self.field_scales.get(f, (0.0, 1.0))[1] for f in self.fields]
            )
        device = torch.device(device) if device is not None else self.device
        input_dim, output_dim = self.config.input_dim, self.config.output_dim
        assert input_dim is not None and output_dim is not None  # set by PDEBase.__init__
        config = Config.__new__(Config)
        config.device = device
        config.training = TrainingConfig.from_dict(training_cfg)
        config.model = ModelConfig.from_architecture(
            architecture,
            {**defaults.get("architectures", {}).get(architecture, {}), **model},
            input_dim=input_dim,
            output_dim=output_dim,
        )
        config.rl = None
        self.config.training = config.training
        self.device = device
        self.config.device = device
        self.unknown_functions.to(device)
        network = PINNModel(config=config, device=device).to(device)
        return PDETrainer(model=network, pde=self, config=config, device=device)

    # ------------------------------------------------------------------ physics
    def _parameter_values(self) -> dict[str, Any]:
        """Parameter name to value: the live ``nn.Parameter`` for trainable ones."""
        return {name: self.get_parameter(name) for name in self.parameters}

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Residuals of every equation at ``x, t``, shape ``(N, k)``."""
        f = Fields(
            model,
            x,
            t,
            self.fields,
            self.coordinates,
            self.device,
            cast("Mapping[str, UnknownFunction]", self.unknown_functions),
        )
        params = self._parameter_values()
        if self.inputs:
            params.update(self._input_values(f.coordinate("t")))
        if self._residual_fn is not None:
            out = self._residual_fn(f, params)
            if not isinstance(out, torch.Tensor):
                out = torch.cat([o.reshape(f.n_points, -1) for o in out], dim=1)
            return out.reshape(f.n_points, -1)
        like = f.value(self.fields[0])
        columns = []
        for expr in self._equations:
            values: dict[str, Any] = dict(params)
            for field in expr.fields:
                values[field] = f.value(field)
            for field, coords in expr.derivatives:
                values[derivative_name(field, coords)] = f.d(field, *coords)
            for c in expr.coordinates:
                values[c] = f.coordinate(c)
            columns.append(expr(values, like))
        return torch.cat(columns, dim=1)

    def _evaluate_map(
        self,
        compiled: dict[str, CompiledExpression] | None,
        fn: Callable | None,
        x: torch.Tensor,
        t: torch.Tensor | None,
    ) -> torch.Tensor:
        """Evaluate an expression map (or callable) to ``(N, k)``; absent fields give 0."""
        if fn is not None:
            out = fn(x) if t is None else fn(x, t)
            return out.reshape(x.shape[0], -1)
        like = x[:, 0:1]
        values: dict[str, Any] = self._parameter_values()
        if t is not None:
            values.update(self._input_values(t))
        for i, c in enumerate(self.spatial_coordinates):
            values[c] = x[:, i : i + 1]
        if t is not None:
            values["t"] = t
        needed = {p for e in (compiled or {}).values() for p in e.parameters}
        for uname in needed & set(self._unknown_names):
            unknown = cast(UnknownFunction, self.unknown_functions[uname])
            values[uname] = unknown.truth({i: values[i].expand_as(like) for i in unknown.inputs})
        columns = []
        for field in self.fields:
            expr = (compiled or {}).get(field)
            columns.append(expr(values, like) if expr is not None else torch.zeros_like(like))
        return torch.cat(columns, dim=1)

    def _create_initial_condition(self, params: dict[str, Any]) -> Callable | None:
        if not params:
            return None
        return lambda x, t: self._evaluate_map(self._ic, self._ic_fn, x, None)

    def _multipliers(self, kind: str) -> list[float]:
        """Per-field multiplier of an ``error`` (IC, BC, data) or ``residual`` column in the loss.

        The root of the field weight, and, when the problem is normalized
        (``build_trainer(normalize=True)``), divided by the field's scale so
        the error is in units of the field's magnitude; a residual is further
        multiplied by the time span (the time scale of ``u_t``) or, for a
        steady problem, by the square of the first axis's length.
        """
        mult = [w**0.5 for w in self._field_weights]
        if not getattr(self, "_normalized", False):
            return mult
        if self.time_dependent:
            t0, t1 = float(self.time_domain[0]), float(self.time_domain[1])
            ref = t1 - t0 if t1 > t0 else 1.0
        else:
            lo, hi = self.domain[0]
            ref = (hi - lo) ** 2
        out = []
        for f, m in zip(self.fields, mult, strict=True):
            scale = self.field_scales.get(f, (0.0, 1.0))[1]
            out.append(m / scale * (ref if kind == "residual" else 1.0))
        return out

    def _weighted(self, error: torch.Tensor, kind: str = "error") -> torch.Tensor:
        """``error`` ``(N, k)`` with each field's column scaled by :meth:`_multipliers`."""
        mult = self._multipliers(kind)
        if all(m == 1.0 for m in mult):
            return error
        w = torch.tensor(mult, device=error.device, dtype=error.dtype)
        return error * w[: error.shape[1]]

    def compute_loss(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """As :meth:`PDEBase.compute_loss`, with the residual columns weighted by ``field_weights``."""
        losses = super().compute_loss(model, x, t)
        if any(m != 1.0 for m in self._multipliers("residual")):
            losses["residual"] = self._apply_loss_fn(
                self._weighted(self.compute_residual(model, x, t), "residual")
            )
            weights = self._training_option("loss_weights") or {}
            adaptive = self._training_option("adaptive_weights")
            if adaptive is not None and getattr(adaptive, "enabled", False):
                physics = (1.0, 1.0, 1.0)
            else:
                physics = (
                    float(weights.get("pde", weights.get("residual", 1.0))),
                    float(weights.get("boundary", 10.0)),
                    float(weights.get("initial", 10.0)),
                )
            losses["total"] = self.weighted_total(losses, *physics)
        return losses

    def evaluate_expression(
        self,
        text: str,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor | None,
        parameters: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """An expression of the fields (``"-alpha*u_x"``) evaluated with ``model`` at ``x, t``.

        The same language as the equations: fields and their derivatives (by
        autograd), coordinates, parameters, unknown functions and inputs. Used
        by sensors and handy for fluxes and other derived quantities.
        ``parameters`` overrides parameter values (tensors allowed), which the
        sensitivity analysis uses to differentiate through the expression.
        """
        if text not in self._observables:
            expr = compile_expression(
                text, self.fields + self._unknown_names, self.coordinates, self.parameters
            )
            unknown = set(expr.parameters) - set(self.parameters) - set(self.inputs)
            if unknown:
                raise ValueError(f"expression {text!r} uses unknown names {sorted(unknown)}")
            self._observables[text] = expr
        expr = self._observables[text]
        if t is None:
            t = torch.zeros(x.shape[0], 1, device=self.device, dtype=x.dtype)
        f = Fields(
            model,
            x,
            t,
            self.fields,
            self.coordinates,
            self.device,
            cast("Mapping[str, UnknownFunction]", self.unknown_functions),
        )
        values: dict[str, Any] = self._parameter_values()
        values.update(parameters or {})
        if self.inputs:
            values.update(self._input_values(f.coordinate("t")))
        for field in expr.fields:
            values[field] = f.value(field)
        for field, coords in expr.derivatives:
            values[derivative_name(field, coords)] = f.d(field, *coords)
        for c in expr.coordinates:
            values[c] = f.coordinate(c)
        return expr(values, f.value(self.fields[0]))

    def sensor_misfit(self, model: torch.nn.Module, level: float = 0.999) -> SensorMisfit:
        """The chi-square of ``model`` against the sensor records, with its acceptance limit.

        See :class:`~pinnrl.pdes.sensors.SensorMisfit`; the check a trained
        twin can make on itself, and the test :func:`~pinnrl.training.restarts.train_with_restarts`
        restarts on.
        """
        return sensor_misfit(self.sensors, model, level)

    def observe(self, noise: float | None = None, seed: int | None = 0) -> list[Sensor]:
        """Fill every sensor's records from the exact solution, through its operator, with noise.

        ``noise`` overrides each sensor's own standard deviation; the seed is
        offset per sensor so records are independent. Needs an exact solution.
        """
        if not self.has_exact_solution:
            raise ValueError(f"{self.config.name}: observe() needs a configured exact_solution")
        pde = self

        class _Exact(torch.nn.Module):
            def forward(self, inp):
                d = len(pde.spatial_coordinates)
                return pde.exact_solution(inp[:, :d], inp[:, d:])

        exact = _Exact()
        # Read the TRUE parameters, not the current guesses (as generate_synthetic_observations).
        saved = self._trainable_params
        try:
            if self._true_parameters:
                self._trainable_params = nn.ParameterDict()
            for k, sensor in enumerate(self.sensors):
                sensor.observe(
                    exact, noise=noise, seed=None if seed is None else int(seed) + 1000 * k
                )
        finally:
            self._trainable_params = saved
        return self.sensors

    def _compute_data_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Observation data (if any, in units of the field scales) plus one term per sensor."""
        obs = getattr(self, "observation_data", None)
        cutoff = getattr(self, "time_cutoff", None)
        if obs and cutoff is not None:  # time-marching: only the records inside the window
            keep = (obs["t"].reshape(-1) <= cutoff).nonzero().reshape(-1)
            obs = {k: v[keep] for k, v in obs.items() if isinstance(v, torch.Tensor)}
            if keep.numel() == 0:
                obs = None
        if obs and (getattr(self, "_normalized", False) or cutoff is not None):
            pred = model(torch.cat([obs["x"], obs["t"]], dim=1))[:, : obs["u"].shape[1]]
            loss = self._apply_loss_fn(self._weighted(pred - obs["u"]))
        elif obs:
            loss = super()._compute_data_loss(model)
        else:
            loss = torch.tensor(0.0, device=self.device)
        for sensor in self.sensors:
            loss = loss + sensor.loss(model, self._apply_loss_fn, time_cutoff=cutoff)
        return loss

    def _input_values(self, t: torch.Tensor) -> dict[str, torch.Tensor]:
        """Every measured input read at the times ``t`` (``(N, 1)``)."""
        return {name: series(t) for name, series in self.inputs.items()}

    def _boundary_target(self, value: Any, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """A boundary value, a number or an expression in coordinates, parameters and inputs."""
        if not isinstance(value, str):
            return torch.full_like(t, float(value))
        if value not in self._bc_expressions:
            expr = compile_expression(value, (), self.coordinates, self.parameters)
            unknown = set(expr.parameters) - set(self.parameters) - set(self.inputs)
            if unknown:
                raise ValueError(
                    f"boundary value {value!r} uses {sorted(unknown)}, which are neither "
                    f"parameters {sorted(self.parameters)} nor inputs {sorted(self.inputs)}"
                )
            self._bc_expressions[value] = expr
        values: dict[str, Any] = self._parameter_values()
        values.update(self._input_values(t))
        for i, c in enumerate(self.spatial_coordinates):
            values[c] = x[:, i : i + 1]
        values["t"] = t
        return self._bc_expressions[value](values, t)

    def _dirichlet_values(self, options, x_lo, x_hi, t):
        """Targets on the two faces; numbers or expressions (``"sin(2*pi*t)"``, an input's name)."""
        if options.get("type") == "exact":
            return super()._dirichlet_values(options, x_lo, x_hi, t)
        # One-sided: ``{left: ...}`` alone (no ``value``, no ``right``) constrains the lower face
        # only, the inlet of a transport problem; the other face is left free (``None``).
        one_sided = "value" not in options and ("left" in options) != ("right" in options)
        value = options.get("value", 0.0)
        left = options.get("left", None if one_sided else value)
        right = options.get("right", None if one_sided else value)
        return (
            None if left is None else self._boundary_target(left, x_lo, t),
            None if right is None else self._boundary_target(right, x_hi, t),
        )

    def _boundary_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """Mean loss over every condition on every face, per field when the conditions are."""
        if self._per_field_bcs is None:
            conditions = {f: self.boundary_conditions for f in self.fields}
        else:
            conditions = self.boundary_conditions
        if not any(conditions.values()):
            return torch.tensor(0.0, device=self.device)
        n = int(self._training_option("num_boundary_points", 100))
        needs_grad = any(
            k in ("neumann", "periodic") for conds in conditions.values() for k in conds
        )
        terms = []
        for axis, x_lo, x_hi, t in self._seeded_points("boundary", n):
            if axis >= self._n_physical:  # no conditions on the faces of a parametric axis
                continue
            x_lo = x_lo.clone().requires_grad_(needs_grad)
            x_hi = x_hi.clone().requires_grad_(needs_grad)
            u_lo = model(torch.cat([x_lo, t], dim=1))
            u_hi = model(torch.cat([x_hi, t], dim=1))
            for j, field in enumerate(self.fields):
                conds = conditions.get(field, {})
                if not conds:
                    continue
                w = self._multipliers("error")[j]
                f_lo, f_hi = u_lo[:, j : j + 1], u_hi[:, j : j + 1]
                if needs_grad:
                    df_lo = self._grad(f_lo, x_lo)[:, axis : axis + 1]
                    df_hi = self._grad(f_hi, x_hi)[:, axis : axis + 1]
                for kind, options in conds.items():
                    if kind == "dirichlet":
                        g_lo, g_hi = self._dirichlet_values(
                            options, x_lo.detach(), x_hi.detach(), t
                        )
                        if g_lo is not None and g_lo.shape[1] > 1:  # type 'exact': every field
                            g_lo, g_hi = g_lo[:, j : j + 1], g_hi[:, j : j + 1]
                        if g_lo is not None:
                            terms.append(w * (f_lo - g_lo))
                        if g_hi is not None:
                            terms.append(w * (f_hi - g_hi))
                    elif kind == "neumann":
                        g_lo = self._boundary_target(options.get("value", 0.0), x_lo.detach(), t)
                        g_hi = self._boundary_target(options.get("value", 0.0), x_hi.detach(), t)
                        terms += [w * (df_lo - g_lo), w * (df_hi - g_hi)]
                    else:  # periodic
                        terms += [w * (f_lo - f_hi), w * (df_lo - df_hi)]
        return torch.stack([self._apply_loss_fn(term) for term in terms]).mean()

    def _initial_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """``u(x, t0) = u0(x)``, plus ``u_t(x, t0) = v0(x)`` for second order in time."""
        if self.initial_condition is None:
            return torch.tensor(0.0, device=self.device)
        n = int(self._training_option("num_initial_points", 100))
        x, t = self._seeded_points("initial", n)
        t = t.clone().requires_grad_(self.time_order >= 2)
        u = model(torch.cat([x, t], dim=1))
        loss = self._apply_loss_fn(self._weighted(u - self.initial_condition(x, t.detach())))
        if self.time_order >= 2:
            velocity = self._evaluate_map(self._velocity, self._velocity_fn, x, None)
            u_t = torch.cat([self._grad(u[:, j : j + 1], t) for j in range(u.shape[1])], dim=1)
            loss = loss + self._apply_loss_fn(self._weighted(u_t - velocity))
        return loss

    def _exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if self._exact is None and self._exact_fn is None:
            return super()._exact_solution(x, t)
        return self._evaluate_map(self._exact, self._exact_fn, x, t)

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor | None:
        """Closed-form ``u(x, t)`` as ``(N, k)``, or ``None`` without one."""
        if not self.has_exact_solution:
            return None
        return self._exact_solution(x, t)

    def surrogate_misfit(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        thetas: Any,
        field: int = 0,
    ) -> torch.Tensor:
        """Sum of squared residuals of the parametric surrogate against data, per parameter value.

        ``x`` ``(N, d)`` holds the physical coordinates of the observations,
        ``t`` ``(N, 1)`` their times and ``y`` ``(N, 1)`` their values;
        ``thetas`` is ``(M, p)`` over the parametric axes in their declared
        order. Returns ``(M,)``, the vectorized misfit a swarm consumes.
        """
        if not self.parametric:
            raise ValueError("surrogate_misfit needs a problem with parametric axes")
        thetas_t = torch.as_tensor(
            np.asarray(thetas, dtype=np.float64), dtype=x.dtype, device=self.device
        )
        thetas_t = thetas_t.reshape(-1, len(self.parametric))
        m, n = thetas_t.shape[0], x.shape[0]
        xx = x.to(self.device).reshape(n, -1).repeat(m, 1)
        tt = t.to(self.device).reshape(n, 1).repeat(m, 1)
        th = thetas_t.repeat_interleave(n, dim=0)
        with torch.no_grad():
            pred = model(torch.cat([xx, th, tt], dim=1))[:, field : field + 1]
        err = (pred - y.to(self.device).reshape(n, 1).repeat(m, 1)) ** 2
        return err.reshape(m, n).sum(dim=1)

    def normalization_bounds(self) -> list[list[float]]:
        """One ``[lo, hi]`` per network input (the spatial axes, then time), from the domain."""
        bounds = [[float(lo), float(hi)] for lo, hi in self.domain]
        t0, t1 = float(self.time_domain[0]), float(self.time_domain[1])
        bounds.append([t0, t1] if t1 > t0 else [t0, t0 + 1.0])
        return bounds

    # --------------------------------------------------------- unknown functions
    def trainable_parameters_iter(self):
        """The scalar unknowns and the weights of every unknown function (for the optimizer)."""
        return itertools.chain(
            super().trainable_parameters_iter(), self.unknown_functions.parameters()
        )

    def evaluate_unknown(
        self,
        name: str,
        x: torch.Tensor,
        t: torch.Tensor | None = None,
        model: torch.nn.Module | None = None,
        truth: bool = False,
    ) -> torch.Tensor:
        """The learned (or, with ``truth=True``, the true) unknown function ``name`` at ``x, t``.

        ``x`` is ``(N, d)`` over the spatial coordinates and ``t`` ``(N, 1)``;
        ``model`` is needed when the function depends on a field.
        """
        fn = cast(UnknownFunction, self.unknown_functions[name])
        x = x.to(self.device).reshape(x.shape[0], -1)
        values: dict[str, torch.Tensor] = {}
        for i, c in enumerate(self.spatial_coordinates):
            values[c] = x[:, i : i + 1]
        if t is not None:
            values["t"] = t.to(self.device).reshape(-1, 1)
        if any(i in self.fields for i in fn.inputs):
            if model is None:
                raise ValueError(f"{name!r} depends on a field; pass the model")
            tt = values.get("t", torch.zeros(x.shape[0], 1, device=self.device))
            u = model(torch.cat([x, tt], dim=1))
            for j, field in enumerate(self.fields):
                values[field] = u[:, j : j + 1]
        inputs = {i: values[i] for i in fn.inputs}
        return fn.truth(inputs) if truth else fn(inputs)

    @torch.no_grad()
    def unknown_function_error(
        self, name: str, num_points: int = 2000, model: torch.nn.Module | None = None
    ) -> float:
        """Relative L2 error of the learned unknown ``name`` against its truth.

        Over a tensor grid of its inputs when they are all coordinates
        (:meth:`UnknownFunction.grid`), else over ``num_points`` uniform
        space-time points with the fields taken from ``model``.
        """
        fn = cast(UnknownFunction, self.unknown_functions[name])
        if all(i in self.coordinates for i in fn.inputs):
            n = max(2, int(round(num_points ** (1.0 / len(fn.inputs)))))
            return fn.relative_error(n=n)
        x, t = self._sample_uniform(num_points)
        pred = self.evaluate_unknown(name, x, t, model=model)
        true = self.evaluate_unknown(name, x, t, model=model, truth=True)
        return float(torch.linalg.norm(pred - true) / torch.linalg.norm(true))

    def unknown_state_dict(self) -> dict[str, torch.Tensor]:
        """Weights of the unknown functions (saved next to the model by the trainer)."""
        return self.unknown_functions.state_dict()

    def load_unknown_state_dict(self, state: Mapping[str, torch.Tensor]) -> None:
        """Restore weights saved by :meth:`unknown_state_dict`."""
        self.unknown_functions.load_state_dict(dict(state))
        self.unknown_functions.to(self.device)

    def __repr__(self) -> str:
        eqs = [e.text for e in self._equations] or ["<callable residual>"]
        inputs = f", inputs={tuple(self.inputs)}" if self.inputs else ""
        unknowns = f", unknown_functions={self._unknown_names}" if self._unknown_names else ""
        return (
            f"CustomPDE({self.config.name!r}, fields={self.fields}, "
            f"coordinates={self.coordinates}, equations={eqs}{unknowns}{inputs})"
        )
