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
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
import yaml

from .pde_base import PDEBase, PDEConfig
from .symbolic import CompiledExpression, compile_expression, derivative_name, split_derivative

_DUMMY_COORDINATE = "_x"  # the spatial input an ODE's network ignores

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
    ):
        """Evaluate ``model`` at ``x, t`` (fresh leaf copies that require grad)."""
        self._x = x.detach().to(device).requires_grad_(True)
        self._t = t.detach().to(device).requires_grad_(True)
        self._fields = tuple(fields)
        self._coordinates = tuple(coordinates)
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
        """The field ``field`` as predicted by the network, shape ``(N, 1)``."""
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
        return sum(self.d(field, c, c) for c in self._spatial)

    def __getattr__(self, name: str) -> torch.Tensor:
        """``f.u``, ``f.u_xx``, ``f.x`` by attribute; anything else is an error."""
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._fields:
            return self.value(name)
        if name in self._coordinates:
            return self.coordinate(name)
        split = split_derivative(name, self._fields, self._coordinates)
        if split is not None:
            return self.d(split[0], *split[1])
        raise AttributeError(
            f"{name!r} is not a field {self._fields}, a coordinate {self._coordinates} "
            "or a derivative such as 'u_xx'"
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
            ``neumann``, ``periodic``), applied to every field.
        exact_solution: ``u(x, t)`` in the same forms as the initial
            condition (expressions may use ``t``), or a callable
            ``u(x, t) -> (N, k)``; enables :meth:`validate`.
        name: Display name for runs and the dashboard.
        trainable_parameters: Parameters to identify in inverse mode.
        parameter_initial_guesses: Their starting values.
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
        self.spatial_coordinates = tuple(k for k in domain if k != "t")
        self.time_dependent = "t" in domain
        self.is_ode = not self.spatial_coordinates
        if self.is_ode:
            self.spatial_coordinates = (_DUMMY_COORDINATE,)
            domain = {_DUMMY_COORDINATE: (0.0, 1.0), **domain}
        self.coordinates = self.spatial_coordinates + (("t",) if self.time_dependent else ())
        self.parameters = {str(k): float(v) for k, v in (parameters or {}).items()}
        self._architecture = architecture
        self._spec = {
            "equations": equations if not callable(equations) else None,
            "domain": {k: list(v) for k, v in domain.items() if k != _DUMMY_COORDINATE},
            "fields": list(self.fields),
            "parameters": dict(self.parameters),
            "initial_condition": None if callable(initial_condition) else initial_condition,
            "initial_velocity": None if callable(initial_velocity) else initial_velocity,
            "boundary_conditions": copy.deepcopy(dict(boundary_conditions or {})),
            "exact_solution": None if callable(exact_solution) else exact_solution,
            "name": name,
            "time_order": time_order,
            "architecture": architecture,
        }

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
            self._equations = [
                compile_expression(t, self.fields, self.coordinates, self.parameters) for t in texts
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
        missing = sorted(self._needed - set(self.parameters))
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
            boundary_conditions=copy.deepcopy(dict(boundary_conditions or {})),
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

    # ----------------------------------------------------------------- building
    def _compile_map(
        self, spec: Any, what: str, with_time: bool
    ) -> dict[str, CompiledExpression] | None:
        if spec is None or callable(spec):
            return None
        coords = self.coordinates if with_time else self.spatial_coordinates
        texts = _as_expression_map(spec, self.fields, what)
        compiled = {}
        for field, text in texts.items():
            expr = compile_expression(text, (), coords, self.parameters)
            if expr.parameters and set(expr.parameters) & set(self.fields):
                raise ValueError(f"{what} for {field!r} must not depend on the fields: {text!r}")
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
            return cls.from_dict(yaml.safe_load(f) or {}, **overrides)

    def to_dict(self) -> dict[str, Any]:
        """The textual spec that rebuilds this problem (callables are left out as ``None``)."""
        return copy.deepcopy(self._spec)

    def build_trainer(
        self,
        architecture: str | None = None,
        config_path: str | None = None,
        device: torch.device | str | None = None,
        **training: Any,
    ):
        """A :class:`~pinnrl.training.trainer.PDETrainer` for this problem with packaged defaults.

        ``architecture`` names an entry of the ``architectures:`` block of
        ``config.yaml`` (``config_path`` for your own file). ``training``
        overrides keys of the ``training:`` block (``num_epochs``,
        ``batch_size``, ``num_collocation_points``, ``learning_rate``,
        ``weight_decay``, ``mode``, ``loss_weights``, ...).
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
        if self.config.trainable_parameters and "mode" not in training:
            training_cfg["mode"] = "inverse"
        architecture = architecture or self._architecture
        device = torch.device(device) if device is not None else self.device
        config = Config.__new__(Config)
        config.device = device
        config.training = TrainingConfig.from_dict(training_cfg)
        config.model = ModelConfig.from_architecture(
            architecture,
            defaults.get("architectures", {}).get(architecture, {}),
            input_dim=self.config.input_dim,
            output_dim=self.config.output_dim,
        )
        config.rl = None
        self.config.training = config.training
        self.device = device
        self.config.device = device
        model = PINNModel(config=config, device=device).to(device)
        return PDETrainer(model=model, pde=self, config=config, device=device)

    # ------------------------------------------------------------------ physics
    def _parameter_values(self) -> dict[str, Any]:
        """Parameter name to value: the live ``nn.Parameter`` for trainable ones."""
        return {name: self.get_parameter(name) for name in self.parameters}

    def compute_residual(
        self, model: torch.nn.Module, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Residuals of every equation at ``x, t``, shape ``(N, k)``."""
        f = Fields(model, x, t, self.fields, self.coordinates, self.device)
        params = self._parameter_values()
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
        for i, c in enumerate(self.spatial_coordinates):
            values[c] = x[:, i : i + 1]
        if t is not None:
            values["t"] = t
        columns = []
        for field in self.fields:
            expr = (compiled or {}).get(field)
            columns.append(expr(values, like) if expr is not None else torch.zeros_like(like))
        return torch.cat(columns, dim=1)

    def _create_initial_condition(self, params: dict[str, Any]) -> Callable | None:
        if not params:
            return None
        return lambda x, t: self._evaluate_map(self._ic, self._ic_fn, x, None)

    def _initial_loss(self, model: torch.nn.Module) -> torch.Tensor:
        """``u(x, t0) = u0(x)``, plus ``u_t(x, t0) = v0(x)`` for second order in time."""
        if self.initial_condition is None:
            return torch.tensor(0.0, device=self.device)
        n = int(self._training_option("num_initial_points", 100))
        x, t = self._seeded_points("initial", n)
        t = t.clone().requires_grad_(self.time_order >= 2)
        u = model(torch.cat([x, t], dim=1))
        loss = self._apply_loss_fn(u - self.initial_condition(x, t.detach()))
        if self.time_order >= 2:
            velocity = self._evaluate_map(self._velocity, self._velocity_fn, x, None)
            u_t = torch.cat([self._grad(u[:, j : j + 1], t) for j in range(u.shape[1])], dim=1)
            loss = loss + self._apply_loss_fn(u_t - velocity)
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

    def __repr__(self) -> str:
        eqs = [e.text for e in self._equations] or ["<callable residual>"]
        return (
            f"CustomPDE({self.config.name!r}, fields={self.fields}, "
            f"coordinates={self.coordinates}, equations={eqs})"
        )
