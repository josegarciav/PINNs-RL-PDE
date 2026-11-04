"""Sensor operators: what a plant measures, as observation operators on the solution.

A probe does not return the field at random points of space-time. It sits at
one place and reports an expression of the fields at a sequence of times
(:class:`PointSensor`); a flux meter at a wall reports ``-alpha*u_x`` on a face
(:class:`BoundarySensor`); a level or a tank temperature reports an integral
over the domain (:class:`IntegralSensor`). Each sensor has its own noise and a
transport or processing ``delay``: the record stamped ``t`` reflects the state
at ``t - delay``.

Sensors attach to a :class:`~pinnrl.pdes.custom.CustomPDE` through
``sensors=[...]``. Their records enter the data loss as one weighted
mean-squared term per sensor, ``pde.observe(seed=...)`` fills the records from
the exact solution (through the same operators, with each sensor's noise), and
:func:`sensor_jacobian` gives the sensitivities ``d H[u] / d theta`` of a
closed-form solution for the Fisher information of a layout, the quantity
sensor-placement studies (K02, J09) optimise.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from .symbolic import compile_expression


def _column(values: Any, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(values, dtype=np.float64)).reshape(-1, 1).to(dtype)


class Sensor:
    """Base class: an expression of the fields observed at a set of space-time points.

    Args:
        expression: What the sensor reports, in the expression language of
            :class:`~pinnrl.pdes.custom.CustomPDE` (``"u"``, ``"-alpha*u_x"``,
            ``"u^2"``); it may use fields, their derivatives, coordinates,
            parameters, unknown functions and inputs.
        times: Timestamps of the records, ``(M,)``.
        noise: Standard deviation of the measurement noise (used by
            ``pde.observe`` and by likelihood weighting).
        delay: The record stamped ``t`` reflects the state at ``t - delay``.
        weight: Multiplier of this sensor's mean-squared error in the data loss.
        name: Label for reports; defaults to the class name.
    """

    kind = "sensor"

    def __init__(
        self,
        expression: str = "u",
        times: Sequence[float] | np.ndarray | None = None,
        *,
        noise: float = 0.0,
        delay: float = 0.0,
        weight: float = 1.0,
        name: str | None = None,
    ):
        self.expression = str(expression)
        self.times = np.asarray(times if times is not None else [], dtype=np.float64).reshape(-1)
        self.noise = float(noise)
        self.delay = float(delay)
        self.weight = float(weight)
        self.name = name or self.kind
        self.values: torch.Tensor | None = None
        self._pde: Any = None

    # ----------------------------------------------------------------- binding
    def bind(self, pde: Any) -> Sensor:
        """Attach to a problem (domain, coordinates, device); called by ``CustomPDE``."""
        self._pde = pde
        return self

    @property
    def n_records(self) -> int:
        """Number of records."""
        return int(self.times.size)

    def evaluation_times(self) -> torch.Tensor:
        """The times the state is read at, ``(M, 1)``: the timestamps minus the delay."""
        return _column(self.times - self.delay)

    def points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Space-time points ``(x, t)`` where the expression is evaluated, one row per record."""
        raise NotImplementedError

    def predict(
        self, model: torch.nn.Module, parameters: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """``H[u]`` for the network ``model``, one value per record, ``(M, 1)``.

        ``parameters`` overrides parameter values inside the expression (for
        sensitivities); the model is responsible for its own dependence.
        """
        x, t = self.points()
        return self._pde.evaluate_expression(self.expression, model, x, t, parameters)

    # ----------------------------------------------------------------- records
    def set_values(self, values: Sequence[float] | np.ndarray | torch.Tensor) -> Sensor:
        """Attach the measured records, one per timestamp."""
        values = _column(
            values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
        )
        if values.shape[0] != self.n_records:
            raise ValueError(
                f"{self.name}: {values.shape[0]} values for {self.n_records} timestamps"
            )
        self.values = values.to(self._pde.device if self._pde is not None else "cpu")
        return self

    def observe(
        self, model: torch.nn.Module, noise: float | None = None, seed: int | None = 0
    ) -> Sensor:
        """Fill the records from ``model`` (the exact solution wrapped, or a trained network)."""
        clean = self.predict(model).detach()  # autograd stays on: the expression may need u_x
        sigma = self.noise if noise is None else float(noise)
        if sigma > 0:
            gen = torch.Generator(device="cpu")
            if seed is not None:
                gen.manual_seed(int(seed))
            clean = clean + sigma * torch.randn(clean.shape, generator=gen).to(clean.device)
        return self.set_values(clean)

    def loss(self, model: torch.nn.Module, reduce) -> torch.Tensor:
        """The sensor's weighted data misfit ``weight * reduce(H[u] - y)``."""
        if self.values is None:
            raise ValueError(
                f"{self.name} has no records; call pde.observe() or sensor.set_values()"
            )
        return self.weight * reduce(self.predict(model) - self.values.to(self._pde.device))

    def to_spec(self) -> dict[str, Any]:
        """The dict that rebuilds the sensor (records included when present)."""
        spec: dict[str, Any] = {
            "kind": self.kind,
            "expression": self.expression,
            "times": self.times.tolist(),
            "noise": self.noise,
            "delay": self.delay,
            "weight": self.weight,
            "name": self.name,
        }
        if self.values is not None:
            spec["values"] = self.values.detach().cpu().reshape(-1).tolist()
        return spec

    @staticmethod
    def from_spec(spec: dict[str, Any]) -> Sensor:
        """Rebuild a sensor from :meth:`to_spec`."""
        spec = dict(spec)
        kind = spec.pop("kind")
        values = spec.pop("values", None)
        cls = {c.kind: c for c in (PointSensor, BoundarySensor, IntegralSensor)}[kind]
        sensor = cls(**spec)
        if values is not None:
            sensor._pending_values = values
        return sensor

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.name!r}, {self.expression!r}, {self.n_records} records)"
        )


class PointSensor(Sensor):
    """A probe at a fixed location, read at the given times.

    Args:
        location: Spatial position, a number in 1-D or a sequence of ``d`` numbers.
        times: Timestamps of the records.
        expression: What is reported there (default the first field).
        **kwargs: ``noise``, ``delay``, ``weight``, ``name`` as for :class:`Sensor`.
    """

    kind = "point"

    def __init__(
        self,
        location: float | Sequence[float],
        times: Sequence[float] | np.ndarray,
        expression: str = "u",
        **kwargs: Any,
    ):
        super().__init__(expression, times, **kwargs)
        self.location = np.asarray(location, dtype=np.float64).reshape(-1)

    def points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The probe location repeated for every record, and the evaluation times."""
        t = self.evaluation_times()
        x = torch.as_tensor(self.location, dtype=t.dtype).reshape(1, -1).expand(t.shape[0], -1)
        return x.contiguous(), t

    def to_spec(self) -> dict[str, Any]:
        """The base spec plus the location."""
        return {**super().to_spec(), "location": self.location.tolist()}


class BoundarySensor(Sensor):
    """A sensor on one face of the domain, e.g. a wall flux ``-alpha*u_x``.

    Args:
        side: ``"left"`` (lower face) or ``"right"`` (upper face) of ``axis``.
        times: Timestamps of the records.
        expression: What is reported on the face.
        axis: Spatial axis of the face (0 in 1-D).
        position: Coordinates along the other axes in more than one dimension
            (defaults to the centre of each).
        **kwargs: ``noise``, ``delay``, ``weight``, ``name`` as for :class:`Sensor`.
    """

    kind = "boundary"

    def __init__(
        self,
        side: str,
        times: Sequence[float] | np.ndarray,
        expression: str = "u",
        axis: int = 0,
        position: Sequence[float] | None = None,
        **kwargs: Any,
    ):
        super().__init__(expression, times, **kwargs)
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', not {side!r}")
        self.side = side
        self.axis = int(axis)
        self.position = None if position is None else [float(p) for p in position]

    def points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Points on the face for every record, and the evaluation times."""
        domain = self._pde.domain
        t = self.evaluation_times()
        centre = [0.5 * (lo + hi) for lo, hi in domain]
        x = list(self.position) if self.position is not None else centre
        x = list(x)
        x[self.axis] = domain[self.axis][0 if self.side == "left" else 1]
        xx = torch.tensor(x, dtype=t.dtype).reshape(1, -1).expand(t.shape[0], -1)
        return xx.contiguous(), t

    def to_spec(self) -> dict[str, Any]:
        """The base spec plus the face."""
        return {
            **super().to_spec(),
            "side": self.side,
            "axis": self.axis,
            "position": self.position,
        }


class IntegralSensor(Sensor):
    """A domain integral ``∫ w(x) H[u](x, t) dx`` read at the given times (a level, a mean).

    Args:
        times: Timestamps of the records.
        expression: The integrand's field expression (default the first field).
        weight_expression: An optional weight ``w(x)`` in the spatial coordinates.
        quadrature: Midpoint-rule points per spatial axis.
        normalize: Divide by the domain volume (a mean instead of an integral).
        **kwargs: ``noise``, ``delay``, ``weight``, ``name`` as for :class:`Sensor`.
    """

    kind = "integral"

    def __init__(
        self,
        times: Sequence[float] | np.ndarray,
        expression: str = "u",
        weight_expression: str | None = None,
        quadrature: int = 64,
        normalize: bool = False,
        **kwargs: Any,
    ):
        super().__init__(expression, times, **kwargs)
        self.weight_expression = weight_expression
        self.quadrature = int(quadrature)
        self.normalize = bool(normalize)
        self._weight_compiled = None

    def _grid(self) -> tuple[torch.Tensor, float]:
        """Midpoint nodes over the spatial domain, ``(Q, d)``, and the cell volume."""
        axes, volume = [], 1.0
        for lo, hi in self._pde.domain:
            h = (hi - lo) / self.quadrature
            axes.append(
                torch.linspace(lo + h / 2, hi - h / 2, self.quadrature, dtype=torch.float64)
            )
            volume *= h
        mesh = torch.meshgrid(*axes, indexing="ij")
        return torch.stack([m.reshape(-1) for m in mesh], dim=1).float(), volume

    def points(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Quadrature nodes repeated for every record, ``(M * Q, d)``, with their times."""
        nodes, _ = self._grid()
        t = self.evaluation_times()
        x = nodes.repeat(t.shape[0], 1)
        tt = t.repeat_interleave(nodes.shape[0], dim=0)
        return x, tt

    def predict(
        self, model: torch.nn.Module, parameters: dict[str, Any] | None = None
    ) -> torch.Tensor:
        """The weighted integral of the expression at every record time, ``(M, 1)``."""
        nodes, volume = self._grid()
        x, t = self.points()
        values = self._pde.evaluate_expression(self.expression, model, x, t, parameters)
        if self.weight_expression is not None:
            if self._weight_compiled is None:
                self._weight_compiled = compile_expression(
                    self.weight_expression, (), self._pde.spatial_coordinates, self._pde.parameters
                )
            inputs: dict[str, Any] = dict(self._pde.parameters)
            for i, c in enumerate(self._pde.spatial_coordinates):
                inputs[c] = x[:, i : i + 1].to(values.device)
            values = values * self._weight_compiled(inputs, values)
        total = values.reshape(-1, nodes.shape[0]).sum(dim=1, keepdim=True) * volume
        if self.normalize:
            total = total / (volume * nodes.shape[0])
        return total

    def to_spec(self) -> dict[str, Any]:
        """The base spec plus the integrand weight and quadrature."""
        return {
            **super().to_spec(),
            "weight_expression": self.weight_expression,
            "quadrature": self.quadrature,
            "normalize": self.normalize,
        }


class _ExactWithThetas(torch.nn.Module):
    """The closed-form solution as a network, with per-point parameter copies that require grad."""

    def __init__(self, pde: Any, compiled: dict, d: int, thetas: dict[str, torch.Tensor]):
        super().__init__()
        self.pde, self.compiled, self.d, self.thetas = pde, compiled, d, thetas

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """``u*(x, t; theta)`` in double precision, ``(N, k)``."""
        xx, tt = inp[:, : self.d].double(), inp[:, self.d :].double()
        like = torch.ones(xx.shape[0], 1, dtype=torch.float64)
        values: dict[str, Any] = {k: float(v) for k, v in self.pde.parameters.items()}
        values.update(self.thetas)
        for i, c in enumerate(self.pde.spatial_coordinates):
            values[c] = xx[:, i : i + 1]
        if self.pde.time_dependent:
            values["t"] = tt
        cols = []
        for field in self.pde.fields:
            expr = self.compiled.get(field)
            cols.append(expr(values, like) if expr is not None else torch.zeros_like(like))
        return torch.cat(cols, dim=1)


def sensor_jacobian(
    pde: Any, parameters: Sequence[str], sensors: Sequence[Sensor] | None = None
) -> torch.Tensor:
    """``d H[u*] / d theta`` of the closed-form solution for every record, ``(M_total, p)``.

    The exact solution is evaluated with per-record copies of the parameters
    that require grad, the sensor expressions (with their derivatives) are
    applied to it, and autograd gives the sensitivity of each record. Feed the
    result to :func:`pinnrl.analysis.fisher_information` with each sensor's
    noise to get the Fisher information and Cramér–Rao bound of a layout.

    Raises:
        ValueError: If the PDE has no expression-valued exact solution or a
            name is not one of its parameters.
    """
    compiled = getattr(pde, "_exact", None)
    if not compiled:
        raise ValueError(
            f"{pde.config.name}: sensor_jacobian needs an expression-valued exact_solution"
        )
    unknown = [p for p in parameters if p not in pde.parameters]
    if unknown:
        raise ValueError(
            f"{unknown} are not parameters of {pde.config.name}: {list(pde.parameters)}"
        )
    sensors = list(pde.sensors if sensors is None else sensors)
    rows = []
    for sensor in sensors:
        sensor.bind(pde)
        x, t = sensor.points()
        n = x.shape[0]
        thetas = {
            name: torch.full(
                (n, 1), float(pde.parameters[name]), dtype=torch.float64, requires_grad=True
            )
            for name in parameters
        }

        pred = sensor.predict(
            _ExactWithThetas(pde, compiled, x.shape[1], thetas), parameters=thetas
        )
        records = sensor.n_records
        cols = []
        for name in parameters:
            g = torch.autograd.grad(pred.sum(), thetas[name], retain_graph=True, allow_unused=True)[
                0
            ]
            g = torch.zeros(n, 1, dtype=torch.float64) if g is None else g
            # an integral sensor has one theta copy per quadrature node: sum them per record
            cols.append(g.reshape(records, -1).sum(dim=1))
        rows.append(torch.stack(cols, dim=1))
    return torch.cat(rows, dim=0)


def sensor_noise_vector(sensors: Sequence[Sensor]) -> torch.Tensor:
    """One noise standard deviation per record, in the order :func:`sensor_jacobian` uses."""
    return torch.cat(
        [torch.full((s.n_records,), float(s.noise), dtype=torch.float64) for s in sensors]
    )
