"""Field-valued unknowns: a learnable function standing in for a coefficient or closure.

A scalar coefficient that is identified in inverse mode is one ``nn.Parameter``.
When the unknown physics is a *function*, a diffusivity that drifts in time
``alpha(t)``, a heterogeneous coefficient field ``alpha(x)`` or a closure
``f(u)``, the same place in the equation is taken by an :class:`UnknownFunction`,
a small network of the coordinates and fields it depends on. Its weights are
trained with the data, like the scalar case, and its name is used in the
residual expressions exactly like a parameter: ``"u_t - alpha*u_xx"``, with
``alpha_x`` for its derivative when the equation needs it.

Constraints that the physics imposes are built into the output so training
never leaves the admissible set: a positive coefficient (``positive=True``),
a bounded one (``bounds=(lo, hi)``) and monotonicity in chosen inputs
(``monotone={"u": "increasing"}``, a network with positive weights), and a
value the function is known to take at one point (``anchor={"at": {"X": 0.05},
"value": 0.0}``: a closure that vanishes at equilibrium, a fouling that starts
at zero), which with monotonicity also pins the function's sign.

A ``truth`` expression records the function the synthetic data came from, so
:meth:`UnknownFunction.relative_error` grades the recovery and the exact
solution of the problem can refer to it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from .symbolic import compile_expression

_ACTIVATIONS: dict[str, Callable[[], nn.Module]] = {
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}
#: Activations that are non-decreasing, the ones a monotone network may use.
_MONOTONE_ACTIVATIONS = {"tanh", "sigmoid", "relu", "softplus", "silu"}


def _inverse_softplus(y: float) -> float:
    """``z`` with ``softplus(z) = y`` for ``y > 0``."""
    if y > 20.0:
        return y
    return math.log(math.expm1(y))


class _ConstrainedLinear(nn.Module):
    """A linear layer whose chosen input columns, or all of its weights, are kept non-negative.

    ``signs`` holds one entry per input column: ``+1`` forces that column's
    weights to be non-negative, ``-1`` non-positive, ``0`` leaves it free. With
    ``all_positive`` every weight is non-negative (the hidden layers of a
    monotone network). Softplus is the map from the raw parameter to the weight.
    """

    signs: torch.Tensor

    def __init__(
        self, n_in: int, n_out: int, signs: Sequence[int] = (), all_positive: bool = False
    ):
        super().__init__()
        self.linear = nn.Linear(n_in, n_out)
        self.all_positive = all_positive
        signs = list(signs) + [0] * (n_in - len(signs))
        self.register_buffer("signs", torch.tensor(signs, dtype=torch.float32))
        self.constrained = all_positive or any(s != 0 for s in signs)

    def weight(self) -> torch.Tensor:
        """The effective weight matrix after the constraints."""
        w = self.linear.weight
        if self.all_positive:
            return nn.functional.softplus(w)
        if not self.constrained:
            return w
        positive = nn.functional.softplus(w)
        fixed = self.signs.abs() > 0
        return torch.where(fixed.unsqueeze(0), positive * self.signs.unsqueeze(0), w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ W.T + b`` with the constrained weights."""
        return nn.functional.linear(x, self.weight(), self.linear.bias)


class _BasisFeatures(nn.Module):
    """Fixed features of one normalized input in ``[-1, 1]``: cosine or Legendre modes."""

    def __init__(self, kind: str, modes: int, skip_constant: bool = False):
        super().__init__()
        if kind not in ("cosine", "legendre"):
            raise ValueError(f"basis type must be 'cosine' or 'legendre', not {kind!r}")
        if modes < 1:
            raise ValueError("a basis needs at least one mode")
        self.kind, self.modes = kind, int(modes)
        self.skip_constant = bool(skip_constant)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``(N, 1)`` in ``[-1, 1]`` to ``(N, modes)``; the first mode is the constant 1.

        With ``skip_constant`` the constant is left out and the modes are
        ``1 .. modes`` (an anchored function supplies its own constant).
        """
        z = z[:, :1]
        first = 1 if self.skip_constant else 0
        total = self.modes + first
        if self.kind == "cosine":
            k = torch.arange(first, total, device=z.device, dtype=z.dtype)
            return torch.cos(k * math.pi * (z + 1.0) / 2.0)
        cols = [torch.ones_like(z), z]
        for n in range(1, total - 1):  # Bonnet recursion
            cols.append(((2 * n + 1) * z * cols[-1] - n * cols[-2]) / (n + 1))
        return torch.cat(cols[first:total], dim=1)


class UnknownFunction(nn.Module):
    """A learnable function of coordinates and fields that replaces a coefficient.

    Args:
        name: The symbol it takes in the equations (``"alpha"``).
        inputs: Names it depends on, coordinates (``"x"``, ``"t"``) or fields
            (``"u"``), in the order they enter the network.
        input_ranges: ``{input: (lo, hi)}`` used to map every input to
            ``[-1, 1]`` before the network, and as the range over which the
            recovery is graded. :class:`~pinnrl.pdes.custom.CustomPDE` fills
            in the domain for coordinates; fields default to ``(-1, 1)``.
        hidden: Hidden widths, default ``(32, 32)``.
        activation: ``tanh`` (default), ``sigmoid``, ``relu``, ``softplus``,
            ``gelu`` or ``silu``; a monotone network needs a non-decreasing one.
        positive: Output through a softplus, so the function is positive.
        bounds: ``(lo, hi)``: output through a sigmoid into that interval.
        monotone: ``{input: "increasing" | "decreasing"}`` (or one of those
            strings when there is a single input): the function is monotone in
            those inputs, by non-negative weights on every path from them.
        initial: The constant the function starts at (the network's last layer
            starts at zero), the analogue of a scalar's initial guess. An
            anchored function starts at its anchor value everywhere.
        anchor: ``{"at": {input: coordinate}, "value": v}`` (``"at"`` may be a
            number when there is a single input): the function takes ``v`` at
            that point exactly, whatever the weights, by subtracting its own
            value there. Combined with ``monotone`` this fixes the sign
            everywhere: a closure with ``anchor={"at": X_eq, "value": 0}`` and
            ``monotone="increasing"`` is non-negative above ``X_eq``. ``v`` must
            be inside ``bounds`` or positive with ``positive``; with a basis the
            constant mode is dropped, since the anchor supplies it.
        basis: ``{"type": "cosine" | "legendre", "modes": n}``: instead of a
            network, a linear combination of ``n`` fixed modes of a single
            input, so the unknown is the coefficient vector
            :attr:`coefficients` and the identifiability of each mode can be
            analysed like a parameter (K48). ``hidden``, ``activation`` and
            ``monotone`` do not apply; ``positive`` and ``bounds`` still do.
        truth: The true function as an expression in the inputs (``"0.1 + 0.05*t"``,
            with ``pi``, ``exp``, ``heaviside``, ...) or a callable
            ``truth(**inputs) -> (N, 1)``. Used for synthetic data, for exact
            solutions that mention the unknown, and by :meth:`relative_error`.

    Raises:
        ValueError: For an empty input list, an unknown activation, a
            non-monotone activation with ``monotone``, inconsistent ``bounds``,
            an ``initial`` or anchor value outside the admissible set, or an
            anchor that does not give every input.
    """

    _lo: torch.Tensor
    _hi: torch.Tensor
    _z_anchor: torch.Tensor

    def __init__(
        self,
        name: str,
        inputs: Sequence[str],
        *,
        input_ranges: Mapping[str, Sequence[float]] | None = None,
        hidden: Sequence[int] = (32, 32),
        activation: str = "tanh",
        positive: bool = False,
        bounds: Sequence[float] | None = None,
        monotone: Mapping[str, str] | str | None = None,
        initial: float = 1.0,
        truth: str | Callable | None = None,
        basis: Mapping[str, Any] | None = None,
        anchor: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        self.name = str(name)
        self.inputs = tuple(str(i) for i in inputs)
        if not self.inputs:
            raise ValueError(f"unknown function {name!r} needs at least one input")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"unknown activation {activation!r} for {name!r}; one of {sorted(_ACTIVATIONS)}"
            )
        if isinstance(monotone, str):
            if len(self.inputs) != 1:
                raise ValueError(
                    f"monotone={monotone!r} for {name!r} needs a single input; "
                    "give {input: direction} instead"
                )
            monotone = {self.inputs[0]: monotone}
        self.monotone: dict[str, str] = {}
        for key, direction in dict(monotone or {}).items():
            if key not in self.inputs:
                raise ValueError(
                    f"monotone input {key!r} is not an input of {name!r} {self.inputs}"
                )
            if direction not in ("increasing", "decreasing"):
                raise ValueError(f"monotone direction for {key!r} must be increasing or decreasing")
            self.monotone[key] = direction
        if self.monotone and activation not in _MONOTONE_ACTIVATIONS:
            raise ValueError(
                f"a monotone network needs a non-decreasing activation, not {activation!r}"
            )
        if bounds is not None:
            bounds = (float(bounds[0]), float(bounds[1]))
            if not bounds[0] < bounds[1]:
                raise ValueError(f"bounds for {name!r} must satisfy lo < hi, got {bounds}")
        self.positive = bool(positive) and bounds is None
        self.bounds = bounds
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = activation
        self.initial = float(initial)
        ranges = {k: (float(v[0]), float(v[1])) for k, v in (input_ranges or {}).items()}
        self.input_ranges: dict[str, tuple[float, float]] = {
            i: ranges.get(i, (-1.0, 1.0)) for i in self.inputs
        }
        lo = torch.tensor([self.input_ranges[i][0] for i in self.inputs])
        hi = torch.tensor([self.input_ranges[i][1] for i in self.inputs])
        self.register_buffer("_lo", lo)
        self.register_buffer("_hi", hi)

        self.anchor: dict[str, Any] | None = None
        if anchor is not None:
            at = anchor.get("at") if isinstance(anchor, Mapping) else None
            if at is None or "value" not in anchor:
                raise ValueError(f"anchor of {name!r} needs 'at' and 'value'")
            if not isinstance(at, Mapping):
                if len(self.inputs) != 1:
                    raise ValueError(f"anchor 'at' of {name!r} must give every input {self.inputs}")
                at = {self.inputs[0]: at}
            missing = [i for i in self.inputs if i not in at]
            if missing:
                raise ValueError(f"anchor of {name!r} gives no coordinate for {missing}")
            self.anchor = {
                "at": {i: float(at[i]) for i in self.inputs},
                "value": float(anchor["value"]),
            }
            self.initial = self.anchor["value"]
            z_a = torch.tensor([[self.anchor["at"][i] for i in self.inputs]])
            self.register_buffer("_z_anchor", 2.0 * (z_a - lo) / (hi - lo) - 1.0)

        self.basis: dict[str, Any] | None = None
        if basis is not None:
            if len(self.inputs) != 1:
                raise ValueError(f"a basis needs a single input; {name!r} has {self.inputs}")
            if self.monotone:
                raise ValueError("monotone is not available with a basis")
            self.basis = {
                "type": str(basis.get("type", "cosine")),
                "modes": int(basis.get("modes", 6)),
            }
            self.net = nn.Sequential(
                _BasisFeatures(
                    self.basis["type"], self.basis["modes"], skip_constant=self.anchor is not None
                ),
                _ConstrainedLinear(self.basis["modes"], 1),
            )
        else:
            signs = [
                {"increasing": 1, "decreasing": -1}.get(self.monotone.get(i, ""), 0)
                for i in self.inputs
            ]
            layers: list[nn.Module] = []
            widths = [len(self.inputs), *self.hidden, 1]
            for k, (n_in, n_out) in enumerate(zip(widths[:-1], widths[1:], strict=True)):
                if k == 0:
                    layers.append(_ConstrainedLinear(n_in, n_out, signs=signs))
                else:
                    layers.append(_ConstrainedLinear(n_in, n_out, all_positive=bool(self.monotone)))
                if k < len(widths) - 2:
                    layers.append(_ACTIVATIONS[activation]())
            self.net = nn.Sequential(*layers)
        self._start_at_constant(self.initial)

        self._truth_fn: Callable | None = truth if callable(truth) else None
        self._truth_expr = None
        self.truth_text: str | None = None
        if truth is not None and not callable(truth):
            self.truth_text = str(truth)
            self._truth_expr = compile_expression(self.truth_text, (), self.inputs)
            if self._truth_expr.parameters:
                raise ValueError(
                    f"truth {self.truth_text!r} of {name!r} uses {self._truth_expr.parameters}, "
                    f"which are not among its inputs {self.inputs}"
                )

    # ------------------------------------------------------------- construction
    def _start_at_constant(self, value: float) -> None:
        """Zero the last layer's weights and set its bias so the output is ``value`` everywhere."""
        last = cast(_ConstrainedLinear, self.net[-1])
        with torch.no_grad():
            if last.constrained:
                last.linear.weight.fill_(-12.0)  # softplus(-12) ≈ 6e-6
            else:
                last.linear.weight.zero_()
            if self.anchor is not None:  # a flat raw output; the anchor supplies the value
                last.linear.bias.zero_()
                self._preimage(value)  # validates the anchor value
                return
            last.linear.bias.fill_(self._preimage(value))
            if self.basis is not None:  # the constant mode carries the start; no bias
                last.linear.bias.zero_()
                last.linear.weight[0, 0] = self._preimage(value)

    @property
    def coefficients(self) -> torch.Tensor:
        """The mode coefficients of a basis function (before the output transform), ``(modes,)``.

        Raises:
            ValueError: When the function is a network rather than a basis.
        """
        if self.basis is None:
            raise ValueError(f"{self.name!r} is a network; coefficients need basis=...")
        return cast(_ConstrainedLinear, self.net[-1]).linear.weight.reshape(-1)

    def modes(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The basis modes at ``values``, ``(N, modes)``, for sensitivity analyses."""
        if self.basis is None:
            raise ValueError(f"{self.name!r} is a network; modes need basis=...")
        z = self._stack(values)
        z = 2.0 * (z - self._lo) / (self._hi - self._lo) - 1.0
        return self.net[0](z)

    def _preimage(self, value: float) -> float:
        if self.bounds is not None:
            lo, hi = self.bounds
            if not lo < value < hi:
                raise ValueError(f"value {value} of {self.name!r} is outside bounds {self.bounds}")
            p = (value - lo) / (hi - lo)
            return math.log(p / (1.0 - p))
        if self.positive:
            if value <= 0:
                raise ValueError(f"value {value} of {self.name!r} must be positive")
            return _inverse_softplus(value)
        return value

    @classmethod
    def from_spec(
        cls,
        name: str,
        spec: Mapping[str, Any],
        coordinate_ranges: Mapping[str, Sequence[float]] | None = None,
    ) -> UnknownFunction:
        """Build from a dict (the YAML form); ``coordinate_ranges`` supplies the domain."""
        spec = dict(spec)
        inputs = spec.pop("inputs", None)
        if inputs is None:
            raise ValueError(f"unknown function {name!r} needs an 'inputs' list")
        if isinstance(inputs, str):
            inputs = [inputs]
        ranges = {k: tuple(v) for k, v in dict(coordinate_ranges or {}).items()}
        ranges.update({k: tuple(v) for k, v in dict(spec.pop("input_ranges", {}) or {}).items()})
        return cls(name, inputs, input_ranges=ranges, **spec)

    def to_spec(self) -> dict[str, Any]:
        """The dict that rebuilds this function (weights excluded; a callable truth is left out)."""
        spec: dict[str, Any] = {
            "inputs": list(self.inputs),
            "input_ranges": {k: list(v) for k, v in self.input_ranges.items()},
            "hidden": list(self.hidden),
            "activation": self.activation,
            "initial": self.initial,
        }
        if self.positive:
            spec["positive"] = True
        if self.bounds is not None:
            spec["bounds"] = list(self.bounds)
        if self.monotone:
            spec["monotone"] = dict(self.monotone)
        if self.truth_text is not None:
            spec["truth"] = self.truth_text
        if self.basis is not None:
            spec["basis"] = dict(self.basis)
        if self.anchor is not None:
            spec["anchor"] = {"at": dict(self.anchor["at"]), "value": self.anchor["value"]}
        return spec

    # --------------------------------------------------------------- evaluation
    def _stack(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        missing = [i for i in self.inputs if i not in values]
        if missing:
            raise KeyError(f"{self.name!r} needs values for {missing}")
        return torch.cat([values[i].reshape(-1, 1) for i in self.inputs], dim=1)

    def _raw(self, z: torch.Tensor) -> torch.Tensor:
        """The network on normalized inputs, shifted so an anchored function hits its anchor."""
        out = self.net(z)
        if self.anchor is None:
            return out
        return out - self.net(self._z_anchor) + self._preimage(self.anchor["value"])

    def forward(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The learned function at ``values`` (``{input: (N, 1)}``), shape ``(N, 1)``."""
        z = self._stack(values)
        z = 2.0 * (z - self._lo) / (self._hi - self._lo) - 1.0
        out = self._raw(z)
        if self.bounds is not None:
            lo, hi = self.bounds
            return lo + (hi - lo) * torch.sigmoid(out)
        if self.positive:
            return nn.functional.softplus(out)
        return out

    @property
    def has_truth(self) -> bool:
        """Whether a true function was given."""
        return self._truth_fn is not None or self._truth_expr is not None

    def truth(self, values: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The true function at ``values``, shape ``(N, 1)``.

        Raises:
            ValueError: When no truth was given.
        """
        if self._truth_fn is not None:
            out = self._truth_fn(**{i: values[i].reshape(-1, 1) for i in self.inputs})
            return out.reshape(-1, 1)
        if self._truth_expr is None:
            raise ValueError(f"unknown function {self.name!r} has no truth to compare with")
        like = values[self.inputs[0]].reshape(-1, 1)
        return self._truth_expr({i: values[i].reshape(-1, 1) for i in self.inputs}, like)

    def fit(
        self,
        values: Mapping[str, Any],
        targets: Any,
        steps: int = 3000,
        lr: float = 1e-2,
    ) -> float:
        """Pre-train the function to ``targets`` at the input ``values`` (least squares, Adam).

        The start of an inverse run when a pointwise estimate of the function
        exists, e.g. a closure read off a dense record (J07b). Returns the
        final mean-squared error.
        """
        inputs = {
            i: torch.as_tensor(np.asarray(values[i]), dtype=torch.float32).reshape(-1, 1)
            for i in self.inputs
        }
        y = torch.as_tensor(np.asarray(targets), dtype=torch.float32).reshape(-1, 1)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss = torch.zeros(())
        for _ in range(int(steps)):
            optimizer.zero_grad()
            loss = ((self(inputs) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
        return float(loss.detach())

    def grid(
        self, n: int = 256, device: torch.device | str | None = None
    ) -> dict[str, torch.Tensor]:
        """``n`` points per input on a tensor grid over the input ranges (``n**len(inputs)`` in all)."""
        axes = [
            torch.linspace(*self.input_ranges[i], n, device=device, dtype=self._lo.dtype)
            for i in self.inputs
        ]
        mesh = torch.meshgrid(*axes, indexing="ij")
        return {i: m.reshape(-1, 1) for i, m in zip(self.inputs, mesh, strict=True)}

    @torch.no_grad()
    def relative_error(
        self, values: Mapping[str, torch.Tensor] | None = None, n: int = 256
    ) -> float:
        """``||f - f*|| / ||f*||`` against the truth, on ``values`` or on :meth:`grid`."""
        if values is None:
            values = self.grid(n, device=self._lo.device)
        truth = self.truth(values)
        return float(torch.linalg.norm(self(values) - truth) / torch.linalg.norm(truth))

    def extra_repr(self) -> str:
        """Signature, widths and constraints, for ``repr``."""
        parts = [f"{self.name}({', '.join(self.inputs)})"]
        parts.append(
            f"basis={self.basis}" if self.basis is not None else f"hidden={list(self.hidden)}"
        )
        if self.positive:
            parts.append("positive")
        if self.bounds is not None:
            parts.append(f"bounds={list(self.bounds)}")
        if self.monotone:
            parts.append(f"monotone={self.monotone}")
        if self.anchor is not None:
            parts.append(f"anchor={self.anchor['value']} at {self.anchor['at']}")
        return ", ".join(parts)
