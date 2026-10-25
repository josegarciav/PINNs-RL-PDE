"""Expressions for user-defined equations: ``"u_t + u*u_x - nu*u_xx"`` to a torch callable.

The notation is the usual subscript shorthand. A field name followed by an
underscore and coordinate names is a partial derivative: ``u_t``, ``u_xx``,
``u_xy``, ``theta_tt``. Everything else is an identifier of one of four kinds,
decided from the names the caller declares:

* a **field** (``u``, ``v``, ``theta``), the network outputs;
* a **coordinate** (``x``, ``y``, ``t``), the network inputs;
* a **function** or the constant ``pi`` (``sin``, ``exp``, ``tanh``, ``sqrt``,
  ``abs``, ``max``, ...; Euler's number is ``exp(1)``);
* otherwise a **parameter** (``nu``, ``alpha``), which must be supplied as a
  number or identified in inverse mode.

Expressions are parsed by SymPy (so ``^`` means a power, and
``2 x`` needs the ``*``) and compiled once into a function of torch tensors.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import sympy
import torch
from sympy.parsing.sympy_parser import convert_xor, parse_expr, standard_transformations

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _reduce_pairwise(op, args):
    """``op`` folded over ``args``; Python numbers become tensors like the first tensor."""
    ref = next((a for a in args if isinstance(a, torch.Tensor)), None)
    out = None
    for a in args:
        if ref is not None and not isinstance(a, torch.Tensor):
            a = torch.full_like(ref, float(a))
        out = a if out is None else op(out, a)
    return out


def _tensor_args(fn: Callable) -> Callable:
    """``fn`` with Python numbers turned into tensors (SymPy prints ``sqrt(2)`` as a call)."""

    def wrapped(*args):
        return fn(*(a if isinstance(a, torch.Tensor) else torch.tensor(float(a)) for a in args))

    return wrapped


#: Functions available inside expressions, evaluated on tensors.
TORCH_FUNCTIONS: dict[str, Callable] = {
    name: _tensor_args(fn)
    for name, fn in {
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": torch.tan,
        "asin": torch.asin,
        "acos": torch.acos,
        "atan": torch.atan,
        "atan2": torch.atan2,
        "sinh": torch.sinh,
        "cosh": torch.cosh,
        "tanh": torch.tanh,
        "exp": torch.exp,
        "log": torch.log,
        "sqrt": torch.sqrt,
        "Abs": torch.abs,
        "sign": torch.sign,
        "erf": torch.erf,
        "sigmoid": torch.sigmoid,
        "relu": torch.relu,
    }.items()
}
TORCH_FUNCTIONS.update(
    {
        "Min": lambda *a: _reduce_pairwise(torch.minimum, a),
        "Max": lambda *a: _reduce_pairwise(torch.maximum, a),
        # Written with ``where`` rather than ``torch.heaviside`` so autograd accepts it
        # (the derivative is 0 almost everywhere; the delta at 0 is dropped).
        "Heaviside": lambda x, h0=0.5: torch.where(
            x > 0,
            torch.ones_like(x),
            torch.where(x < 0, torch.zeros_like(x), torch.full_like(x, float(h0))),
        ),
    }
)
#: Spellings accepted in expressions for the functions above.
_FUNCTION_ALIASES = {"abs": "Abs", "min": "Min", "max": "Max", "heaviside": "Heaviside"}
_SYMPY_FUNCTIONS = {
    "sigmoid": sympy.Function("sigmoid"),
    "relu": sympy.Function("relu"),
}
_CONSTANTS = {"pi": sympy.pi}  # ``e`` and ``E`` stay free names (``exp(1)`` is Euler's number)
_TRANSFORMATIONS = standard_transformations + (convert_xor,)

Coordinates = tuple[str, ...]


def split_derivative(token: str, fields: Sequence[str], coordinates: Sequence[str]):
    """``"u_xxt"`` to ``("u", ("x", "x", "t"))``, or ``None`` when ``token`` is not one.

    The subscript is read left to right, longest coordinate name first, so
    coordinates with several letters (``r``, ``theta``) work as long as no
    name is a prefix of another in an ambiguous way.
    """
    for field in sorted(fields, key=len, reverse=True):
        if not token.startswith(field + "_") or len(token) <= len(field) + 1:
            continue
        rest = token[len(field) + 1 :]
        orders: list[str] = []
        while rest:
            for coord in sorted(coordinates, key=len, reverse=True):
                if rest.startswith(coord):
                    orders.append(coord)
                    rest = rest[len(coord) :]
                    break
            else:
                break
        if not rest and orders:
            return field, tuple(orders)
    return None


@dataclass(frozen=True)
class CompiledExpression:
    """One parsed expression and the names it needs.

    Attributes:
        text: The source string.
        expr: The SymPy expression.
        fields: Fields referenced by value.
        derivatives: ``(field, coordinates)`` pairs referenced, e.g. ``("u", ("x", "x"))``.
        coordinates: Coordinates referenced directly.
        parameters: Free names that are neither fields, derivatives nor coordinates.
    """

    text: str
    expr: sympy.Expr
    fields: tuple[str, ...]
    derivatives: tuple[tuple[str, Coordinates], ...]
    coordinates: tuple[str, ...]
    parameters: tuple[str, ...]
    _fn: Callable
    _args: tuple[str, ...]

    @property
    def time_order(self) -> int:
        """Highest order of ``∂/∂t`` in the expression (0 when ``t`` is not a coordinate)."""
        return max((c.count("t") for _, c in self.derivatives), default=0)

    def __call__(self, values: dict[str, torch.Tensor | float], like: torch.Tensor) -> torch.Tensor:
        """Evaluate with ``values`` (name to tensor or number); constants broadcast to ``like``.

        Derivative names use the subscript spelling the expression used
        (``"u_xx"``), as :func:`derivative_name` produces.
        """
        out = self._fn(*(values[a] for a in self._args))
        if not isinstance(out, torch.Tensor):
            return torch.full_like(like, float(out))
        if out.shape != like.shape:
            out = out.expand_as(like) if out.dim() <= like.dim() else out
        return out


def derivative_name(field: str, coordinates: Coordinates) -> str:
    """Canonical spelling of a derivative: the field, ``_``, the coordinates sorted."""
    return f"{field}_{''.join(sorted(coordinates))}"


def compile_expression(
    text: str,
    fields: Sequence[str],
    coordinates: Sequence[str],
    known_parameters: Iterable[str] = (),
) -> CompiledExpression:
    """Parse ``text`` and compile it into a torch callable.

    Args:
        text: The expression, e.g. ``"u_t + u*u_x - nu*u_xx"``.
        fields: Field (output) names.
        coordinates: Coordinate (input) names, including ``"t"`` when the
            problem is time dependent.
        known_parameters: Parameter names the caller can supply; any other
            free name is still accepted and reported in ``parameters`` so the
            caller can raise a precise error.

    Raises:
        ValueError: If the text does not parse or uses an unknown function.
    """
    fields = tuple(fields)
    coordinates = tuple(coordinates)
    local: dict[str, object] = {}
    seen_fields, seen_derivs, seen_coords, seen_params = [], [], [], []
    for ident in dict.fromkeys(_IDENTIFIER.findall(text)):
        canonical = _FUNCTION_ALIASES.get(ident, ident)
        if ident in fields:
            local[ident] = sympy.Symbol(ident)
            seen_fields.append(ident)
            continue
        split = split_derivative(ident, fields, coordinates)
        if split is not None:
            name = derivative_name(*split)
            local[ident] = sympy.Symbol(name)
            if split not in seen_derivs:
                seen_derivs.append(split)
            continue
        if ident in coordinates:
            local[ident] = sympy.Symbol(ident)
            seen_coords.append(ident)
        elif canonical in _SYMPY_FUNCTIONS:
            local[ident] = _SYMPY_FUNCTIONS[canonical]
        elif canonical in TORCH_FUNCTIONS:
            local[ident] = getattr(sympy, canonical)
        elif ident in _CONSTANTS:
            local[ident] = _CONSTANTS[ident]
        else:
            looks_like = [f for f in fields if ident.startswith(f + "_")]
            if looks_like:
                raise ValueError(
                    f"{ident!r} in {text!r} reads as a derivative of {looks_like[0]!r}, but "
                    f"{ident[len(looks_like[0]) + 1 :]!r} is not made of the coordinates "
                    f"{tuple(coordinates)}"
                )
            local[ident] = sympy.Symbol(ident)
            seen_params.append(ident)
    try:
        expr = parse_expr(text, local_dict=local, transformations=_TRANSFORMATIONS)
    except Exception as exc:  # SyntaxError, TokenError, TypeError from SymPy
        raise ValueError(f"Cannot parse expression {text!r}: {exc}") from exc
    if not isinstance(expr, (sympy.Expr, sympy.Basic)):
        raise ValueError(f"Expression {text!r} is not a formula")
    for func in expr.atoms(sympy.Function):
        name = type(func).__name__
        if name not in TORCH_FUNCTIONS and name not in _SYMPY_FUNCTIONS:
            raise ValueError(
                f"Unknown function {name!r} in {text!r}; available: "
                + ", ".join(sorted(k for k in TORCH_FUNCTIONS if k[0].islower()))
            )
    deriv_names = [derivative_name(f, c) for f, c in seen_derivs]
    args = tuple(seen_fields + deriv_names + seen_coords + seen_params)
    symbols = [sympy.Symbol(a) for a in args]
    fn = sympy.lambdify(symbols, expr, modules=[TORCH_FUNCTIONS, {"pi": math.pi}])
    return CompiledExpression(
        text=text,
        expr=expr,
        fields=tuple(seen_fields),
        derivatives=tuple(seen_derivs),
        coordinates=tuple(seen_coords),
        parameters=tuple(seen_params),
        _fn=fn,
        _args=args,
    )
