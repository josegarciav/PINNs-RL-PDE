"""Sensitivities, Fisher information and the Cramér–Rao bound for identified parameters.

The sensitivity Jacobian ``J[i, j] = du(x_i, t_i) / d theta_j`` says how much each
observation moves when a parameter moves. With Gaussian observation noise of
standard deviation ``sigma``, the Fisher information ``F = J^T J / sigma^2``
bounds the covariance of any unbiased estimator from below by ``F^{-1}`` (the
Cramér–Rao bound). Its eigen-decomposition tells which parameter combinations
the observations determine well and which they do not (sloppy directions).

This module computes ``J`` from a closed-form solution, by symbolic
differentiation of the expression a :class:`~pinnrl.pdes.custom.CustomPDE` was
given. It is exact, so it serves as the reference for sensitivities obtained
any other way (finite differences of a numerical solver, or the trained
surrogate).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import sympy
import torch

from pinnrl.pdes.symbolic import TORCH_FUNCTIONS, CompiledExpression


@dataclass
class FisherSummary:
    """Fisher information for a parameter set and what it implies.

    Attributes:
        parameters: Parameter names, in the order of every matrix axis.
        information: The Fisher matrix ``F``, shape ``(p, p)``.
        covariance: The Cramér–Rao bound ``F^{-1}``, shape ``(p, p)``.
        std: Lower bound on the standard deviation of each parameter.
        correlation: Correlation matrix of the bound.
        eigenvalues: Eigenvalues of ``F``, descending.
        eigenvectors: Columns are the corresponding parameter directions.
        condition_number: Ratio of the largest to the smallest eigenvalue.
    """

    parameters: tuple[str, ...]
    information: torch.Tensor
    covariance: torch.Tensor
    std: torch.Tensor
    correlation: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    condition_number: float

    def relative_std(self, true_values: dict[str, float]) -> dict[str, float]:
        """``std / |true value|`` per parameter."""
        return {
            name: float(self.std[i]) / abs(float(true_values[name]))
            for i, name in enumerate(self.parameters)
        }


def exact_sensitivities(
    pde: Any,
    parameters: Sequence[str],
    x: torch.Tensor,
    t: torch.Tensor | None = None,
    values: dict[str, float] | None = None,
) -> torch.Tensor:
    """``du/dtheta`` of a closed-form solution at ``(x, t)``, shape ``(N, k, p)``.

    Args:
        pde: A ``CustomPDE`` with an expression-valued ``exact_solution``.
        parameters: Names of the parameters to differentiate with respect to.
        x: Spatial coordinates, ``(N, d)``.
        t: Time, ``(N, 1)``; ``None`` for a steady problem.
        values: Parameter values to evaluate at; defaults to the PDE's true
            values (``pde.parameters``), not the current trainable guesses.

    Raises:
        ValueError: If the PDE has no expression-valued exact solution or a
            parameter does not appear in it.
    """
    compiled = getattr(pde, "_exact", None)
    if not compiled:
        raise ValueError(
            f"{pde.config.name}: exact_sensitivities needs an expression-valued exact_solution"
        )
    params = dict(pde.parameters)
    params.update(values or {})
    like = x[:, 0:1]
    inputs: dict[str, Any] = {}
    for i, c in enumerate(pde.spatial_coordinates):
        inputs[c] = x[:, i : i + 1]
    if t is not None:
        inputs["t"] = t
    columns = []
    for field in pde.fields:
        expr: CompiledExpression | None = compiled.get(field)
        per_param = []
        for name in parameters:
            if expr is None or name not in expr.parameters:
                per_param.append(torch.zeros(x.shape[0], dtype=x.dtype, device=x.device))
                continue
            per_param.append(_differentiate(expr, name)(inputs, params, like).reshape(-1))
        columns.append(torch.stack(per_param, dim=-1))  # (N, p)
    return torch.stack(columns, dim=1)  # (N, k, p)


def _differentiate(expr: CompiledExpression, name: str):
    """A torch callable for ``d expr / d name`` with the parent's argument order."""
    d_expr = sympy.diff(expr.expr, sympy.Symbol(name))
    symbols = [sympy.Symbol(a) for a in expr._args]
    fn = sympy.lambdify(symbols, d_expr, modules=[TORCH_FUNCTIONS, {"pi": math.pi}])

    def evaluate(inputs, params, like):
        values = {**params, **inputs}
        out = fn(*(values[a] for a in expr._args))
        if not isinstance(out, torch.Tensor):
            return torch.full_like(like, float(out))
        return out.expand_as(like) if out.shape != like.shape else out

    return evaluate


def fisher_information(
    jacobian: torch.Tensor, sigma: float | torch.Tensor, parameters: Sequence[str]
) -> FisherSummary:
    """Fisher information ``J^T Sigma^{-1} J`` for Gaussian noise and its Cramér–Rao bound.

    Args:
        jacobian: Sensitivities, ``(N, p)`` or ``(N, k, p)`` (flattened over ``k``).
        sigma: Noise standard deviation, a scalar or one value per observation
            row (``(N,)`` or ``(N, k)``).
        parameters: Names for the ``p`` axis.
    """
    J = jacobian.reshape(-1, jacobian.shape[-1]).double()
    if isinstance(sigma, torch.Tensor):
        s = sigma.reshape(-1).double()
        if s.numel() == 1:
            s = s.expand(J.shape[0])
        elif s.numel() != J.shape[0]:
            s = s.repeat_interleave(J.shape[0] // s.numel())
        Jw = J / s[:, None]
    else:
        Jw = J / float(sigma)
    F = Jw.T @ Jw
    eigenvalues, eigenvectors = torch.linalg.eigh(F)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    cov = torch.linalg.pinv(F)
    std = cov.diagonal().clamp_min(0).sqrt()
    denom = std[:, None] * std[None, :]
    corr = torch.where(denom > 0, cov / denom.clamp_min(1e-300), torch.zeros_like(cov))
    smallest = float(eigenvalues[-1])
    cond = float(eigenvalues[0]) / smallest if smallest > 0 else math.inf
    return FisherSummary(
        parameters=tuple(parameters),
        information=F,
        covariance=cov,
        std=std,
        correlation=corr,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        condition_number=cond,
    )


def closed_form_bound(
    pde: Any,
    parameters: Sequence[str],
    x: torch.Tensor,
    t: torch.Tensor | None,
    sigma: float | torch.Tensor,
) -> FisherSummary:
    """The Cramér–Rao bound for ``parameters`` from observations at ``(x, t)`` with noise ``sigma``.

    A convenience around :func:`exact_sensitivities` and :func:`fisher_information`.
    """
    J = exact_sensitivities(pde, parameters, x, t)
    return fisher_information(J, sigma, parameters)
