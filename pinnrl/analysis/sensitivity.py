"""Sensitivities, Fisher information and the Cramér–Rao bound for identified parameters.

The sensitivity Jacobian ``J[i, j] = du(x_i, t_i) / d theta_j`` says how much each
observation moves when a parameter moves. With Gaussian observation noise of
standard deviation ``sigma``, the Fisher information ``F = J^T J / sigma^2``
bounds the covariance of any unbiased estimator from below by ``F^{-1}`` (the
Cramér–Rao bound). Its eigen-decomposition tells which parameter combinations
the observations determine well and which they do not (sloppy directions).

This module computes ``J`` from a closed-form solution by automatic
differentiation of the compiled expression a
:class:`~pinnrl.pdes.custom.CustomPDE` was given, with respect to per-point
copies of the parameters. It is exact to floating point, so it serves as the
reference for sensitivities obtained any other way (finite differences of a
numerical solver, or the trained surrogate).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class FisherSummary:
    """Fisher information for a parameter set and what it implies.

    Attributes:
        parameters: Parameter names, in the order of every matrix axis.
        information: The Fisher matrix ``F``, shape ``(p, p)``.
        covariance: The Cramér–Rao bound ``F^{-1}`` on the identifiable
            subspace; ``inf`` on the diagonal of a parameter with a component
            in the null space of ``F``.
        std: Lower bound on the standard deviation of each parameter (``inf``
            when the parameter is not identifiable from these observations).
        correlation: Correlation matrix of the bound (``nan`` where undefined).
        identifiable: Per parameter, whether it lies outside the null space of ``F``.
        eigenvalues: Eigenvalues of ``F``, descending.
        eigenvectors: Columns are the corresponding parameter directions.
        condition_number: Ratio of the largest to the smallest eigenvalue
            (``inf`` when ``F`` is singular).
    """

    parameters: tuple[str, ...]
    information: torch.Tensor
    covariance: torch.Tensor
    std: torch.Tensor
    correlation: torch.Tensor
    identifiable: torch.Tensor
    eigenvalues: torch.Tensor
    eigenvectors: torch.Tensor
    condition_number: float

    def relative_std(self, true_values: dict[str, float]) -> dict[str, float]:
        """``std / |true value|`` per parameter (``inf`` when not identifiable)."""
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

    The compiled expression is evaluated with a per-point copy of each
    parameter that requires grad, so the derivative comes from autograd and
    works for every function the expression language accepts (``abs``,
    ``heaviside``, ``sigmoid``, ...). It is computed in double precision.

    Args:
        pde: A ``CustomPDE`` with an expression-valued ``exact_solution``.
        parameters: Names of the parameters to differentiate with respect to.
        x: Spatial coordinates, ``(N, d)``.
        t: Time, ``(N, 1)``; ``None`` for a steady problem.
        values: Parameter values to evaluate at; defaults to the PDE's true
            values (``pde.parameters``), not the current trainable guesses.

    Raises:
        ValueError: If the PDE has no expression-valued exact solution, or a
            name is not one of its parameters.
    """
    compiled = getattr(pde, "_exact", None)
    if not compiled:
        raise ValueError(
            f"{pde.config.name}: exact_sensitivities needs an expression-valued exact_solution"
        )
    unknown = [p for p in parameters if p not in pde.parameters]
    if unknown:
        raise ValueError(
            f"{unknown} are not parameters of {pde.config.name}: {list(pde.parameters)}"
        )
    params = dict(pde.parameters)
    params.update(values or {})
    n = x.shape[0]
    like = torch.ones(n, 1, dtype=torch.float64)
    inputs: dict[str, Any] = {}
    for i, c in enumerate(pde.spatial_coordinates):
        inputs[c] = x[:, i : i + 1].detach().double()
    if t is not None:
        inputs["t"] = t.detach().double()
    # Per-point copies of the parameters to differentiate; the others are constants.
    thetas = {
        name: torch.full((n, 1), float(params[name]), dtype=torch.float64, requires_grad=True)
        for name in parameters
    }
    constants = {name: float(v) for name, v in params.items() if name not in thetas}
    columns = []
    for field in pde.fields:
        expr = compiled.get(field)
        if expr is None:
            columns.append(torch.zeros(n, len(parameters), dtype=torch.float64))
            continue
        u = expr({**constants, **thetas, **inputs}, like)
        per_param = []
        for name in parameters:
            if name not in expr.parameters:
                per_param.append(torch.zeros(n, dtype=torch.float64))
                continue
            (g,) = torch.autograd.grad(u.sum(), thetas[name], retain_graph=True, allow_unused=True)
            per_param.append(
                g.reshape(-1) if g is not None else torch.zeros(n, dtype=torch.float64)
            )
        columns.append(torch.stack(per_param, dim=-1))  # (N, p)
    return torch.stack(columns, dim=1).to(x.dtype)  # (N, k, p)


def fisher_information(
    jacobian: torch.Tensor,
    sigma: float | torch.Tensor,
    parameters: Sequence[str],
    rank_tol: float | None = None,
) -> FisherSummary:
    """Fisher information ``J^T Sigma^{-1} J`` for Gaussian noise and its Cramér–Rao bound.

    Args:
        jacobian: Sensitivities, ``(N, p)`` or ``(N, k, p)`` (flattened over ``k``).
        sigma: Noise standard deviation: a scalar, one value per observation
            point ``(N,)``, one per field ``(k,)`` or one per entry ``(N, k)``.
        parameters: Names for the ``p`` axis.
        rank_tol: Eigenvalues of ``F`` at or below ``rank_tol * lambda_max``
            count as zero (default: ``p`` times the double-precision epsilon).

    Raises:
        ValueError: If ``sigma`` has none of the accepted shapes.
    """
    if jacobian.dim() == 2:
        jacobian = jacobian[:, None, :]
    n, k, p = jacobian.shape
    J = jacobian.reshape(n * k, p).double()
    if isinstance(sigma, torch.Tensor):
        s = sigma.double()
        if s.numel() == 1:
            s = s.reshape(1).expand(n * k)
        elif tuple(s.shape) == (n, k):
            s = s.reshape(-1)
        elif tuple(s.shape) == (n,):
            s = s.repeat_interleave(k)
        elif tuple(s.shape) == (k,):
            s = s.repeat(n)
        else:
            raise ValueError(
                f"sigma must be a scalar or have shape ({n},), ({k},) or ({n}, {k}); "
                f"got {tuple(s.shape)}"
            )
        Jw = J / s[:, None]
    else:
        Jw = J / float(sigma)
    F = Jw.T @ Jw
    eigenvalues, eigenvectors = torch.linalg.eigh(F)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    lam_max = float(eigenvalues[0]) if p else 0.0
    rel_tol = rank_tol if rank_tol is not None else p * torch.finfo(torch.float64).eps
    tol = rel_tol * lam_max
    keep = eigenvalues > tol
    inv = torch.where(keep, 1.0 / eigenvalues.clamp_min(1e-300), torch.zeros_like(eigenvalues))
    cov = eigenvectors @ torch.diag(inv) @ eigenvectors.T
    null = eigenvectors[:, ~keep]
    if null.shape[1]:
        identifiable = (null**2).sum(dim=1) <= 1e-12
    else:
        identifiable = torch.ones(p, dtype=torch.bool)
    std = cov.diagonal().clamp_min(0).sqrt()
    std = torch.where(identifiable, std, torch.full_like(std, math.inf))
    cov = cov.clone()
    cov.diagonal()[~identifiable] = math.inf
    denom = std[:, None] * std[None, :]
    corr = cov / denom
    corr[~torch.isfinite(denom)] = math.nan
    corr.diagonal()[identifiable] = 1.0
    smallest = float(eigenvalues[-1]) if p else 0.0
    cond = lam_max / smallest if smallest > tol and smallest > 0 else math.inf
    return FisherSummary(
        parameters=tuple(parameters),
        information=F,
        covariance=cov,
        std=std,
        correlation=corr,
        identifiable=identifiable,
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
