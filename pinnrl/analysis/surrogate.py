"""Sensitivities of a trained PINN surrogate to its PDE parameters (experiment K01).

The network output does not depend on the PDE parameters directly: they enter
through the residual. At a minimum ``w*(theta)`` of the physics loss
``L(w, theta)``, the implicit-function theorem gives

    dw/dtheta = -H_ww^{-1} H_wtheta,

with ``H`` the Hessian of ``L``. The sensitivity of the solution at the
observation points follows by the chain rule, ``J = (du_w/dw) (dw/dtheta)``.
Both Hessian blocks are applied matrix-free with double backward, and the
linear solve is a damped conjugate gradient. The result is checked against
:func:`~pinnrl.analysis.sensitivity.exact_sensitivities` on problems with a
closed form; see ``test_analysis_surrogate.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.func import functional_call, jvp


@dataclass
class SurrogateSensitivity:
    """The Jacobian ``du/dtheta`` of a trained surrogate and how the solve went.

    Attributes:
        jacobian: ``(N, k, p)`` sensitivities at the observation points.
        parameters: The parameter names along the last axis.
        cg_iterations: Conjugate-gradient iterations used per parameter.
        cg_residuals: Relative residual of each solve.
        damping: Damping each solve ended with (raised from the requested value
            when the damped Hessian showed negative curvature).
        physics_loss: The physics loss at which the Hessian was taken.
    """

    jacobian: torch.Tensor
    parameters: tuple[str, ...]
    cg_iterations: list[int]
    cg_residuals: list[float]
    damping: list[float]
    physics_loss: float


def surrogate_sensitivities(
    pde: Any,
    model: torch.nn.Module,
    parameters: Sequence[str],
    x: torch.Tensor,
    t: torch.Tensor | None,
    *,
    n_collocation: int = 2048,
    damping: float = 1e-3,
    cg_iterations: int = 300,
    cg_tol: float = 1e-5,
    seed: int = 0,
) -> SurrogateSensitivity:
    """``du/dtheta`` at ``(x, t)`` from the trained surrogate, by the implicit-function theorem.

    Pass the same ``pde`` the model was trained with: the Hessian is taken of
    that object's physics loss (its loss weights, boundary and initial point
    counts, loss function and device), at the current values of its
    parameters. Parameters that are not registered as trainable are made
    trainable for the duration of the call, so a forward-trained problem works
    as it is.

    Args:
        pde: The problem the model was trained on.
        model: The trained network, a minimum of the physics loss for the
            parameters' current values.
        parameters: Names of the PDE parameters to differentiate with respect to.
        x: Spatial coordinates of the observation points, ``(N, d)``.
        t: Time, ``(N, 1)``; ``None`` for a steady problem (the network then
            receives ``t = 0``, as the trainer does).
        n_collocation: Collocation points on which the Hessian is evaluated.
        damping: Tikhonov damping ``lambda`` added to ``H_ww`` (the Hessian of
            a trained network is singular along flat directions). When the
            damped Hessian still shows negative curvature, which happens when
            the network is not at an exact minimum, the solve restarts with
            ten times the damping, up to six times.
        cg_iterations: Maximum conjugate-gradient iterations per parameter.
        cg_tol: Relative residual at which the solve stops.
        seed: Seed of the collocation sample (drawn in a forked RNG, so the
            caller's random state is untouched).

    Raises:
        ValueError: If a name is not a parameter of ``pde``, or a solve diverges.
    """
    unknown = [p for p in parameters if p not in getattr(pde, "parameters", {})]
    if unknown:
        raise ValueError(f"{unknown} are not parameters of {pde.config.name}")
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    try:
        with _temporarily_trainable(pde, parameters) as thetas:
            result = _sensitivities(
                pde,
                model,
                parameters,
                thetas,
                x,
                t,
                device,
                n_collocation,
                damping,
                cg_iterations,
                cg_tol,
                seed,
            )
    finally:
        if was_training:
            model.train()
    return result


@contextmanager
def _temporarily_trainable(pde, parameters):
    """Register ``parameters`` as trainable on ``pde`` for the block, at their current values."""
    registry: nn.ParameterDict = pde._trainable_params
    added = []
    for name in parameters:
        if name not in registry:
            value = float(pde.parameters[name])
            registry[name] = nn.Parameter(torch.tensor(value, device=pde.device))
            added.append(name)
    try:
        yield [registry[name] for name in parameters]
    finally:
        for name in added:
            del registry[name]


def _sensitivities(
    pde,
    model,
    parameters,
    thetas,
    x,
    t,
    device,
    n_collocation,
    damping,
    cg_iterations,
    cg_tol,
    seed,
):
    weights = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    shapes = [w.shape for w in weights]

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        xc, tc = pde.generate_collocation_points(n_collocation)
    losses = pde.compute_loss(model, xc.to(device), tc.to(device))
    lw = pde._training_option("loss_weights") or {}
    physics = (
        float(lw.get("pde", lw.get("residual", 1.0))) * losses["residual"]
        + float(lw.get("boundary", 10.0)) * losses["boundary"]
        + float(lw.get("initial", 10.0)) * losses["initial"]
    )
    grads = torch.autograd.grad(physics, weights + thetas, create_graph=True, allow_unused=True)

    def flatten(grads_or_none):
        return torch.cat(
            [
                (g if g is not None else torch.zeros_like(w)).reshape(-1)
                for g, w in zip(grads_or_none, weights, strict=True)
            ]
        )

    flat_g = flatten(grads[: len(weights)])
    g_theta = grads[len(weights) :]

    def hvp(v: torch.Tensor, lam: float) -> torch.Tensor:
        hv = torch.autograd.grad(flat_g @ v, weights, retain_graph=True, allow_unused=True)
        return flatten(hv) + lam * v

    def unflatten(v: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = {}, 0
        for name, shape in zip(names, shapes, strict=True):
            size = math.prod(shape)
            out[name] = v[i : i + size].reshape(shape)
            i += size
        return out

    if t is None:
        t = torch.zeros(x.shape[0], 1, dtype=x.dtype)
    inputs = torch.cat([x, t], dim=1).to(device)
    base = {n: p.detach() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def forward(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(model, ({**base, **params}, buffers), (inputs,))

    with torch.no_grad():
        out_dim = model(inputs).shape[1]
    columns, iters, resids, dampings = [], [], [], []
    for name, g_t in zip(parameters, g_theta, strict=True):
        if g_t is None or not g_t.requires_grad:
            # The loss does not depend on this parameter: no sensitivity.
            columns.append(torch.zeros(inputs.shape[0], out_dim, device=device))
            iters.append(0)
            resids.append(0.0)
            dampings.append(damping)
            continue
        b = flatten(torch.autograd.grad(g_t, weights, retain_graph=True, allow_unused=True))
        dw, n_iter, resid, lam = _damped_solve(hvp, -b, damping, cg_iterations, cg_tol, name)
        iters.append(n_iter)
        resids.append(resid)
        dampings.append(lam)
        _, du = jvp(forward, (base,), (unflatten(dw.detach()),))
        columns.append(du.reshape(inputs.shape[0], -1))
    return SurrogateSensitivity(
        jacobian=torch.stack(columns, dim=-1),
        parameters=tuple(parameters),
        cg_iterations=iters,
        cg_residuals=resids,
        damping=dampings,
        physics_loss=float(physics.detach()),
    )


def _damped_solve(hvp, b, damping, max_iter, tol, name=""):
    """Solve ``(H + lam I) x = b`` by CG, raising ``lam`` tenfold whenever negative curvature appears."""
    lam = damping
    for _attempt in range(7):
        x, n, resid, indefinite = _conjugate_gradient(
            lambda v, lam=lam: hvp(v, lam), b, max_iter, tol, name
        )
        if not indefinite:
            return x, n, resid, lam
        lam *= 10.0
    return x, n, resid, lam


def _conjugate_gradient(apply, b, max_iter, tol, name=""):
    """Solve ``A x = b`` for symmetric positive-definite ``A`` given as a function.

    Returns ``(x, iterations, relative residual, indefinite)``. A zero
    right-hand side returns ``x = 0`` at once. Non-positive or non-finite
    curvature stops the iteration and flags ``indefinite``; the caller then
    raises the damping.

    Raises:
        ValueError: If the iteration produces non-finite values.
    """
    b_norm = float(b.norm())
    x = torch.zeros_like(b)
    if b_norm == 0.0:
        return x, 0, 0.0, False
    r = b.clone()
    p = r.clone()
    rs = r @ r
    n = 0
    for n in range(1, max_iter + 1):  # noqa: B007
        Ap = apply(p)
        curvature = p @ Ap
        if not torch.isfinite(curvature) or curvature <= 0:
            return x, n, float(rs.sqrt()) / b_norm, True
        alpha = rs / curvature
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if not torch.isfinite(rs_new):
            raise ValueError(f"conjugate gradient diverged for parameter {name!r}")
        if float(rs_new.sqrt()) / b_norm < tol:
            rs = rs_new
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, n, float(rs.sqrt()) / b_norm, False
