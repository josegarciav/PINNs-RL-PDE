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

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.func import functional_call, jvp


@dataclass
class SurrogateSensitivity:
    """The Jacobian ``du/dtheta`` of a trained surrogate and how the solve went.

    Attributes:
        jacobian: ``(N, k, p)`` sensitivities at the observation points.
        parameters: The parameter names along the last axis.
        cg_iterations: Conjugate-gradient iterations used per parameter.
        cg_residuals: Relative residual of each solve.
        physics_loss: The physics loss at which the Hessian was taken.
    """

    jacobian: torch.Tensor
    parameters: tuple[str, ...]
    cg_iterations: list[int]
    cg_residuals: list[float]
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

    Args:
        pde: The problem; ``parameters`` must be registered as trainable on it
            so the loss is differentiable with respect to them.
        model: The trained network (a minimum of the physics loss for the
            parameters' current values).
        parameters: Names of the PDE parameters to differentiate with respect to.
        x: Spatial coordinates of the observation points, ``(N, d)``.
        t: Time, ``(N, 1)``; ``None`` for a steady problem.
        n_collocation: Collocation points on which the Hessian is evaluated.
        damping: Tikhonov damping ``lambda`` added to ``H_ww`` (the Hessian of
            a trained network is singular along flat directions).
        cg_iterations: Maximum conjugate-gradient iterations per parameter.
        cg_tol: Relative residual at which the solve stops.
        seed: Seed of the collocation sample.

    Raises:
        ValueError: If a parameter is not trainable on ``pde``.
    """
    trainable = getattr(pde, "_trainable_params", {})
    missing = [p for p in parameters if p not in trainable]
    if missing:
        raise ValueError(
            f"{missing} must be in trainable_parameters so the loss depends on them as tensors"
        )
    was_training = model.training
    model.eval()
    weights = [p for p in model.parameters() if p.requires_grad]
    thetas = [trainable[name] for name in parameters]

    torch.manual_seed(seed)
    xc, tc = pde.generate_collocation_points(n_collocation)
    losses = pde.compute_loss(model, xc.to(pde.device), tc.to(pde.device))
    lw = pde._training_option("loss_weights") or {}
    physics = (
        float(lw.get("pde", lw.get("residual", 1.0))) * losses["residual"]
        + float(lw.get("boundary", 10.0)) * losses["boundary"]
        + float(lw.get("initial", 10.0)) * losses["initial"]
    )
    grads = torch.autograd.grad(physics, weights + thetas, create_graph=True)
    g_w, g_theta = grads[: len(weights)], grads[len(weights) :]
    flat_g = torch.cat([g.reshape(-1) for g in g_w])

    def hvp(v: torch.Tensor) -> torch.Tensor:
        dot = flat_g @ v
        hv = torch.autograd.grad(dot, weights, retain_graph=True, allow_unused=True)
        return (
            torch.cat(
                [
                    (h if h is not None else torch.zeros_like(w)).reshape(-1)
                    for h, w in zip(hv, weights, strict=True)
                ]
            )
            + damping * v
        )

    names = [n for n, p in model.named_parameters() if p.requires_grad]
    shapes = [w.shape for w in weights]

    def unflatten(v: torch.Tensor) -> dict[str, torch.Tensor]:
        out, i = {}, 0
        for name, shape in zip(names, shapes, strict=True):
            n = int(torch.tensor(shape).prod()) if len(shape) else 1
            out[name] = v[i : i + n].reshape(shape)
            i += n
        return out

    inputs = x if t is None else torch.cat([x, t], dim=1)
    inputs = inputs.to(pde.device)
    base = {n: p.detach() for n, p in model.named_parameters()}
    buffers = dict(model.named_buffers())

    def forward(params: dict[str, torch.Tensor]) -> torch.Tensor:
        return functional_call(model, ({**base, **params}, buffers), (inputs,))

    columns, iters, resids = [], [], []
    for g_t in g_theta:
        # H_wtheta: the gradient of dL/dtheta with respect to the weights.
        b = torch.autograd.grad(g_t, weights, retain_graph=True, allow_unused=True)
        b = torch.cat(
            [
                (h if h is not None else torch.zeros_like(w)).reshape(-1)
                for h, w in zip(b, weights, strict=True)
            ]
        )
        dw, n_iter, resid = _conjugate_gradient(hvp, -b, cg_iterations, cg_tol)
        iters.append(n_iter)
        resids.append(resid)
        tangent = unflatten(dw.detach())
        _, du = jvp(forward, (base,), (tangent,))
        columns.append(du.reshape(inputs.shape[0], -1))
    if was_training:
        model.train()
    jac = torch.stack(columns, dim=-1)
    return SurrogateSensitivity(
        jacobian=jac,
        parameters=tuple(parameters),
        cg_iterations=iters,
        cg_residuals=resids,
        physics_loss=float(physics.detach()),
    )


def _conjugate_gradient(apply, b, max_iter, tol):
    """Solve ``A x = b`` for symmetric positive-definite ``A`` given as a function."""
    x = torch.zeros_like(b)
    r = b.clone()
    p = r.clone()
    rs = r @ r
    b_norm = float(b.norm()) or 1.0
    n = 0
    for n in range(1, max_iter + 1):  # noqa: B007
        Ap = apply(p)
        alpha = rs / (p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if float(rs_new.sqrt()) / b_norm < tol:
            rs = rs_new
            break
        p = r + (rs_new / rs) * p
        rs = rs_new
    return x, n, float(rs.sqrt()) / b_norm
