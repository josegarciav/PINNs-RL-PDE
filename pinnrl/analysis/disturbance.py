"""Disturbance and stability analysis of identified parameters and trained surrogates.

Three questions, three tools:

- **How far does the estimate move when the data are disturbed?**
  :func:`perturb_observations` applies the disturbances a plant produces
  (extra noise, sensor bias, dropout, timestamp delay, outliers) to an
  observation set, and :func:`linear_response` gives the first-order
  prediction of the resulting parameter shift,
  ``delta_theta = (J^T Sigma^-1 J)^-1 J^T Sigma^-1 delta_y``,
  so an experiment can compare what the PINN does with what the linear
  theory says (experiment K05).
- **Does the surrogate respect the equation's own stability?**
  :func:`field_response` measures how much the trained solution changes
  when the problem (initial data, boundary data, a parameter) is perturbed,
  to be compared with the reference solution's change (experiment K06).
- **How does the training residual translate into solution error?**
  :func:`stability_ratio` evaluates the residual and the error on a dense
  independent grid and returns ``error / residual``, the empirical
  stability constant of the error estimate
  ``error <= C (residual + quadrature error)`` (experiment K07).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

Observations = dict[str, torch.Tensor]


def perturb_observations(
    obs: Observations,
    kind: str,
    magnitude: float,
    *,
    seed: int = 0,
    sensors: torch.Tensor | None = None,
) -> Observations:
    """A disturbed copy of ``{"x", "t", "u"}`` observations.

    Args:
        obs: Observation tensors of shape ``(N, d)``, ``(N, 1)``, ``(N, k)``.
        kind: One of
            ``"noise"`` (add Gaussian noise of standard deviation ``magnitude``),
            ``"bias"`` (add the constant ``magnitude`` to the values),
            ``"dropout"`` (remove a fraction ``magnitude`` of the points at random),
            ``"delay"`` (the values are attributed to times ``t + magnitude``:
            a sensor whose timestamps lag the physics),
            ``"outliers"`` (replace a fraction ``magnitude`` of the values by
            draws of ten times the data's standard deviation).
        magnitude: Size of the disturbance, in the units the kind names.
        seed: Seed of the random draws.
        sensors: Optional boolean mask ``(N,)`` selecting the points a
            ``"bias"`` or ``"delay"`` applies to (a single faulty probe);
            all points by default.

    Raises:
        ValueError: For an unknown kind.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x, t, u = obs["x"].clone(), obs["t"].clone(), obs["u"].clone()
    n = u.shape[0]
    mask = torch.ones(n, dtype=torch.bool) if sensors is None else sensors.bool().cpu()
    if kind == "noise":
        u = u + magnitude * torch.randn(u.shape, generator=gen).to(u)
    elif kind == "bias":
        u[mask] = u[mask] + magnitude
    elif kind == "dropout":
        keep = torch.rand(n, generator=gen) >= magnitude
        x, t, u = x[keep], t[keep], u[keep]
    elif kind == "delay":
        t[mask] = t[mask] + magnitude
    elif kind == "outliers":
        hit = torch.rand(n, generator=gen) < magnitude
        scale = 10.0 * float(u.std())
        u[hit] = scale * torch.randn((int(hit.sum()), u.shape[1]), generator=gen).to(u)
    else:
        raise ValueError(
            f"unknown disturbance {kind!r}; use noise, bias, dropout, delay or outliers"
        )
    return {"x": x, "t": t, "u": u}


def linear_response(
    jacobian: torch.Tensor, sigma: float | torch.Tensor, delta_y: torch.Tensor
) -> torch.Tensor:
    """First-order parameter shift for a data disturbance ``delta_y``.

    ``delta_theta = (J^T Sigma^-1 J)^-1 J^T Sigma^-1 delta_y`` with
    ``Sigma = sigma^2 I``, the response of a weighted least-squares estimate.
    A ``(N, k, p)`` Jacobian is flattened over ``k``; ``delta_y`` is ``(N,)``
    or ``(N, k)``. Returns ``(p,)``.
    """
    J = jacobian.reshape(-1, jacobian.shape[-1]).double()
    d = delta_y.reshape(-1).double()
    if isinstance(sigma, torch.Tensor):
        s = sigma.reshape(-1).double()
        if s.numel() == 1:
            s = s.expand(J.shape[0])
        Jw, dw = J / s[:, None], d / s
    else:
        Jw, dw = J / float(sigma), d / float(sigma)
    F = Jw.T @ Jw
    return torch.linalg.solve(F, Jw.T @ dw)


@dataclass
class StabilityRatio:
    """Error, residual and their ratio on a dense independent grid.

    Attributes:
        error: Relative L2 error of the field against the closed form.
        residual: Root-mean-square residual on the grid (all equations).
        boundary_error: RMS mismatch of the field against the closed form on
            the spatial boundary (zero when the exact solution is unavailable
            there).
        initial_error: RMS mismatch against the initial condition at ``t = 0``.
        ratio: ``error / residual``, the empirical stability constant.
        points: Grid points used.
    """

    error: float
    residual: float
    boundary_error: float
    initial_error: float
    ratio: float
    points: int


def stability_ratio(
    pde: Any, model: torch.nn.Module, n_points: int = 8192, seed: int = 0
) -> StabilityRatio:
    """``error / residual`` of a trained surrogate on a dense independent sample of the domain.

    The residual is the equation applied to the network at fresh uniform
    points (not the training points), so it includes the quadrature gap;
    the error is against the closed form. Requires ``pde.has_exact_solution``.
    """
    if not pde.has_exact_solution:
        raise ValueError(f"{pde.config.name}: stability_ratio needs an exact solution")
    was_training = model.training
    model.eval()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    d = max(int(pde.dimension), 1)
    lo = torch.tensor([a for a, _ in pde.domain[:d]], dtype=torch.float32)
    hi = torch.tensor([b for _, b in pde.domain[:d]], dtype=torch.float32)
    t0, t1 = pde.time_domain
    x = (lo + (hi - lo) * torch.rand(n_points, d, generator=gen)).to(pde.device)
    t = (t0 + (t1 - t0) * torch.rand(n_points, 1, generator=gen)).to(pde.device)
    residual = pde.compute_residual(model, x, t).detach()
    with torch.no_grad():
        u = model(torch.cat([x, t], dim=1))
        exact = pde.exact_solution(x, t)
        u = u[:, : exact.shape[1]]
        err = float((u - exact).norm() / exact.norm().clamp_min(1e-12))
        # Boundary: both faces of the first spatial axis at the sampled times.
        xb = torch.cat([x.clone(), x.clone()])
        xb[:n_points, 0], xb[n_points:, 0] = lo[0], hi[0]
        tb = torch.cat([t, t])
        ub = model(torch.cat([xb, tb], dim=1))[:, : exact.shape[1]]
        b_err = float((ub - pde.exact_solution(xb, tb)).pow(2).mean().sqrt())
        i_err = 0.0
        if pde.time_dependent and pde.initial_condition is not None:
            ti = torch.full((n_points, 1), float(t0), device=pde.device)
            ui = model(torch.cat([x, ti], dim=1))[:, : exact.shape[1]]
            i_err = float((ui - pde.initial_condition(x, ti)).pow(2).mean().sqrt())
    res = float(residual.pow(2).mean().sqrt())
    if was_training:
        model.train()
    return StabilityRatio(
        error=err,
        residual=res,
        boundary_error=b_err,
        initial_error=i_err,
        ratio=err / res if res > 0 else float("inf"),
        points=n_points,
    )


def field_response(
    model_base: torch.nn.Module,
    model_perturbed: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    reference_delta: torch.Tensor | None = None,
) -> dict[str, float]:
    """Change of the trained field between a base and a perturbed problem, on ``(x, t)``.

    Returns the relative L2 norm of ``u_perturbed - u_base`` and, when the
    reference solution's change ``reference_delta`` is given, the
    amplification ratio ``||delta u_PINN|| / ||delta u_ref||`` and the
    relative error of the PINN's response.
    """
    inputs = torch.cat([x, t], dim=1)
    with torch.no_grad():
        ub = model_base(inputs)[:, :1]
        up = model_perturbed(inputs)[:, :1]
    delta = up - ub
    out = {"response": float(delta.norm() / ub.norm().clamp_min(1e-12))}
    if reference_delta is not None:
        ref = reference_delta.reshape(delta.shape)
        out["amplification"] = float(delta.norm() / ref.norm().clamp_min(1e-12))
        out["response_error"] = float((delta - ref).norm() / ref.norm().clamp_min(1e-12))
    return out


__all__: Sequence[str] = [
    "StabilityRatio",
    "field_response",
    "linear_response",
    "perturb_observations",
    "stability_ratio",
]
