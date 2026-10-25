"""Conditioning of PINN training: loss-Hessian and neural-tangent-kernel spectra (experiment K03).

Two spectra describe how hard a PINN is to train, as opposed to how well the
data determine its parameters (:mod:`pinnrl.analysis.sensitivity`):

- the **loss Hessian** ``H = d^2 L / dw^2`` at the current weights, whose extreme
  eigenvalues bound the stable learning rate (``2 / lambda_max`` for gradient
  descent) and whose spread is the condition number of the local quadratic
  model. It is estimated by Lanczos iteration with Hessian-vector products.
- the **neural tangent kernel** of each loss term, ``K = J_w J_w^T`` with
  ``J_w`` the Jacobian of the term's per-point values with respect to the
  weights. Its eigenvalues are the rates at which gradient descent fits each
  mode of that term; a residual kernel whose eigenvalues dwarf the boundary
  kernel's is the gradient-pathology diagnosis of Wang, Teng & Perdikaris
  (2021) and Wang, Yu & Perdikaris (2022).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass
class HessianSpectrum:
    """Ritz values of the loss Hessian from Lanczos iteration.

    Attributes:
        eigenvalues: Ritz values, descending; the extremes converge first.
        iterations: Lanczos steps taken.
        loss: The loss the Hessian belongs to.
    """

    eigenvalues: torch.Tensor
    iterations: int
    loss: float

    @property
    def largest(self) -> float:
        """The largest eigenvalue (the curvature that limits the learning rate)."""
        return float(self.eigenvalues[0])

    @property
    def smallest(self) -> float:
        """The smallest Ritz value (negative when the point is not a minimum)."""
        return float(self.eigenvalues[-1])

    def condition_number(self, floor: float = 1e-12) -> float:
        """Largest over smallest positive Ritz value."""
        positive = self.eigenvalues[self.eigenvalues > floor]
        return float(positive[0] / positive[-1]) if positive.numel() else float("inf")


def loss_hessian_spectrum(
    loss_fn: Callable[[], torch.Tensor],
    parameters: list[torch.nn.Parameter],
    iterations: int = 50,
    seed: int = 0,
) -> HessianSpectrum:
    """Extreme eigenvalues of ``d^2 loss / d parameters^2`` by Lanczos with full reorthogonalization.

    Args:
        loss_fn: Returns the scalar loss as a function of ``parameters`` (it is
            called once; the graph is kept for Hessian-vector products).
        parameters: The tensors to differentiate with respect to.
        iterations: Lanczos steps, hence the number of Ritz values. With 50
            steps the few largest and smallest eigenvalues are converged to
            several digits for typical PINN Hessians.
        seed: Seed of the starting vector (a forked RNG, so the caller's state
            is untouched).
    """
    loss = loss_fn()
    grads = torch.autograd.grad(loss, parameters, create_graph=True, allow_unused=True)
    flat_g = torch.cat(
        [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, parameters, strict=True)
        ]
    )

    def hvp(v: torch.Tensor) -> torch.Tensor:
        hv = torch.autograd.grad(flat_g @ v, parameters, retain_graph=True, allow_unused=True)
        return torch.cat(
            [
                (h if h is not None else torch.zeros_like(p)).reshape(-1)
                for h, p in zip(hv, parameters, strict=True)
            ]
        )

    n = flat_g.numel()
    m = min(iterations, n)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        q = torch.randn(n, device=flat_g.device, dtype=flat_g.dtype)
    q = q / q.norm()
    Q = [q]
    alphas, betas = [], []
    for k in range(m):
        z = hvp(Q[k]).detach()
        alpha = float(Q[k] @ z)
        z = z - alpha * Q[k] - (betas[-1] * Q[k - 1] if k else 0.0)
        # Full reorthogonalization keeps the Ritz values honest at this size.
        for qq in Q:
            z = z - (qq @ z) * qq
        beta = float(z.norm())
        alphas.append(alpha)
        if beta < 1e-10 or k == m - 1:
            break
        betas.append(beta)
        Q.append(z / beta)
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if betas:
        off = torch.tensor(betas[: len(alphas) - 1], dtype=torch.float64)
        T += torch.diag(off, 1) + torch.diag(off, -1)
    ritz = torch.linalg.eigvalsh(T).flip(0)
    return HessianSpectrum(eigenvalues=ritz, iterations=len(alphas), loss=float(loss.detach()))


@dataclass
class KernelSpectrum:
    """Eigenvalues of the neural tangent kernel of a loss term on a batch.

    Attributes:
        eigenvalues: Descending eigenvalues of ``J_w J_w^T`` (one per point).
        trace: Their sum, the total rate at which the term is fitted.
    """

    eigenvalues: torch.Tensor
    trace: float

    def condition_number(self, floor: float = 1e-12) -> float:
        """Largest over smallest eigenvalue above ``floor``."""
        positive = self.eigenvalues[self.eigenvalues > floor]
        return float(positive[0] / positive[-1]) if positive.numel() else float("inf")


def tangent_kernel_spectrum(
    model: torch.nn.Module,
    values_fn: Callable[[torch.nn.Module], torch.Tensor],
) -> KernelSpectrum:
    """Spectrum of the tangent kernel of a per-point quantity of the network.

    Args:
        model: The network.
        values_fn: Maps the network to the per-point values of the term, shape
            ``(N,)`` or ``(N, k)``: the residual on collocation points, the
            boundary mismatch, the network output itself, ... Derivatives
            inside it must be taken with ``create_graph=True`` so that the
            values stay differentiable with respect to the weights.

    The kernel is ``K = J J^T`` with ``J = d values / d weights`` of shape
    ``(N k, P)``, assembled row by row with one backward pass per value, so
    keep ``N k`` in the hundreds.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    values = values_fn(model).reshape(-1)
    rows = []
    for i in range(values.numel()):
        grads = torch.autograd.grad(values[i], params, retain_graph=True, allow_unused=True)
        rows.append(
            torch.cat(
                [
                    (g if g is not None else torch.zeros_like(p)).reshape(-1)
                    for g, p in zip(grads, params, strict=True)
                ]
            )
        )
    J = torch.stack(rows).double()
    K = J @ J.T
    eig = torch.linalg.eigvalsh(K).flip(0).clamp_min(0)
    return KernelSpectrum(eigenvalues=eig, trace=float(eig.sum()))


__all__ = [
    "HessianSpectrum",
    "KernelSpectrum",
    "loss_hessian_spectrum",
    "tangent_kernel_spectrum",
]
