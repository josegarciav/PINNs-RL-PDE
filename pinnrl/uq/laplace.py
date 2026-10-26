"""Gaussian (Laplace) posteriors for identified parameters, and likelihood weights.

With Gaussian observation noise the PINN loss is a negative log-posterior
(see the Bayesian view in the docs) and the data weight is set by the noise:
:func:`likelihood_weights` turns the noise level into loss weights. At the
trained optimum, the posterior over the parameters is approximated by a
Gaussian whose covariance is the inverse Fisher information of the surrogate's
sensitivities, ``(J^T J / sigma^2)^-1``: :func:`laplace_posterior`. It reuses
:func:`pinnrl.analysis.surrogate_sensitivities`, so it inherits that method's
validity condition: the surrogate must resolve the solution (K01).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from pinnrl.analysis import fisher_information, surrogate_sensitivities


def likelihood_weights(
    n_data: int,
    sigma_data: float,
    n_residual: int,
    sigma_residual: float = 1.0,
    *,
    boundary: float = 10.0,
    initial: float = 10.0,
) -> dict[str, float]:
    """Loss weights that make the mean-squared loss a negative log-likelihood up to scale.

    The loss uses means, so a term with ``n`` points and noise ``sigma``
    contributes ``n / (2 sigma^2)`` times its mean square to ``-log L``. With
    the residual weight fixed at 1, the data weight is
    ``(n_data / sigma_data^2) / (n_residual / sigma_residual^2)``, where
    ``sigma_residual`` is the model-error scale admitted for the equation.
    """
    data = (n_data / sigma_data**2) / (n_residual / sigma_residual**2)
    return {"residual": 1.0, "boundary": boundary, "initial": initial, "data": float(data)}


@dataclass
class LaplacePosterior:
    """A Gaussian posterior over identified parameters.

    Attributes:
        parameters: Names.
        mean: The point estimate (the trained values).
        covariance: ``(p, p)``.
        std: Marginal standard deviations.
        correlation: ``(p, p)``.
    """

    parameters: tuple[str, ...]
    mean: torch.Tensor
    covariance: torch.Tensor
    std: torch.Tensor
    correlation: torch.Tensor

    def interval(self, level: float = 0.9) -> torch.Tensor:
        """Central interval per parameter, ``(2, p)``."""
        from pinnrl.uq.metrics import _normal_quantile

        z = _normal_quantile(0.5 + level / 2)
        return torch.stack([self.mean - z * self.std, self.mean + z * self.std])

    def sample(self, n: int, seed: int = 0) -> torch.Tensor:
        """``n`` draws, ``(n, p)``."""
        gen = torch.Generator().manual_seed(seed)
        L = torch.linalg.cholesky(self.covariance)
        z = torch.randn(n, self.mean.numel(), generator=gen, dtype=self.covariance.dtype)
        return self.mean.double() + z @ L.T


def laplace_posterior(
    pde: Any,
    model: torch.nn.Module,
    parameters: Sequence[str],
    sigma: float | torch.Tensor,
    x: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
    **sensitivity_kwargs: Any,
) -> LaplacePosterior:
    """Gaussian posterior over ``parameters`` at the trained estimate.

    The covariance is the inverse Fisher information of the surrogate's
    sensitivities at the observation points (``pde.observation_data`` by
    default, or ``x, t``). This is the Laplace approximation restricted to the
    parameters with the generalized Gauss–Newton curvature; it coincides with
    the Cramér–Rao bound at the estimate.
    """
    if x is None:
        obs = pde.observation_data
        if obs is None:
            raise ValueError(
                "laplace_posterior needs observation points: pde.observation_data or x, t"
            )
        x, t = obs["x"], obs["t"]
    sens = surrogate_sensitivities(pde, model, parameters, x, t, **sensitivity_kwargs)
    summary = fisher_information(sens.jacobian, sigma, parameters)
    values = pde.get_trainable_parameter_values()
    mean = torch.tensor(
        [values.get(n, float(pde.get_parameter(n))) for n in parameters], dtype=torch.float64
    )
    return LaplacePosterior(
        parameters=tuple(parameters),
        mean=mean,
        covariance=summary.covariance,
        std=summary.std,
        correlation=summary.correlation,
    )


__all__ = ["LaplacePosterior", "laplace_posterior", "likelihood_weights"]
