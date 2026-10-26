"""Conditioning and identifiability diagnostics for identified parameters.

The first layer is exact: sensitivities by symbolic differentiation of a
closed-form solution, the Fisher information they imply, and its Cramér–Rao
bound. Later layers (sensitivities of a numerical reference or of the trained
surrogate, profile likelihood, loss-Hessian spectra) are checked against it.
"""

from pinnrl.analysis.disturbance import (
    StabilityRatio,
    field_response,
    linear_response,
    perturb_observations,
    stability_ratio,
)
from pinnrl.analysis.sensitivity import (
    FisherSummary,
    closed_form_bound,
    exact_sensitivities,
    fisher_information,
)
from pinnrl.analysis.spectrum import (
    HessianSpectrum,
    KernelSpectrum,
    loss_hessian_spectrum,
    tangent_kernel_spectrum,
)
from pinnrl.analysis.surrogate import SurrogateSensitivity, surrogate_sensitivities

__all__ = [
    "FisherSummary",
    "StabilityRatio",
    "field_response",
    "linear_response",
    "perturb_observations",
    "stability_ratio",
    "HessianSpectrum",
    "KernelSpectrum",
    "SurrogateSensitivity",
    "closed_form_bound",
    "exact_sensitivities",
    "fisher_information",
    "loss_hessian_spectrum",
    "surrogate_sensitivities",
    "tangent_kernel_spectrum",
]
