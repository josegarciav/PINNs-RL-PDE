"""Conditioning and identifiability diagnostics for identified parameters.

The first layer is exact: sensitivities by symbolic differentiation of a
closed-form solution, the Fisher information they imply, and its Cramér–Rao
bound. Later layers (sensitivities of a numerical reference or of the trained
surrogate, profile likelihood, loss-Hessian spectra) are checked against it.
"""

from pinnrl.analysis.sensitivity import (
    FisherSummary,
    closed_form_bound,
    exact_sensitivities,
    fisher_information,
)

__all__ = ["FisherSummary", "closed_form_bound", "exact_sensitivities", "fisher_information"]
