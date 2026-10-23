"""Tests for the RAD parameters of ``PDEBase._sample_residual_based``."""

import torch
import torch.nn as nn

from tests.unit_tests.test_utils import create_pde_from_config


def _pde_and_model():
    torch.manual_seed(0)
    pde = create_pde_from_config("heat", torch.device("cpu"), dimension=1)
    model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
    return pde, model


def test_rad_kwargs_are_forwarded():
    """k, c, pool_factor and replacement reach the sampler through the public API."""
    pde, model = _pde_and_model()
    x, t = pde.generate_collocation_points(
        200, strategy="residual_based", model=model, k=2.0, c=1.0, pool_factor=10, replacement=False
    )
    assert x.shape == (200, 1) and t.shape == (200, 1)
    points = torch.cat([x, t], dim=1)
    assert torch.unique(points, dim=0).shape[0] == 200  # no duplicates without replacement


def test_large_c_approaches_uniform_coverage():
    """A large uniform floor c spreads points over the whole domain."""
    pde, model = _pde_and_model()
    x, _ = pde.generate_collocation_points(
        2000, strategy="residual_based", model=model, k=1.0, c=1e6, pool_factor=4
    )
    lo, hi = pde.domain[0]
    counts = torch.histc(x.squeeze(), bins=4, min=lo, max=hi)
    assert counts.min() > 0.15 * 2000


def test_without_model_falls_back_to_uniform():
    """RAD needs a model; without one it returns uniform points."""
    pde, _ = _pde_and_model()
    x, t = pde.generate_collocation_points(100, strategy="residual_based", k=2.0)
    assert x.shape == (100, 1)
