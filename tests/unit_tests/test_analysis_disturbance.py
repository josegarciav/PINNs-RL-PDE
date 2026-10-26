"""Disturbance tools (pinnrl.analysis.disturbance)."""

import math

import pytest
import torch

from pinnrl.analysis import (
    exact_sensitivities,
    field_response,
    linear_response,
    perturb_observations,
    stability_ratio,
)
from pinnrl.pdes.catalog import from_catalog


@pytest.fixture
def obs():
    torch.manual_seed(0)
    pde = from_catalog("heat")
    return pde, pde.generate_synthetic_observations(n_points=100, noise_std=0.0, seed=0)


def test_perturbations(obs):
    pde, o = obs
    noisy = perturb_observations(o, "noise", 0.05, seed=1)
    assert noisy["u"].shape == o["u"].shape and (noisy["u"] - o["u"]).std() == pytest.approx(
        0.05, rel=0.3
    )
    biased = perturb_observations(o, "bias", 0.1)
    assert torch.allclose(biased["u"], o["u"] + 0.1)
    half = torch.arange(100) < 50
    part = perturb_observations(o, "bias", 0.1, sensors=half)
    assert torch.allclose(part["u"][half], o["u"][half] + 0.1) and torch.equal(
        part["u"][~half], o["u"][~half]
    )
    dropped = perturb_observations(o, "dropout", 0.5, seed=2)
    assert 30 < dropped["u"].shape[0] < 70 and dropped["x"].shape[0] == dropped["t"].shape[0]
    delayed = perturb_observations(o, "delay", 0.02)
    assert torch.allclose(delayed["t"], o["t"] + 0.02) and torch.equal(delayed["u"], o["u"])
    out = perturb_observations(o, "outliers", 0.1, seed=3)
    assert (out["u"] != o["u"]).sum() > 0
    with pytest.raises(ValueError, match="unknown disturbance"):
        perturb_observations(o, "fog", 1.0)


def test_linear_response_recovers_a_bias_through_the_sensitivity(obs):
    # A disturbance exactly along J shifts the estimate by the matching amount.
    pde, o = obs
    J = exact_sensitivities(pde, ["alpha"], o["x"], o["t"])
    delta_alpha = 0.01
    delta_y = J[:, 0, 0] * delta_alpha
    shift = linear_response(J, 0.05, delta_y)
    assert shift.shape == (1,)
    assert float(shift[0]) == pytest.approx(delta_alpha, rel=1e-6)
    # Orthogonal disturbances give no shift.
    orth = torch.randn(100)
    orth = orth - (orth @ J[:, 0, 0]) / (J[:, 0, 0] @ J[:, 0, 0]) * J[:, 0, 0]
    assert abs(float(linear_response(J, 0.05, orth)[0])) < 1e-6


def test_stability_ratio_on_the_exact_solution_is_near_zero_residual():
    pde = from_catalog("heat")

    class Exact(torch.nn.Module):
        def forward(self, xt):
            return torch.exp(-0.1 * math.pi**2 * xt[:, 1:2]) * torch.sin(math.pi * xt[:, 0:1])

    r = stability_ratio(pde, Exact(), n_points=512)
    assert (
        r.error < 1e-5 and r.residual < 1e-4 and r.initial_error < 1e-5 and r.boundary_error < 1e-5
    )


def test_stability_ratio_on_a_trained_network():
    torch.manual_seed(0)
    pde = from_catalog("heat")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [32, 32], "layer_norm": False},
        num_collocation_points=512,
        batch_size=512,
        num_boundary_points=128,
        num_initial_points=128,
        learning_rate=2e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1, "boundary": 10, "initial": 10},
    )
    trainer.train_steps(num_steps=300, resample_every=100)
    r = stability_ratio(pde, trainer.model, n_points=1024)
    assert 0 < r.error < 1 and r.residual > 0 and r.ratio == pytest.approx(r.error / r.residual)
    assert r.points == 1024


def test_field_response():
    base = torch.nn.Linear(2, 1)
    pert = torch.nn.Linear(2, 1)
    with torch.no_grad():
        pert.weight.copy_(base.weight)
        pert.bias.copy_(base.bias + 0.5)
    x, t = torch.rand(20, 1), torch.rand(20, 1)
    out = field_response(base, pert, x, t, reference_delta=torch.full((20, 1), 0.5))
    assert out["amplification"] == pytest.approx(1.0, rel=1e-6)
    assert out["response_error"] < 1e-6
