"""Surrogate sensitivities by the implicit-function theorem, checked against the closed form."""

import math

import pytest
import torch

from pinnrl.analysis import exact_sensitivities, surrogate_sensitivities
from pinnrl.pdes.catalog import from_catalog


@pytest.fixture(scope="module")
def trained_heat():
    torch.manual_seed(0)
    pde = from_catalog("heat")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [32, 32, 32], "layer_norm": False},
        num_collocation_points=1024,
        batch_size=1024,
        num_boundary_points=256,
        num_initial_points=256,
        learning_rate=2e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0},
    )
    trainer.train_steps(num_steps=1500, resample_every=100)
    return trainer.model


def test_surrogate_sensitivity_matches_the_closed_form(trained_heat):
    inverse = from_catalog(
        "heat", trainable_parameters=["alpha"], parameter_initial_guesses={"alpha": 0.1}
    )
    torch.manual_seed(1)
    x, t = torch.rand(48, 1), torch.rand(48, 1)
    exact = exact_sensitivities(inverse, ["alpha"], x, t)[:, 0, 0]
    result = surrogate_sensitivities(inverse, trained_heat, ["alpha"], x, t, n_collocation=1024)
    approx = result.jacobian[:, 0, 0]
    assert result.jacobian.shape == (48, 1, 1)
    assert (
        result.cg_residuals[0] < 0.5
    )  # the damped Hessian is ill-conditioned; the Jacobian is checked below
    rel = (approx - exact).norm() / exact.norm()
    corr = torch.corrcoef(torch.stack([approx, exact]))[0, 1]
    assert corr > 0.95, corr
    assert rel < 0.35, rel


def test_requires_a_trainable_parameter(trained_heat):
    pde = from_catalog("heat")
    with pytest.raises(ValueError, match="trainable_parameters"):
        surrogate_sensitivities(pde, trained_heat, ["alpha"], torch.rand(4, 1), torch.rand(4, 1))


def test_sensitivity_scales_with_time(trained_heat):
    # du/dalpha = -pi^2 t u: zero at t = 0, growing in |t|.
    inverse = from_catalog(
        "heat", trainable_parameters=["alpha"], parameter_initial_guesses={"alpha": 0.1}
    )
    x = torch.full((3, 1), 0.5)
    t = torch.tensor([[0.0], [0.5], [1.0]])
    J = surrogate_sensitivities(inverse, trained_heat, ["alpha"], x, t).jacobian[:, 0, 0]
    assert abs(J[0]) < 0.3 * abs(J[1])
    assert J[1] < 0 and J[2] < 0
    expected_ratio = (1.0 * math.exp(-0.1 * math.pi**2)) / (0.5 * math.exp(-0.05 * math.pi**2))
    assert J[2] / J[1] == pytest.approx(expected_ratio, rel=0.35)
