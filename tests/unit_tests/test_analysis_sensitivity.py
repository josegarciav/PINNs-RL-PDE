"""Closed-form sensitivities and Fisher information (pinnrl.analysis)."""

import math

import pytest
import torch

from pinnrl.analysis import closed_form_bound, exact_sensitivities, fisher_information
from pinnrl.pdes.catalog import from_catalog
from pinnrl.pdes.custom import CustomPDE

torch.manual_seed(0)
X = torch.rand(64, 1)
T = torch.rand(64, 1)


def test_heat_sensitivity_matches_the_hand_derivative():
    pde = from_catalog("heat")
    J = exact_sensitivities(pde, ["alpha"], X, T)
    assert J.shape == (64, 1, 1)
    u = torch.exp(-0.1 * math.pi**2 * T) * torch.sin(math.pi * X)
    assert torch.allclose(J[:, 0, 0:1], -(math.pi**2) * T * u, atol=1e-6)


def test_sensitivity_at_other_parameter_values():
    pde = from_catalog("heat")
    J = exact_sensitivities(pde, ["alpha"], X, T, values={"alpha": 0.3})
    u = torch.exp(-0.3 * math.pi**2 * T) * torch.sin(math.pi * X)
    assert torch.allclose(J[:, 0, 0:1], -(math.pi**2) * T * u, atol=1e-6)


def test_fisher_scalar_matches_the_closed_form():
    pde = from_catalog("heat")
    sigma = 0.05
    summary = closed_form_bound(pde, ["alpha"], X, T, sigma)
    u = torch.exp(-0.1 * math.pi**2 * T) * torch.sin(math.pi * X)
    info = ((math.pi**2 * T * u) ** 2).sum().item() / sigma**2
    assert summary.information.item() == pytest.approx(info, rel=1e-5)
    assert summary.std.item() == pytest.approx(1 / math.sqrt(info), rel=1e-5)
    assert summary.relative_std({"alpha": 0.1})["alpha"] == pytest.approx(
        10 / math.sqrt(info), rel=1e-5
    )
    assert summary.condition_number == 1.0


def test_two_parameters_give_a_symmetric_bound_with_correlation():
    pde = CustomPDE(
        "u_t + a*u*u_x + b*u_xxx",
        domain={"x": (-15, 15), "t": (0, 2)},
        parameters={"a": 6.0, "b": 1.0},
        exact_solution="3/a/cosh(sqrt(1/b)/2*(x - t))^2",
    )
    x = torch.linspace(-10, 10, 200)[:, None]
    t = torch.rand(200, 1) * 2
    summary = closed_form_bound(pde, ["a", "b"], x, t, 0.01)
    F = summary.information
    assert torch.allclose(F, F.T)
    assert (summary.eigenvalues > 0).all()
    assert summary.eigenvalues[0] >= summary.eigenvalues[1]
    assert abs(summary.correlation[0, 1]) < 1 and summary.correlation[0, 0] == pytest.approx(1.0)
    assert torch.allclose(summary.covariance, torch.linalg.inv(F), rtol=1e-6)


def test_parameter_absent_from_the_solution_has_zero_sensitivity():  # noqa: D103
    pde = CustomPDE(
        "u_t - alpha*u_xx + beta*u",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.1, "beta": 0.0},
        exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",
    )
    J = exact_sensitivities(pde, ["alpha", "beta"], X, T)
    assert J[..., 1].abs().max() == 0
    summary = fisher_information(J, 0.1, ["alpha", "beta"])
    assert math.isinf(summary.condition_number)


def test_per_observation_noise():
    pde = from_catalog("heat")
    J = exact_sensitivities(pde, ["alpha"], X, T)
    sigma = torch.full((64,), 0.05)
    a = fisher_information(J, sigma, ["alpha"]).information
    b = fisher_information(J, 0.05, ["alpha"]).information
    assert torch.allclose(a, b)


def test_without_closed_form_raises():
    pde = from_catalog("burgers")
    with pytest.raises(ValueError, match="exact_solution"):
        exact_sensitivities(pde, ["nu"], X, T)


def test_unidentifiable_parameter_reports_infinite_bound():
    pde = CustomPDE(
        "u_t - alpha*u_xx + beta*u",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.1, "beta": 0.0},
        exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",
    )
    summary = closed_form_bound(pde, ["alpha", "beta"], X, T, 0.1)
    assert summary.identifiable.tolist() == [True, False]
    assert math.isinf(summary.std[1]) and math.isfinite(summary.std[0])
    assert math.isinf(summary.relative_std({"alpha": 0.1, "beta": 1.0})["beta"])
    assert math.isnan(summary.correlation[0, 1])


def test_unknown_parameter_name_raises():
    pde = from_catalog("heat")
    with pytest.raises(ValueError, match="not parameters"):
        exact_sensitivities(pde, ["alhpa"], X, T)


def test_per_field_and_per_entry_noise_shapes():
    J = torch.randn(10, 2, 1, dtype=torch.float64)
    per_field = torch.tensor([0.1, 0.5])
    a = fisher_information(J, per_field, ["a"]).information
    manual = (J[:, 0, 0] ** 2).sum() / 0.1**2 + (J[:, 1, 0] ** 2).sum() / 0.5**2
    assert a.item() == pytest.approx(manual.item(), rel=1e-5)
    per_entry = per_field.expand(10, 2)
    assert fisher_information(J, per_entry, ["a"]).information.item() == pytest.approx(a.item())
    with pytest.raises(ValueError, match="sigma must"):
        fisher_information(J, torch.ones(3), ["a"])


def test_nonsmooth_functions_differentiate():
    pde = CustomPDE(
        "u_t",
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"c": 0.3},
        exact_solution="abs(x - c*t) + heaviside(x - c*t)",
    )
    J = exact_sensitivities(pde, ["c"], X, T)[:, 0, 0]
    expected = -T[:, 0] * torch.sign(X[:, 0] - 0.3 * T[:, 0])
    assert torch.allclose(J, expected, atol=1e-6)
