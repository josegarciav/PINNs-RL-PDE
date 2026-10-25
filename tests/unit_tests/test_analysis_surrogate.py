"""Surrogate sensitivities by the implicit-function theorem, checked against the closed form."""

import math

import pytest
import torch

from pinnrl.analysis import exact_sensitivities, surrogate_sensitivities
from pinnrl.pdes.catalog import from_catalog
from pinnrl.pdes.custom import CustomPDE


def _train(pde, steps, **overrides):
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
        **overrides,
    )
    trainer.train_steps(num_steps=steps, resample_every=100)
    return trainer.model


@pytest.fixture(scope="module")
def trained_heat():
    torch.manual_seed(0)
    pde = from_catalog("heat")
    return pde, _train(pde, 1500)


def test_surrogate_sensitivity_matches_the_closed_form(trained_heat):
    pde, model = trained_heat
    torch.manual_seed(1)
    x, t = torch.rand(48, 1), torch.rand(48, 1)
    exact = exact_sensitivities(pde, ["alpha"], x, t)[:, 0, 0]
    result = surrogate_sensitivities(pde, model, ["alpha"], x, t, n_collocation=1024)
    approx = result.jacobian[:, 0, 0]
    assert result.jacobian.shape == (48, 1, 1)
    assert result.cg_residuals[0] < 0.5  # the damped Hessian is ill-conditioned; J is checked
    rel = (approx - exact).norm() / exact.norm()
    corr = torch.corrcoef(torch.stack([approx, exact]))[0, 1]
    assert corr > 0.95, corr
    assert rel < 0.35, rel
    # The call leaves the problem as it found it: alpha is not trainable afterwards.
    assert "alpha" not in pde._trainable_params


def test_unknown_parameter_raises(trained_heat):
    pde, model = trained_heat
    with pytest.raises(ValueError, match="not parameters"):
        surrogate_sensitivities(pde, model, ["kappa"], torch.rand(4, 1), torch.rand(4, 1))


def test_sensitivity_scales_with_time(trained_heat):
    # du/dalpha = -pi^2 t u: zero at t = 0, growing in |t|.
    pde, model = trained_heat
    x = torch.full((3, 1), 0.5)
    t = torch.tensor([[0.0], [0.5], [1.0]])
    J = surrogate_sensitivities(pde, model, ["alpha"], x, t).jacobian[:, 0, 0]
    assert abs(J[0]) < 0.3 * abs(J[1])
    assert J[1] < 0 and J[2] < 0
    expected_ratio = (1.0 * math.exp(-0.1 * math.pi**2)) / (0.5 * math.exp(-0.05 * math.pi**2))
    assert J[2] / J[1] == pytest.approx(expected_ratio, rel=0.35)


def test_rng_state_is_untouched(trained_heat):
    pde, model = trained_heat
    torch.manual_seed(123)
    before = torch.rand(3)
    torch.manual_seed(123)
    surrogate_sensitivities(
        pde, model, ["alpha"], torch.rand(4, 1), torch.rand(4, 1), cg_iterations=5
    )
    after = torch.rand(3)
    assert not torch.equal(before, after)  # the call consumed the same draws it would have anyway
    torch.manual_seed(123)
    torch.rand(4, 1)
    torch.rand(4, 1)
    assert torch.equal(torch.rand(3), after)


def test_steady_problem_takes_t_none():
    torch.manual_seed(0)
    pde = CustomPDE(
        "u_xx + k*u + sin(pi*x)",  # exact u = sin(pi x) / (pi^2 - k)
        domain={"x": (0, 1)},
        parameters={"k": 1.0},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="sin(pi*x)/(pi^2 - k)",
    )
    model = _train(pde, 2000)
    x = torch.linspace(0.1, 0.9, 9)[:, None]
    exact = exact_sensitivities(pde, ["k"], x, None)[:, 0, 0]
    approx = surrogate_sensitivities(pde, model, ["k"], x, None, n_collocation=512).jacobian[
        :, 0, 0
    ]
    assert approx.shape == exact.shape
    assert torch.corrcoef(torch.stack([approx, exact]))[0, 1] > 0.9


def test_parameter_the_loss_ignores_gives_zero(trained_heat):
    pde, model = trained_heat
    pde.parameters["unused"] = 3.0
    try:
        J = surrogate_sensitivities(pde, model, ["unused"], torch.rand(5, 1), torch.rand(5, 1))
        assert J.jacobian.abs().max() == 0
        assert J.cg_iterations == [0]
    finally:
        del pde.parameters["unused"]


def test_minres_solves_an_indefinite_system():
    from pinnrl.analysis.surrogate import _minres

    torch.manual_seed(3)
    A = torch.randn(40, 40, dtype=torch.float64)
    A = (A + A.T) / 2  # symmetric, with eigenvalues of both signs
    b = torch.randn(40, dtype=torch.float64)
    x, n, resid = _minres(lambda v: A @ v, b, 200, 1e-10)
    assert resid < 1e-8
    assert torch.allclose(x, torch.linalg.solve(A, b), atol=1e-6)
    x0, n0, r0 = _minres(lambda v: A @ v, torch.zeros_like(b), 200, 1e-10)
    assert n0 == 0 and r0 == 0.0 and x0.abs().max() == 0
