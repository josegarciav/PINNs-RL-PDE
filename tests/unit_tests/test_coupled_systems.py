"""Coupled multi-field systems: per-field boundary conditions, field weights, and a two-field
reaction-diffusion problem with a closed form that trains (roadmap v0.6 item 2)."""

import pytest
import torch
import yaml

from pinnrl import CustomPDE
from pinnrl.pdes.catalog import from_catalog

torch.manual_seed(0)
X = torch.rand(50, 1)
T = torch.rand(50, 1)


def _two_heat(**kwargs):
    return CustomPDE(
        ["u_t - u_xx", "v_t - v_xx"],
        domain={"x": (0, 1), "t": (0, 1)},
        fields=["u", "v"],
        initial_condition={"u": "1", "v": "x"},
        **kwargs,
    )


class _Model(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, inp):
        return self.fn(inp[:, :1], inp[:, 1:])


def test_per_field_boundary_conditions_apply_each_kind_to_its_field():
    pde = _two_heat(
        boundary_conditions={"u": {"dirichlet": {"value": 1.0}}, "v": {"neumann": {"value": 0.0}}}
    )
    assert pde.boundary_conditions == {
        "u": {"dirichlet": {"value": 1.0}},
        "v": {"neumann": {"value": 0.0}},
    }
    good = _Model(lambda x, t: torch.cat([1 + 0 * x, 2 + 0 * x], dim=1))  # u = 1, v_x = 0
    assert pde._boundary_loss(good).item() == pytest.approx(0.0, abs=1e-12)
    bad = _Model(lambda x, t: torch.cat([0 * x, x], dim=1))  # u = 0 (error 1), v_x = 1 (error 1)
    assert pde._boundary_loss(bad).item() == pytest.approx(1.0)
    # a field left out has no condition
    only_u = _two_heat(boundary_conditions={"u": {"dirichlet": {"value": 0.0}}})
    assert only_u._boundary_loss(_Model(lambda x, t: torch.cat([0 * x, x], dim=1))).item() == 0.0


def test_field_weights_scale_residual_initial_and_boundary_terms():
    pde = _two_heat(
        boundary_conditions={"u": {"dirichlet": {"value": 1.0}}, "v": {"neumann": {"value": 0.0}}},
        field_weights={"v": 4.0},
    )
    bad = _Model(lambda x, t: torch.cat([0 * x, x], dim=1))
    assert pde._boundary_loss(bad).item() == pytest.approx(2.5)  # mean of [1, 1, 4, 4]
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2))
    losses = pde.compute_loss(net, X, T)
    r = pde.compute_residual(net, X, T)
    assert losses["residual"].item() == pytest.approx(
        ((r[:, 0] ** 2 + 4 * r[:, 1] ** 2).mean() / 2).item(), rel=1e-5
    )
    u = net(torch.cat(pde._seeded_points("initial", 100), dim=1))
    x0 = pde._seeded_points("initial", 100)[0]
    ic_err = u - torch.cat([torch.ones_like(x0), x0], dim=1)
    assert losses["initial"].item() == pytest.approx(
        ((ic_err[:, 0] ** 2 + 4 * ic_err[:, 1] ** 2).mean() / 2).item(), rel=1e-5
    )
    assert losses["total"].item() == pytest.approx(
        (losses["residual"] + 10 * losses["boundary"] + 10 * losses["initial"]).item(), rel=1e-6
    )
    unweighted = _two_heat(boundary_conditions={"u": {"dirichlet": {"value": 1.0}}})
    assert unweighted.compute_loss(net, X, T)["residual"].item() == pytest.approx(
        (r**2).mean().item(), rel=1e-5
    )


def test_spec_round_trip_and_errors():
    pde = _two_heat(
        boundary_conditions={
            "u": {"dirichlet": {"left": 1.0, "right": 0.0}},
            "v": {"periodic": {}},
        },
        field_weights={"u": 2.0},
    )
    spec = yaml.safe_load(yaml.safe_dump(pde.to_dict()))
    rebuilt = CustomPDE.from_dict(spec)
    assert rebuilt.boundary_conditions == pde.boundary_conditions
    assert rebuilt._field_weights == [2.0, 1.0]
    with pytest.raises(ValueError, match="mixes field names"):
        _two_heat(boundary_conditions={"u": {"dirichlet": {}}, "periodic": {}})
    with pytest.raises(ValueError, match="not fields"):
        _two_heat(field_weights={"w": 1.0})
    with pytest.raises(ValueError, match="Unknown boundary condition"):
        _two_heat(boundary_conditions={"u": {"robin": {}}})


def test_exact_type_dirichlet_per_field_takes_that_fields_column():
    pde = from_catalog(
        "reaction_diffusion_pair",
        boundary_conditions={
            "u": {"dirichlet": {"type": "exact"}},
            "v": {"dirichlet": {"type": "exact"}},
        },
    )
    exact = _Model(lambda x, t: pde.exact_solution(x, t))
    assert pde._boundary_loss(exact).item() == pytest.approx(0.0, abs=1e-10)


def test_two_field_reaction_diffusion_trains_to_its_closed_form():
    torch.manual_seed(0)
    pde = from_catalog("reaction_diffusion_pair")
    exact = _Model(lambda x, t: pde.exact_solution(x, t))
    assert pde.compute_residual(exact, X, T).abs().max().item() < 1e-5
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [32, 32, 32], "layer_norm": False},
        num_collocation_points=512,
        batch_size=512,
        num_boundary_points=128,
        num_initial_points=128,
        learning_rate=3e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    before = pde.validate(trainer.model, 2000)["l2_error"]
    trainer.train_steps(num_steps=600, resample_every=50)
    after = pde.validate(trainer.model, 2000)["l2_error"]
    assert after < 0.15 and after < before / 2
