"""Parametric surrogates: parameters as network inputs, no conditions on their faces, the
surrogate misfit for a swarm, and the spec round trip (K45)."""

import math

import numpy as np
import pytest
import torch
import yaml

from pinnrl import CustomPDE
from pinnrl.pdes.catalog import from_catalog

NET = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def test_parametric_axis_is_an_input_sampled_over_its_range():
    pde = from_catalog("heat", parameters={}, parametric={"alpha": (0.02, 0.5)})
    assert pde.coordinates == ("x", "alpha", "t") and pde.config.input_dim == 3
    assert pde.parametric == ("alpha",) and pde._n_physical == 1
    x, t = pde.generate_collocation_points(200)
    assert x.shape == (200, 2) and 0.02 <= x[:, 1].min() and x[:, 1].max() <= 0.5
    # the exact solution reads alpha from its column
    u = pde.exact_solution(x, t)
    expected = torch.exp(-x[:, 1:2] * math.pi**2 * t) * torch.sin(math.pi * x[:, 0:1])
    assert torch.allclose(u, expected, atol=1e-6)
    assert pde.compute_residual(NET, x, t).shape == (200, 1)
    spec = yaml.safe_load(yaml.safe_dump(pde.to_dict()))
    assert spec["parametric"] == {"alpha": [0.02, 0.5]} and "alpha" not in spec["domain"]
    assert CustomPDE.from_dict(spec).coordinates == ("x", "alpha", "t")
    with pytest.raises(ValueError, match="already"):
        from_catalog("heat", parametric={"alpha": (0.02, 0.5)})


def test_no_boundary_condition_on_the_parametric_faces():
    pde = from_catalog("heat", parameters={}, parametric={"alpha": (0.02, 0.5)})

    class Exact(torch.nn.Module):
        def forward(self, inp):
            return pde.exact_solution(inp[:, :2], inp[:, 2:])

    # the exact solution satisfies the Dirichlet condition on the x faces only; a condition
    # on the alpha faces would be violated and give a nonzero loss
    assert pde._boundary_loss(Exact()).item() == pytest.approx(0.0, abs=1e-10)
    assert pde._initial_loss(Exact()).item() == pytest.approx(0.0, abs=1e-10)


def test_surrogate_misfit_matches_the_closed_form_when_the_model_is_exact():
    pde = from_catalog("heat", parameters={}, parametric={"alpha": (0.02, 0.5)})

    class Exact(torch.nn.Module):
        def forward(self, inp):
            return pde.exact_solution(inp[:, :2], inp[:, 2:])

    data = from_catalog("heat").generate_synthetic_observations(50, 0.05, seed=0)
    thetas = np.linspace(0.05, 0.2, 4)[:, None]
    m = pde.surrogate_misfit(Exact(), data["x"], data["t"], data["u"], thetas)
    x, t, y = data["x"].numpy(), data["t"].numpy(), data["u"].numpy()
    exact = [
        float(((np.exp(-a * math.pi**2 * t) * np.sin(math.pi * x) - y) ** 2).sum())
        for a in thetas.ravel()
    ]
    assert np.allclose(m.numpy(), exact, rtol=1e-4)
    with pytest.raises(ValueError, match="parametric"):
        from_catalog("heat").surrogate_misfit(NET, data["x"], data["t"], data["u"], thetas)


def test_two_parametric_axes_train_a_few_steps():
    pde = from_catalog(
        "kdv_two_parameter", parameters={}, parametric={"alpha": (2, 12), "beta": (0.3, 3)}
    )
    assert pde.coordinates == ("x", "alpha", "beta", "t")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [16, 16]},
        num_collocation_points=64,
        batch_size=64,
        num_boundary_points=16,
        num_initial_points=16,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    trainer.train_steps(num_steps=3)
    assert math.isfinite(pde.validate(trainer.model, 100)["l2_error"])
