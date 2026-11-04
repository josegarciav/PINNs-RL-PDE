"""The drying droplet (J07): the reference trajectory, its CustomPDE form and the unknown closure."""

import numpy as np
import pytest
import torch

from pinnrl.pdes.symbolic import compile_expression
from pinnrl.process import DryingDroplet


def test_reference_dries_through_constant_and_falling_rate_periods():
    dr = DryingDroplet()
    ref = dr.reference(300.0)
    assert ref.X[0] == pytest.approx(dr.X0) and ref.T[0] == pytest.approx(dr.T0)
    assert np.all(np.diff(ref.X) <= 1e-12)  # moisture only falls
    assert ref.X[-1] == pytest.approx(dr.X_eq, abs=2e-3)  # ends at equilibrium
    assert ref.T[-1] == pytest.approx(363.15, abs=0.5)  # and at the air temperature
    wet_bulb = ref.T[(ref.t > 10) & (ref.t < 40)]
    assert np.ptp(wet_bulb) < 0.5 and 300 < wet_bulb.mean() < 310  # a wet-bulb plateau
    assert ref.d[-1] < 0.6 * ref.d[0]  # shrinkage
    assert ref.m[0] == pytest.approx(dr.m0)


def test_custom_pde_equations_vanish_on_the_reference():
    dr = DryingDroplet(T_air=lambda t: 363.15 + 5.0 * np.sin(t / 50.0))
    ref = dr.reference(200.0, nt=2001)
    pde = dr.custom_pde(200.0)
    assert pde.fields == ("X", "T") and list(pde.inputs) == ["Tair"]
    X_t, T_t = np.gradient(ref.X, ref.t), np.gradient(ref.T, ref.t)
    rhs = np.array([dr.rhs(t, [x, T]) for t, x, T in zip(ref.t, ref.X, ref.T, strict=True)])
    n = ref.t.size
    col = lambda a: torch.tensor(np.asarray(a), dtype=torch.float64).reshape(-1, 1)  # noqa: E731
    values = {p: float(v) for p, v in pde.parameters.items()}
    values.update(
        X=col(ref.X),
        T=col(ref.T),
        X_t=col(X_t),
        T_t=col(T_t),
        Tair=col([dr.T_air(t) for t in ref.t]),
        _x=torch.zeros(n, 1, dtype=torch.float64),
        t=col(ref.t),
    )
    for k, text in enumerate(pde.to_dict()["equations"]):
        expr = compile_expression(text, ["X", "T"], ["_x", "t"], list(pde.parameters) + ["Tair"])
        residual = expr(values, values["X"]).numpy().ravel()
        assert np.abs(residual[5:-5]).max() / np.abs(rhs[:, k]).max() < 1e-2  # the gradient's error


def test_unknown_closure_builds_bounded_monotone_and_trains():
    dr = DryingDroplet()
    pde = dr.custom_pde(100.0, unknown_rate={"hidden": [8, 8]})
    f = pde.unknown_functions["f"]
    assert f.inputs == ("X",)
    X = torch.linspace(0.0, dr.X0, 50).reshape(-1, 1)
    truth = f.truth({"X": X}).ravel()
    assert truth[0] == 0.0 and truth[-1] == 1.0 and bool((truth.diff() >= 0).all())
    guess = f({"X": X}).ravel()
    assert bool((guess >= 0).all()) and bool((guess <= 1).all())
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [16, 16]},
        normalize=True,
        num_collocation_points=64,
        batch_size=64,
        num_initial_points=16,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    assert trainer.config.training.mode == "inverse"
    x, t = pde.generate_collocation_points(32)
    losses = pde.compute_loss(trainer.model, x, t)
    assert all(np.isfinite(v.item()) for v in losses.values())
    trainer.train_steps(num_steps=2)
