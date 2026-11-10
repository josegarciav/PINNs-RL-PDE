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


def test_closure_read_off_a_clean_record_and_fitted_to_the_unknown_function():
    dr = DryingDroplet()
    ref = dr.reference(200.0, nt=2001)
    times = np.arange(1.0, 200.0 + 1e-9, 1.0)
    m = np.interp(times, ref.t, ref.m)
    T = np.interp(times, ref.t, ref.T)
    X_pts, f_pts = dr.closure_from_record(times, m, T)
    truth = dr.drying_curve(X_pts)
    assert np.linalg.norm(f_pts - truth) / np.linalg.norm(truth) < 0.05
    assert X_pts.min() > dr.X_eq and bool((f_pts >= 0).all()) and bool((f_pts <= 1).all())
    pde = dr.custom_pde(200.0, unknown_rate={"hidden": [8, 8]})
    fn = pde.unknown_functions["f"]
    torch.manual_seed(0)
    mse = fn.fit({"X": X_pts}, f_pts, steps=500)
    X = torch.linspace(dr.X_eq, dr.X0, 100).reshape(-1, 1)
    with torch.no_grad():
        fitted = fn({"X": X}).ravel().numpy()
    exact = dr.drying_curve(X.ravel().numpy())
    assert mse < 5e-3
    assert np.linalg.norm(fitted - exact) / np.linalg.norm(exact) < 0.1


def test_record_misfit_is_one_for_the_plant_curve_and_ranks_wrong_closures():
    """The reference-solver guard: noise level for the truth, large for a wrong or unphysical closure."""
    dr = DryingDroplet()
    ref = dr.reference(200.0, nt=2001)
    times = np.arange(1.0, 200.0 + 1e-9, 1.0)
    rng = np.random.default_rng(0)
    sigma_m, sigma_T = 0.005 * dr.m0, 0.5
    mass = np.interp(times, ref.t, ref.m) + sigma_m * rng.standard_normal(times.size)
    temp = np.interp(times, ref.t, ref.T) + sigma_T * rng.standard_normal(times.size)
    truth = dr.record_misfit(times, mass, temp, sigma_m, sigma_T)
    assert 0.7 < truth["mass"] < 1.3 and 0.7 < truth["temperature"] < 1.3
    assert truth["total"] == pytest.approx(0.5 * (truth["mass"] + truth["temperature"]))
    # The same curve passed as a closure is the same model.
    same = dr.record_misfit(times, mass, temp, sigma_m, sigma_T, closure=dr.drying_curve)
    assert same["total"] == pytest.approx(truth["total"], rel=1e-6)
    # A closure with the wrong falling-rate exponent misses the record several-fold. (A curve
    # scaled down uniformly would not: the droplet warms up and the rate barely changes.)
    from dataclasses import replace

    wrong = dr.record_misfit(
        times, mass, temp, sigma_m, sigma_T, closure=replace(dr, n=2.0).drying_curve
    )
    assert wrong["total"] > 4 * truth["total"]
    # A constant closure dries the droplet past zero volume: an infinite misfit, not NaN.
    broken = dr.record_misfit(times, mass, temp, sigma_m, sigma_T, closure=lambda X: 0.0 * X + 0.8)
    assert broken["total"] == np.inf
