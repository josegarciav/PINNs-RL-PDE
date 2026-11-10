"""The falling-film evaporator (J01): steady state, dynamics, grid convergence, and its
CustomPDE form with measured inputs, one-sided inlet conditions and an unknown fouling."""

import numpy as np
import pytest
import torch

from pinnrl.process import FallingFilmEvaporator


def test_steady_state_conserves_solids_and_evaporates():
    ev = FallingFilmEvaporator()
    ss = ev.steady_state(nz=101)
    assert ss["W"][0] == pytest.approx(0.1) and ss["X"][0] == pytest.approx(0.1)
    assert np.ptp(ss["W"] * ss["X"]) < 1e-9  # solids conserved along the tube
    assert 0.25 < 1 - ss["W"][-1] / 0.1 < 0.35  # about a third of the feed evaporates per pass
    assert np.all(np.diff(ss["W"]) < 0) and np.all(np.diff(ss["X"]) > 0)
    # fouling lowers the flux and the evaporation
    fouled = FallingFilmEvaporator(R_f=2e-4).steady_state(nz=101)
    assert fouled["q"][0] < ss["q"][0] and fouled["X"][-1] < ss["X"][-1]


def test_dynamics_carry_a_step_with_the_residence_time_and_converge_in_z():
    ev = FallingFilmEvaporator(T_s=lambda t: 355.15 + (5.0 if t > 100 else 0.0))
    ref = ev.reference(t_max=300.0, nz=201, nt=301)
    out = ref.outlet()
    x_before = np.interp(99.0, out["t"], out["X"])
    x_mid = np.interp(110.0, out["t"], out["X"])
    x_after = np.interp(200.0, out["t"], out["X"])
    assert abs(x_before - out["X"][0]) < 1e-3  # steady before the step
    assert x_before < x_mid < x_after  # the outlet reacts within the 20 s residence time
    assert x_after > x_before + 0.02
    # first-order upwind: the outlet error halves with the grid
    finals = [ev.reference(t_max=300.0, nz=nz, nt=61).outlet()["X"][-1] for nz in (51, 101, 201)]
    d1, d2 = abs(finals[0] - finals[1]), abs(finals[1] - finals[2])
    assert 1.5 < d1 / d2 < 2.6
    assert ref.W.interpolate(np.array([5.0]), np.array([50.0])).shape == (1,)


def test_custom_pde_form_matches_the_solver_equations():
    ev = FallingFilmEvaporator(T_s=lambda t: 355.15 + 2.0 * np.sin(t / 30.0))
    ref = ev.reference(t_max=120.0, nz=401, nt=241)
    pde = ev.custom_pde(t_max=120.0)
    assert pde.fields == ("W", "X") and set(pde.inputs) == {"W_in", "X_in", "T_s", "R_f"}
    assert pde.boundary_conditions == {
        "W": {"dirichlet": {"left": "W_in"}},
        "X": {"dirichlet": {"left": "X_in"}},
    }
    # The text equations, evaluated with finite differences of the reference, vanish up to
    # the discretisation error (first order in z and t).
    W, X, z, t = ref.W.u, ref.X.u, ref.z, ref.t
    q = ref.q
    W_t, W_z = np.gradient(W, t, axis=1), np.gradient(W, z, axis=0)
    X_t, X_z = np.gradient(X, t, axis=1), np.gradient(X, z, axis=0)
    P, lam, v = ev.perimeter, ev.latent_heat, ev.velocity
    rW = W_t + v * W_z + P * q / lam
    rX = X_t + v * X_z - X * P * q / (lam * W)
    inner = (slice(2, -2), slice(2, -2))
    assert np.abs(rW[inner]).max() / np.abs(P * q / lam).max() < 0.05
    assert np.abs(rX[inner]).max() / np.abs(X * P * q / (lam * W)).max() < 0.05
    # the inputs read at the right times and the initial condition is the steady profile
    tt = torch.tensor([[0.0], [47.1]])
    expected = torch.tensor([355.15, 355.15 + 2.0 * np.sin(47.1 / 30.0)], dtype=torch.float32)
    assert torch.allclose(pde.inputs["T_s"](tt).ravel(), expected, atol=1e-3)
    zz = torch.tensor([[0.0], [10.0]])
    u0 = pde.initial_condition(zz, torch.zeros(2, 1))
    assert u0[0, 0].item() == pytest.approx(0.1) and u0[1, 1].item() == pytest.approx(
        ref.X.u[-1, 0], rel=1e-3
    )


def test_custom_pde_with_unknown_fouling_trains_normalized():
    ev = FallingFilmEvaporator(R_f=1e-4)
    pde = ev.custom_pde(t_max=60.0, unknown_fouling={"hidden": [8, 8], "initial": 5e-5})
    assert "R_f" in pde.unknown_functions and "R_f" not in pde.inputs
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [16, 16]},
        normalize=True,
        num_collocation_points=64,
        batch_size=64,
        num_boundary_points=16,
        num_initial_points=16,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    assert trainer.config.training.mode == "inverse"
    assert trainer.config.model.input_bounds == [[0.0, 10.0], [0.0, 60.0]]
    x, t = pde.generate_collocation_points(32)
    losses = pde.compute_loss(trainer.model, x, t)
    assert all(np.isfinite(v.item()) for v in losses.values())
    trainer.train_steps(num_steps=2)


def test_reference_rejects_an_unknown_initial_keyword():
    import pytest

    from pinnrl.process import FallingFilmEvaporator

    with pytest.raises(ValueError, match="'steady' or a mapping"):
        FallingFilmEvaporator().reference(t_max=10.0, nz=11, nt=3, initial="stead")
