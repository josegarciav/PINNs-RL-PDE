"""The counter-current tower: the boundary-value steady state, the transient and the CustomPDE form."""

import numpy as np
import pytest
import torch

from pinnrl.process import CounterCurrentDryer, CounterCurrentProfile, CounterCurrentTransient


def test_steady_state_is_physical_and_the_balances_close():
    dr = CounterCurrentDryer()
    prof = dr.steady(n=801)
    assert isinstance(prof, CounterCurrentProfile) and prof.z[-1] == pytest.approx(dr.height)
    times = dr.transit_times()
    assert times["droplets"] == pytest.approx(8.0 / 1.5) and 80 < times["air"] < 130
    # the droplets dry all the way down; the air, read downward, gets hotter and drier
    assert np.all(np.diff(prof.X) < 0) and np.all(np.diff(prof.Y) < 0)
    assert np.all(np.diff(prof.T_a) > 0) and np.all(prof.T_a >= prof.T_p - 1e-6)
    assert prof.T_a[-1] == pytest.approx(433.15, abs=1e-6)  # the inlet air at the bottom
    assert prof.Y[-1] == pytest.approx(0.01, abs=1e-9)
    out = prof.outlet()
    assert 0.05 < out["X"] < 0.09 and 315 < out["T_a"] < 335 and out["T_p"] > 380
    bal = dr.balances(prof)
    assert abs(bal["water"]) < 1e-10
    assert abs(bal["energy"]) < 5e-4
    assert 0.05 < bal["wall_loss_fraction"] < 0.25
    # the hottest air meets the driest powder: the margin is largest at the bottom
    margin = dr.stickiness_margin(prof)
    assert margin[-1] > 40.0 and margin[0] > margin[-1]
    assert dr.glass_transition(0.0) == pytest.approx(dr.T_g_solids)


def test_operating_point_and_closure_move_the_outlet_the_right_way():
    dr = CounterCurrentDryer()
    base = dr.steady().outlet()
    hot = dr.operating_point(T_in=453.15).steady().outlet()
    assert hot["X"] < base["X"] and hot["T_a"] > base["T_a"]
    loaded = dr.operating_point(solids_flow=0.012).steady().outlet()
    assert loaded["X"] > base["X"] and loaded["T_a"] < base["T_a"]
    slow = dr.with_closure(lambda X: 0.5 * dr.droplet.drying_curve(X)).steady().outlet()
    assert slow["X"] > base["X"] and slow["Y"] < base["Y"]
    assert dr.droplet.rate_closure is None


def test_transient_starts_at_rest_and_settles_on_the_new_steady_state():
    dr = CounterCurrentDryer()
    nz = 51
    s = dr.steady_cells(nz)
    assert np.abs(dr.mol_rhs(nz)(0.0, s)).max() < 1e-8
    rest = dr.transient(200.0, nz=nz, nt=21)
    assert isinstance(rest, CounterCurrentTransient) and rest.X.shape == (nz, 21)
    assert np.abs(rest.X - rest.X[:, :1]).max() < 1e-6
    assert np.abs(rest.T_a - rest.T_a[:, :1]).max() < 1e-4
    # the first-order upwind grid is 10 % wet at 51 cells and converges to the shooting profile
    exact = dr.steady().outlet()["X"]
    coarse = dr.steady_cells(51)[50]
    fine = dr.steady_cells(201)[200]
    assert 0.0 < (fine - exact) < 0.5 * (coarse - exact)
    # a step of the inlet air: the powder dries; the response settles with a time constant of about 125 s
    step = dr.operating_point(T_in=lambda t: 433.15 + (15.0 if t > 10.0 else 0.0))
    tr = step.transient(1200.0, nz=nz, nt=61)
    out = tr.outlet()
    assert out["X"][0] == pytest.approx(coarse, abs=1e-6) and out["X"][-1] < 0.8 * out["X"][0]
    final = step.steady_cells(nz, t=1200.0)
    assert out["X"][-1] == pytest.approx(final[nz - 1], rel=2e-3)
    assert out["T_a"][-1] == pytest.approx(final[3 * nz], abs=0.05)
    assert tr.stats["steps"] > 10 and tr.stats["seconds"] > 0
    rec = step.records(tr, probe_heights=(0.5, 4.0))
    assert rec["T_air"].shape == (2, 61) and rec["T_air"][1, -1] > rec["T_air"][1, 0]
    assert rec["Y_out"][-1] > rec["Y_out"][0]  # more water in the air
    with pytest.raises(ValueError, match="probe heights"):
        step.records(tr, probe_heights=[9.0])
    sols = tr.as_reference_solutions()
    assert set(sols) == {"X", "T", "Y", "Ta"} and sols["Ta"].u.shape == (nz, 61)


class _Reference(torch.nn.Module):
    """The steady profiles as a network would return them: ``(z, t) -> [X, T, Y, Ta]``."""

    def __init__(self, prof):
        super().__init__()
        self.prof = prof

    def forward(self, inp):
        z = inp[:, :1]
        grid = torch.tensor(self.prof.z, dtype=z.dtype)
        cols = []
        for name in ("X", "T_p", "Y", "T_a"):
            y = torch.tensor(getattr(self.prof, name), dtype=z.dtype)
            idx = torch.clamp(
                torch.searchsorted(grid, z.detach().contiguous()), 1, grid.numel() - 1
            )
            w = (z - grid[idx - 1]) / (grid[idx] - grid[idx - 1])
            cols.append(y[idx - 1] + w * (y[idx] - y[idx - 1]))
        return torch.cat(cols, dim=1)


def test_custom_pde_residual_vanishes_on_the_steady_profile():
    dr = CounterCurrentDryer()
    prof = dr.steady(n=4001)
    pde = dr.custom_pde(300.0)
    assert list(pde.fields) == ["X", "T", "Y", "Ta"] and set(pde.inputs) == {"Tin", "Yin"}
    z = torch.linspace(0.2, dr.height - 0.2, 40, dtype=torch.float64).reshape(-1, 1)
    z.requires_grad_(True)
    t = torch.full_like(z, 100.0, requires_grad=True)
    scales = dr.residual_scales()
    assert scales["T"] > 100.0 * scales["Ta"] > 0.0 and scales["X"] > 100.0 * scales["Y"]
    res = pde.compute_residual(_Reference(prof).double(), z, t).detach().numpy()
    res = res * np.array([scales[k] for k in ("X", "T", "Y", "Ta")])  # back to physical units
    state = _Reference(prof)(torch.cat([z, t], 1)).detach().numpy()
    rhs = np.stack([np.asarray(dr.rates(*s)) for s in state])
    # the steady profile is a solution: the residual is the interpolation error of its slopes
    assert np.all(np.abs(res) < 2e-2 * np.abs(rhs).max(axis=0) + 1e-6)
    # the initial condition is the steady profile
    u0 = pde.initial_condition(torch.cat([z, t], 1)[:, :1], torch.zeros_like(z))
    ref0 = _Reference(prof)(torch.cat([z, t], 1)).to(u0.dtype)  # interpolated on 201 points
    assert torch.allclose(u0, ref0, rtol=1e-3, atol=1e-3)
    # parameters to identify, and the closure as an unknown function
    pde_i = dr.custom_pde(
        300.0, identify=["Xcr", "ncdc"], initial_guesses={"Xcr": 0.8, "ncdc": 1.0}
    )
    assert set(pde_i.config.trainable_parameters) == {"Xcr", "ncdc"}
    with pytest.raises(ValueError, match="identify accepts"):
        dr.custom_pde(300.0, identify=["lam"])
    pde_u = dr.custom_pde(300.0, unknown_rate={})
    f = pde_u.unknown_functions["f"]
    assert f.anchor == {"at": {"X": dr.droplet.X_eq}, "value": 0.0} and f.monotone
