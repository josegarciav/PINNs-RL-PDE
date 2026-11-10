"""The spray-dryer tower: profiles, balances, records and the CustomPDE form."""

import numpy as np
import pytest
import torch

from pinnrl.process import DryerReference, SprayDryer


def test_profiles_are_physical_and_the_balances_close():
    dr = SprayDryer()
    ref = dr.reference(n=801)
    assert isinstance(ref, DryerReference) and ref.z[-1] == pytest.approx(dr.height)
    assert dr.residence_time() == pytest.approx(8.0 / 1.5)
    assert np.all(np.diff(ref.X) < 0) and np.all(
        np.diff(ref.Y) > 0
    )  # the droplets dry the whole way
    assert np.all(np.diff(ref.T_a) < 0) and np.all(ref.T_a >= ref.T_p - 1e-6)
    out = ref.outlet()
    assert 0.06 < out["X"] < 0.12 and 340 < out["T_a"] < 355
    bal = dr.balances(ref)
    assert abs(bal["water"]) < 1e-10
    assert abs(bal["energy"]) < 5e-4  # quadrature error of the output grid
    assert abs(dr.balances(dr.reference(n=201))["energy"]) > abs(bal["energy"])
    assert 0.05 < bal["wall_loss_fraction"] < 0.25
    assert abs(bal["latent_inconsistency"]) < 0.08


def test_operating_point_and_closure_move_the_records_the_right_way():
    dr = SprayDryer()
    base = dr.records()
    assert base["T_air"].shape == (6,) and base["Y_out"].shape == (1,)
    hot = dr.operating_point(T_in=473.15).records()
    assert hot["T_air"][-1] > base["T_air"][-1] and hot["X_out"][0] < base["X_out"][0]
    loaded = dr.operating_point(solids_flow=0.013).records()
    assert loaded["T_air"][-1] < base["T_air"][-1] and loaded["X_out"][0] > base["X_out"][0]
    slow = dr.with_closure(lambda X: 0.5 * dr.droplet.drying_curve(X)).records()
    assert slow["X_out"][0] > base["X_out"][0] and slow["T_air"][2] > base["T_air"][2]
    assert slow["Y_out"][0] < base["Y_out"][0]
    assert dr.droplet.rate_closure is None  # the original is untouched
    with pytest.raises(ValueError, match="probe heights"):
        dr.records(probe_heights=[9.0])


class _Reference(torch.nn.Module):
    """The reference profiles as a network would return them: ``(x, t) -> [X, T, Y, Ta]``."""

    def __init__(self, ref):
        super().__init__()
        self.ref = ref

    def forward(self, inp):
        t = inp[:, -1:]
        tau = torch.tensor(self.ref.tau, dtype=t.dtype)
        cols = []
        for name in ("X", "T_p", "Y", "T_a"):
            y = torch.tensor(getattr(self.ref, name), dtype=t.dtype)
            idx = torch.clamp(torch.searchsorted(tau, t.detach().contiguous()), 1, tau.numel() - 1)
            w = (t - tau[idx - 1]) / (tau[idx] - tau[idx - 1])
            cols.append(y[idx - 1] + w * (y[idx] - y[idx - 1]))
        return torch.cat(cols, dim=1)


def test_custom_pde_residual_vanishes_on_the_reference():
    dr = SprayDryer()
    ref = dr.reference(n=4001)
    pde = dr.custom_pde()
    assert list(pde.fields) == ["X", "T", "Y", "Ta"]
    t = torch.linspace(0.3, dr.residence_time() - 0.3, 40, dtype=torch.float64).reshape(-1, 1)
    t.requires_grad_(True)
    x = torch.full_like(t, 0.5)
    res = pde.compute_residual(_Reference(ref).double(), x, t)
    rhs = np.stack(
        [
            dr.rhs(float(ti), s)
            for ti, s in zip(
                t.detach().ravel(),
                _Reference(ref)(torch.cat([x, t], 1)).detach().numpy(),
                strict=True,
            )
        ]
    )
    # piecewise-linear interpolation: the residual is the interpolation error of the slopes
    assert np.all(np.abs(res.detach().numpy()) < 2e-2 * np.abs(rhs).max(axis=0) + 1e-6)
    # with the closure unknown: anchored at zero at equilibrium, monotone
    pde_u = dr.custom_pde(unknown_rate={})
    f = pde_u.unknown_functions["f"]
    assert f.anchor == {"at": {"X": dr.droplet.X_eq}, "value": 0.0} and f.monotone
    assert f({"X": torch.tensor([[dr.droplet.X_eq]])}).item() == pytest.approx(0.0, abs=1e-6)


def test_glass_transition_and_stickiness_margin():
    dr = SprayDryer()
    assert dr.glass_transition(0.0) == pytest.approx(dr.T_g_solids)
    assert dr.glass_transition(1e9) == pytest.approx(dr.T_g_water, abs=1e-3)  # pure water
    X = np.linspace(0.0, 1.5, 50)
    assert np.all(np.diff(dr.glass_transition(X)) < 0)  # water plasticises
    slope = (dr.glass_transition(0.09) - dr.glass_transition(0.08)) / 0.01
    assert -11.0 < slope / 100 < -7.0  # about 9 K per 0.01 of moisture
    ref = dr.reference()
    margin = dr.stickiness_margin(ref)
    assert margin.shape == ref.z.shape
    assert margin[-1] == pytest.approx(ref.T_p[-1] - dr.glass_transition(ref.X[-1]))
    assert 5.0 < margin[-1] < 20.0 and margin[0] > 100.0  # wet droplets are far above T_g
    wet = dr.operating_point(T_in=438.15, solids_flow=0.011)
    assert wet.stickiness_margin()[-1] > margin[-1] + 20.0
