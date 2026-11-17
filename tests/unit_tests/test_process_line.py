"""The process line: the evaporator passes in series feeding the dryer, and the propagation of each unit's error."""

import numpy as np
import pytest

from pinnrl.process import (
    CounterCurrentDryer,
    FallingFilmEvaporator,
    LineReference,
    ProcessLine,
    SprayDryer,
)


def test_default_line_concentrates_and_dries():
    line = ProcessLine.default()
    assert len(line.passes) == 3 and isinstance(line.dryer, SprayDryer)
    s = line.steady()
    X, W = s["X"], s["W"]
    assert X[0] == pytest.approx(0.1) and W[0] == pytest.approx(0.1)
    assert np.all(np.diff(X) > 0) and np.all(np.diff(W) < 0)
    assert 0.38 < X[-1] < 0.45 and 0.02 < W[-1] < 0.03
    assert np.allclose(
        np.array(W) * np.array(X), 0.01
    )  # the solids are conserved through the passes
    assert 0.06 < s["powder"] < 0.1 and 0.0 < s["margin"] < 30.0
    # the concentrate's solids and flow set the dryer's feed
    dryer = line.dryer_for(W[-1], X[-1])
    assert dryer.droplet.solids_fraction == pytest.approx(X[-1])
    assert dryer.solids_flow == pytest.approx(W[-1] * X[-1])
    assert dryer.droplet.d0 == line.dryer.droplet.d0  # the atomiser's droplet size is untouched
    # one pass alone is the standard tube: the first pass reproduces it
    single = FallingFilmEvaporator(T_s=353.15).steady_state()
    assert X[1] == pytest.approx(single["X"][-1], rel=1e-6)
    with pytest.raises(ValueError, match="one entry per pass"):
        ProcessLine.default(n_passes=2, steam=[353.15])


def test_fouling_in_one_pass_reaches_the_powder():
    clean = ProcessLine.default()
    fouled = ProcessLine.default(fouling=[2e-4, 0.0, 0.0])
    a, b = clean.steady(), fouled.steady()
    assert b["X"][1] < a["X"][1] and b["X"][-1] < a["X"][-1]  # less concentrated
    assert (
        b["powder"] > a["powder"] and b["margin"] > a["margin"]
    )  # the dryer gets more water and leaves it wetter


def test_reference_chains_the_passes_and_follows_the_steady_state():
    line = ProcessLine.default(fouling=[lambda t: 2e-4 * t / 1200.0, 0.0, 0.0])
    ref = line.reference(1200.0, nz=51, dryer_every=20)
    assert isinstance(ref, LineReference) and len(ref.passes) == 3
    assert ref.t.size == 241 and ref.X.shape == ref.powder.shape == ref.margin.shape == (241,)
    # the inlet row of pass k+1 is the outlet of pass k
    for k in range(2):
        assert np.allclose(ref.passes[k + 1].X.u[0], ref.passes[k].X.u[-1])
        assert np.allclose(ref.passes[k + 1].W.u[0], ref.passes[k].W.u[-1])
    s0, s1 = line.steady(t=0.0), line.steady(t=1200.0)
    assert ref.X[0] == pytest.approx(s0["X"][-1], rel=5e-3)
    assert ref.powder[0] == pytest.approx(s0["powder"], rel=2e-2)
    # the fouling grows slowly against the passes' residence time: the end is near the new steady state
    assert ref.X[-1] == pytest.approx(s1["X"][-1], rel=2e-2) and ref.X[-1] < ref.X[0]
    assert ref.powder[-1] > ref.powder[0]
    rec = line.records(ref)
    assert (
        rec["X_pass"].shape == (3, 241)
        and rec["T_air"].shape == (6, 241)
        and rec["Y_out"].shape == (241,)
    )
    assert np.allclose(rec["X_pass"][-1], ref.X) and np.allclose(rec["X_powder"], ref.powder)
    assert (
        rec["T_air"][-1, -1] < rec["T_air"][-1, 0]
    )  # a wetter concentrate cools the dryer's outlet air
    assert rec["Y_out"][-1] > rec["Y_out"][0]


def test_propagation_budgets_the_powder_between_the_units():
    line = ProcessLine.default()
    std = {"R_f_0": 2e-5, "R_f_2": 2e-5, "X_cr": 0.03, "d0": 25e-6, "W_feed": 0.002}
    out = line.propagate(std, samples=40, seed=1)
    lin = out["linear"]
    for target in ("X_concentrate", "powder", "margin"):
        shares = lin[target]["shares"]
        assert set(shares) == set(std) and sum(shares.values()) == pytest.approx(1.0)
        assert lin[target]["std"] > 0.0
        mc = out["monte_carlo"][target]["std"]
        assert (
            0.5 * lin[target]["std"] < mc < 2.5 * lin[target]["std"]
        )  # the map bends, but not wildly
    # the dryer's parameters do not reach the concentrate; the fouling reaches everything
    assert (
        lin["X_concentrate"]["shares"]["X_cr"] == 0.0
        and lin["X_concentrate"]["shares"]["d0"] == 0.0
    )
    assert lin["X_concentrate"]["shares"]["R_f_0"] > 0.0 and lin["powder"]["shares"]["R_f_0"] > 0.0
    assert (
        out["sensitivity"]["powder"]["R_f_0"] > 0.0
    )  # fouling leaves water in the concentrate and in the powder
    with pytest.raises(ValueError, match="neither"):
        line.propagate({"nonsense": 1.0})


def test_line_accepts_a_counter_current_dryer():
    line = ProcessLine(passes=ProcessLine.default().passes, dryer=CounterCurrentDryer())
    s = line.steady()
    assert 0.0 < s["powder"] < 0.2 and s["margin"] > 0.0
    ref = line.reference(60.0, nz=21, dryer_every=6, probe_heights=(0.5, 4.0))
    assert ref.powder.shape == (13,) and ref.dryer is not None
    rec = line.records(ref)
    assert rec["T_air"].shape == (2, 13) and rec["Y_out"].shape == (13,)
