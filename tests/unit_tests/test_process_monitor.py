"""The evaporator's fouling monitor (``pinnrl.process.monitor``)."""

import math

import numpy as np

from pinnrl.process import FallingFilmEvaporator, monitor_fouling

T_MAX, R_END, NZ = 600.0, 1e-4, 11


def _steam(t):
    return 355.15 + math.sin(2 * math.pi * t / 400.0)


def _records(seed=0, offset=0.0):
    """Outlet records of a tube fouling linearly, from the monitor's own grid (no model error)."""
    plant = FallingFilmEvaporator(T_s=lambda t: _steam(t) - offset, R_f=lambda t: R_END * t / T_MAX)
    ref = plant.reference(t_max=T_MAX, nz=NZ, nt=121)
    rng = np.random.default_rng(seed)
    y = np.stack([ref.X.u[-1, 1:], ref.W.u[-1, 1:]], axis=1)
    y[:, 0] += 0.002 * rng.standard_normal(y.shape[0])
    y[:, 1] += 0.001 * rng.standard_normal(y.shape[0])
    return ref.t[1:], y


def test_monitor_tracks_a_ramp_and_predicts_the_cleaning_time():
    times, records = _records()
    mon = monitor_fouling(
        FallingFilmEvaporator(T_s=_steam), times, records, nz=NZ, threshold=2 * R_END, level=0.99
    )
    assert mon.t.shape == mon.fouling.shape == mon.lower.shape == (times.size,)
    assert np.all(mon.lower <= mon.fouling) and np.all(mon.fouling <= mon.upper)
    assert mon.lower[-1] < R_END < mon.upper[-1]
    assert mon.upper[-1] - mon.lower[-1] < 0.4 * R_END
    assert abs(mon.rate[-1] - R_END / T_MAX) < 3 * mon.rate_std[-1]
    # the hindsight estimate is tighter than what was known at mid-record
    k = times.size // 2
    assert mon.smoothed_upper[k] - mon.smoothed_lower[k] < mon.upper[k] - mon.lower[k]
    assert 0.6 < mon.misfit[0] < 1.6 and 0.6 < mon.misfit[1] < 1.6
    tt = mon.time_to_clean  # the ramp reaches twice its end value one window later
    assert tt.lower < T_MAX < tt.upper and tt.probability > 0.95
    text = mon.summary()
    assert "Cleaning threshold" in text and "Rate" in text and mon.steam_bias is None


def test_estimating_the_steam_offset_widens_the_band_and_recovers_the_fouling():
    times, records = _records(seed=1, offset=0.4)  # the record reads 0.4 K high
    model = FallingFilmEvaporator(T_s=_steam)
    trusted = monitor_fouling(model, times, records, nz=NZ, level=0.99)
    hedged = monitor_fouling(model, times, records, nz=NZ, steam_bias_std=0.3, level=0.99)
    width = lambda m: m.upper[-1] - m.lower[-1]  # noqa: E731
    assert width(hedged) > 2 * width(trusted)
    assert hedged.lower[-1] < R_END < hedged.upper[-1]
    assert not (trusted.lower[-1] < R_END < trusted.upper[-1])  # confident and wrong
    offset, std = hedged.steam_bias
    assert abs(offset - 0.4) < 3 * std and "Steam record offset" in hedged.summary()


def test_demo_export_runs_through_the_historian_to_the_monitor(tmp_path):
    from pinnrl.process import Historian, monitor_from_files, write_demo_export

    export, tags = write_demo_export(tmp_path)
    hist = Historian.from_file(export)
    assert hist.tags == ["TT205.PV", "DT310.PV", "FT320.PV"] and hist.span() == (0.0, 1200.0)
    mon = monitor_from_files(export, tags, nz=NZ, threshold=3e-4, level=0.99)
    assert mon.t[-1] == 1200.0 and mon.records.shape == (241, 2)
    assert abs(mon.fouling[-1] - 2e-4) < 0.3e-4  # the coarse grid's bias included
    assert mon.time_to_clean is not None and 200.0 < mon.time_to_clean.median < 1200.0


def test_monitor_from_historian_needs_both_outlet_sensors(tmp_path):
    import pytest
    import yaml

    from pinnrl.process import Historian, monitor_from_historian, write_demo_export

    export, tags = write_demo_export(tmp_path)
    spec = yaml.safe_load(tags.read_text())
    spec["sensors"] = spec["sensors"][:1]
    with pytest.raises(ValueError, match="sensors reporting"):
        monitor_from_historian(Historian.from_file(export).apply(spec))
