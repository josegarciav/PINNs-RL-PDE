"""The historian adapter: exports in plant units to model inputs and sensor records."""

import numpy as np
import pytest

from pinnrl.pdes import PointSensor
from pinnrl.process import (
    ExtendedKalmanFilter,
    FallingFilmEvaporator,
    Historian,
    augment_random_walk,
)

WIDE = """timestamp;TT205.PV;FT101.PV;DT310.PV
2026-03-01T06:00:00;82.0;360.0;11.0
2026-03-01T06:00:10;82.5;Bad;11.2
2026-03-01T06:00:20;83.0;362.0;
2026-03-01T06:00:30;83.5;364.0;11.6
2026-03-01T06:00:40;84.0;366.0;11.8
"""

LONG = """time,tag,value,quality
0,TT205.PV,82.0,Good
0,FT101.PV,360.0,Good
500,TT205.PV,82.5,Good
500,FT101.PV,999.0,Bad
1000,TT205.PV,83.0,Good
1000,FT101.PV,362.0,Good
1000,TT205.PV,85.0,Good
"""


def test_wide_export_with_dates_bad_cells_and_a_semicolon_delimiter(tmp_path):
    path = tmp_path / "export.csv"
    path.write_text(WIDE)
    hist = Historian.from_file(path)
    assert hist.tags == ["TT205.PV", "FT101.PV", "DT310.PV"]
    t, v = hist.readings("FT101.PV")  # the "Bad" cell is dropped
    np.testing.assert_allclose(t, [0.0, 20.0, 30.0, 40.0])
    np.testing.assert_allclose(v, [360.0, 362.0, 364.0, 366.0])
    assert hist.span("DT310.PV") == (0.0, 40.0)
    steam = hist.series("TT205.PV", offset=273.15)  # degC -> K
    np.testing.assert_allclose(steam.numpy(np.array([0.0, 15.0])), [355.15, 355.90])
    feed = hist.series("FT101.PV", gain=1 / 3600.0, start=10.0, end=40.0)  # kg/h -> kg/s, shifted
    assert feed.span == (0.0, 30.0)
    np.testing.assert_allclose(feed.numpy(np.array([0.0])), [361.0 / 3600.0])  # interpolated end
    # An origin moves the clock's zero.
    later = Historian.from_file(path, origin="2026-03-01T06:00:20")
    assert later.span("TT205.PV") == (-20.0, 20.0)


def test_long_export_with_quality_flags_units_and_repeated_stamps(tmp_path):
    path = tmp_path / "long.csv"
    path.write_text(LONG)
    hist = Historian.from_file(path, time_column="time", quality_column="quality", time_unit="ms")
    t, v = hist.readings("FT101.PV")
    np.testing.assert_allclose(t, [0.0, 1.0])  # ms -> s, the bad row dropped
    np.testing.assert_allclose(v, [360.0, 362.0])
    t, v = hist.readings("TT205.PV")
    np.testing.assert_allclose(t, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(v, [82.0, 82.5, 84.0])  # two readings at 1 s averaged
    with pytest.raises(KeyError, match="no tag"):
        hist.readings("XX")


def test_resample_interpolates_averages_and_holds():
    t = np.arange(0.0, 10.0, 1.0)
    hist = Historian({"a": (t, t**2)})
    times = np.array([2.0, 4.0, 6.0])
    np.testing.assert_allclose(hist.resample("a", times), [4.0, 16.0, 36.0])
    # the mean of the readings in (0, 2], (2, 4], (4, 6]
    np.testing.assert_allclose(hist.resample("a", times, "mean"), [2.5, 12.5, 30.5])
    np.testing.assert_allclose(hist.resample("a", np.array([2.5, 9.7]), "last"), [4.0, 81.0])
    with pytest.raises(ValueError, match="aggregate"):
        hist.resample("a", times, "median")


def _evaporator_export(tmp_path, t_max=300.0, nz=6):
    """A wide export of an evaporator run in plant units: degC, kg/h, percent solids."""
    plant = FallingFilmEvaporator(T_s=lambda t: 355.15 + np.sin(2 * np.pi * t / 150.0), R_f=1.5e-4)
    ref = plant.reference(t_max=t_max, nz=nz, nt=int(t_max) + 1)
    out = ref.outlet()
    rng = np.random.default_rng(0)
    lines = ["timestamp,TT205.PV,FT101.PV,DT310.PV,FT320.PV"]
    for k, t in enumerate(out["t"]):
        steam = 355.15 + np.sin(2 * np.pi * t / 150.0) - 273.15
        x = 100.0 * (out["X"][k] + 0.0005 * rng.standard_normal())
        w = 3600.0 * (out["W"][k] + 0.0002 * rng.standard_normal())
        lines.append(f"{t:.1f},{steam:.4f},360.0,{x:.5f},{w:.4f}")
    path = tmp_path / "evaporator.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


TAG_MAP = {
    "inputs": {
        "T_s": {"tag": "TT205.PV", "offset": 273.15},
        "W_in": {"tag": "FT101.PV", "gain": 1 / 3600.0},
    },
    "sensors": [
        {
            "tag": "DT310.PV",
            "name": "outlet X",
            "location": 10.0,
            "expression": "X",
            "gain": 0.01,
            "noise": 0.0005,
            "every": 5.0,
            "aggregate": "mean",
        },
        {
            "tag": "FT320.PV",
            "name": "outlet W",
            "location": 10.0,
            "expression": "W",
            "gain": 1 / 3600.0,
            "noise": 0.0002,
            "every": 5.0,
            "aggregate": "mean",
        },
    ],
}


def test_tag_map_gives_inputs_sensors_and_filter_records(tmp_path):
    hist = Historian.from_file(_evaporator_export(tmp_path))
    data = hist.apply({**TAG_MAP, "window": [50.0, 250.0]})
    assert data.t_max == 200.0 and data.t_start == 50.0
    assert set(data.inputs) == {"T_s", "W_in"}
    np.testing.assert_allclose(data.inputs["W_in"].numpy(np.array([10.0])), [0.1])
    steam_at_start = 355.15 + np.sin(2 * np.pi * 50.0 / 150.0)
    np.testing.assert_allclose(
        data.inputs["T_s"].numpy(np.array([0.0])), [steam_at_start], atol=1e-3
    )
    x_sensor, w_sensor = data.sensors
    assert isinstance(x_sensor, PointSensor) and x_sensor.expression == "X"
    np.testing.assert_allclose(x_sensor.times, np.arange(5.0, 200.0 + 1e-9, 5.0))
    assert 0.10 < float(x_sensor.values.mean()) < 0.15  # percent -> fraction
    times, records = data.records(["outlet X", "outlet W"])
    assert records.shape == (40, 2)
    np.testing.assert_allclose(records[:, 1], w_sensor.values.numpy().ravel())
    with pytest.raises(ValueError, match="no sensor named"):
        data.records(["outlet T"])


def test_tag_map_from_yaml_and_its_errors(tmp_path):
    import yaml

    hist = Historian.from_file(_evaporator_export(tmp_path, t_max=60.0))
    path = tmp_path / "tags.yaml"
    path.write_text(yaml.safe_dump(TAG_MAP))
    data = hist.apply(path)
    assert data.t_max == 60.0 and len(data.sensors) == 2
    with pytest.raises(ValueError, match="does not have"):
        hist.apply({"inputs": {"T_s": {"tag": "TT999.PV"}}})
    uneven = {
        "sensors": [
            {**TAG_MAP["sensors"][0]},
            {**TAG_MAP["sensors"][1], "every": 10.0},
        ]
    }
    with pytest.raises(ValueError, match="different timestamps"):
        hist.apply(uneven).records()


def test_filter_on_historian_records_recovers_the_fouling(tmp_path):
    """End to end: export -> tag map -> the evaporator filter, in the model's units."""
    nz = 6
    data = Historian.from_file(_evaporator_export(tmp_path, nz=nz)).apply(TAG_MAP)
    times, records = data.records(["outlet X", "outlet W"])
    model = FallingFilmEvaporator(
        T_s=data.inputs["T_s"].numpy, W_in=data.inputs["W_in"].numpy, R_f=1.5e-4
    )
    f, h, s0 = model.fouling_state_space(nz)
    n = s0.size
    ekf = ExtendedKalmanFilter(
        augment_random_walk(f, n, 1),
        lambda t, x: h(t, x[:n]),
        Q=np.concatenate([np.full(n, 1e-12), [(2e-7) ** 2]]),
        R=np.array([0.0005**2, 0.0002**2]) / 5.0,  # five-reading means
    )
    x0 = np.concatenate([s0, [0.5e-4]])
    P0 = np.diag(np.concatenate([np.full(n, 1e-6), [(1e-4) ** 2]]))
    res = ekf.run(times, records, x0, P0)
    assert abs(res.filtered[-1, -1] - 1.5e-4) < 0.1e-4


def test_parquet_export(tmp_path):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    table = pa.table({"timestamp": [0.0, 1.0, 2.0], "TT205.PV": [82.0, None, 83.0]})
    path = tmp_path / "export.parquet"
    pq.write_table(table, path)
    hist = Historian.from_file(path)
    t, v = hist.readings("TT205.PV")
    np.testing.assert_allclose(t, [0.0, 2.0])
    np.testing.assert_allclose(v, [82.0, 83.0])


def test_records_of_a_sensor_without_values_is_a_clear_error():
    import numpy as np
    import pytest

    from pinnrl.pdes import PointSensor
    from pinnrl.process import TwinData

    data = TwinData(inputs={}, sensors=[PointSensor(1.0, np.arange(3.0), name="s")], t_max=2.0)
    with pytest.raises(ValueError, match="carry no records"):
        data.records()
