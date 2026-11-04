"""Measured inputs: time series from files, inputs in equations and boundary values,
expression-valued boundary conditions, and heat driven by a CSV (roadmap v0.6 item 3)."""

import math

import numpy as np
import pytest
import torch
import yaml

from pinnrl import CustomPDE, TimeSeries

torch.manual_seed(0)
NET = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
X = torch.rand(30, 1)
T = torch.rand(30, 1)


def _write_csv(path, t, values, time_column="t", value_column="T_left"):
    lines = [f"{time_column},{value_column}"] + [f"{a},{b}" for a, b in zip(t, values, strict=True)]
    path.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------- TimeSeries
def test_time_series_interpolates_and_holds_outside_the_record():
    ts = TimeSeries([0.0, 1.0, 2.0], [0.0, 10.0, 0.0], name="q")
    out = ts(torch.tensor([[-1.0], [0.0], [0.25], [1.0], [1.5], [3.0]]))
    assert out.shape == (6, 1)
    assert torch.allclose(out.ravel(), torch.tensor([0.0, 0.0, 2.5, 10.0, 5.0, 0.0]))
    assert ts.span == (0.0, 2.0)
    assert np.allclose(ts.numpy(np.array([0.5, 5.0])), [5.0, 0.0])
    assert out.dtype == torch.float32
    with pytest.raises(ValueError, match="strictly increasing"):
        TimeSeries([0.0, 1.0, 1.0], [0, 1, 2])
    with pytest.raises(ValueError, match="at least two"):
        TimeSeries([0.0], [1.0])
    with pytest.raises(ValueError, match="times but"):
        TimeSeries([0.0, 1.0], [1.0])


def test_time_series_from_csv_and_npz(tmp_path):
    t = np.linspace(0, 1, 6)
    _write_csv(tmp_path / "left.csv", t, 2 * t)
    ts = TimeSeries.from_file(tmp_path / "left.csv")
    assert ts.name == "T_left" and ts.value_column == "T_left"
    assert np.allclose(ts.values, 2 * t)
    np.savez(tmp_path / "left.npz", time=t, q=3 * t, other=t)
    ts2 = TimeSeries.from_file(tmp_path / "left.npz", time_column="time", value_column="q")
    assert np.allclose(ts2.values, 3 * t)
    # unsorted rows and a non-numeric row are tolerated
    (tmp_path / "messy.csv").write_text("t,v\n1.0,10\nbad,1\n0.0,0\n0.5,5\n")
    ts3 = TimeSeries.from_file(tmp_path / "messy.csv")
    assert np.allclose(ts3.t, [0.0, 0.5, 1.0]) and np.allclose(ts3.values, [0, 5, 10])
    with pytest.raises(ValueError, match="no column 'time'"):
        TimeSeries.from_file(tmp_path / "left.csv", time_column="time")
    with pytest.raises(ValueError, match=".csv or .npz"):
        TimeSeries.from_file(tmp_path / "left.json")
    spec = ts.to_spec()
    assert spec["path"].endswith("left.csv") and spec["value_column"] == "T_left"
    assert TimeSeries.from_spec("q", spec).values.tolist() == ts.values.tolist()
    inline = TimeSeries([0, 1], [1, 2]).to_spec()
    assert inline == {"t": [0.0, 1.0], "values": [1.0, 2.0]}
    with pytest.raises(ValueError, match="give a TimeSeries"):
        TimeSeries.from_spec("q", {"nonsense": 1})


# ------------------------------------------------------------------ in a PDE
def test_input_in_the_equation_is_read_at_each_points_time():
    q = TimeSeries([0.0, 1.0], [0.0, 1.0], name="q")  # q(t) = t
    forced = CustomPDE(
        "u_t - u_xx - q", domain={"x": (0, 1), "t": (0, 1)}, inputs={"q": q}, initial_condition="0"
    )
    plain = CustomPDE("u_t - u_xx - t", domain={"x": (0, 1), "t": (0, 1)}, initial_condition="0")
    assert torch.allclose(
        forced.compute_residual(NET, X, T), plain.compute_residual(NET, X, T), atol=1e-6
    )
    assert "q" in repr(forced)
    with pytest.raises(ValueError, match="clashes"):
        CustomPDE(
            "u_t - u_xx - q",
            domain={"x": (0, 1), "t": (0, 1)},
            parameters={"q": 1},
            inputs={"q": q},
        )
    with pytest.raises(ValueError, match="time-dependent"):
        CustomPDE("u_xx - q", domain={"x": (0, 1)}, inputs={"q": q})


def test_expression_and_input_boundary_values():
    q = TimeSeries([0.0, 0.5, 1.0], [1.0, 0.0, 1.0], name="q")
    pde = CustomPDE(
        "u_t - u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        inputs={"q": q},
        boundary_conditions={"dirichlet": {"left": "q", "right": "0.5*sin(2*pi*t)"}},
        initial_condition="1 - x",
    )

    class Exact(torch.nn.Module):
        def forward(self, inp):
            x, t = inp[:, :1], inp[:, 1:]
            return (1 - x) * q(t) + x * 0.5 * torch.sin(2 * math.pi * t)

    assert pde._boundary_loss(Exact()).item() == pytest.approx(0.0, abs=1e-10)
    assert pde._boundary_loss(NET).item() > 0
    # Neumann with an expression value; a plain number still works
    neumann = CustomPDE(
        "u_t - u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        boundary_conditions={"neumann": {"value": "2*t"}},
        initial_condition="x",
    )

    class Slope(torch.nn.Module):
        def forward(self, inp):
            return 2 * inp[:, 1:] * inp[:, :1]

    assert neumann._boundary_loss(Slope()).item() == pytest.approx(0.0, abs=1e-10)
    with pytest.raises(ValueError, match="neither parameters"):
        CustomPDE(
            "u_t - u_xx",
            domain={"x": (0, 1), "t": (0, 1)},
            boundary_conditions={"dirichlet": {"left": "zeta"}},
        )._boundary_loss(NET)


def test_yaml_spec_resolves_relative_input_paths(tmp_path):
    t = np.linspace(0, 1, 5)
    (tmp_path / "data").mkdir()
    _write_csv(tmp_path / "data" / "left.csv", t, t)
    spec = {
        "equations": "u_t - alpha*u_xx",
        "domain": {"x": [0, 1], "t": [0, 1]},
        "parameters": {"alpha": 0.1},
        "inputs": {"q": {"path": "data/left.csv", "value_column": "T_left"}},
        "boundary_conditions": {"dirichlet": {"left": "q", "right": 0.0}},
        "initial_condition": "0",
    }
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    pde = CustomPDE.from_yaml(tmp_path / "spec.yaml")
    assert pde.inputs["q"].span == (0.0, 1.0)
    rebuilt = CustomPDE.from_dict(yaml.safe_load(yaml.safe_dump(pde.to_dict())))
    assert rebuilt.inputs["q"].path.endswith("left.csv")
    assert rebuilt.boundary_conditions == {"dirichlet": {"left": "q", "right": 0.0}}


def test_heat_boundary_driven_by_a_csv_trains(tmp_path):
    """The roadmap criterion: the heat equation's boundary follows a measured signal."""
    torch.manual_seed(0)
    t = np.linspace(0, 1, 21)
    _write_csv(tmp_path / "left.csv", t, 0.5 + 0.5 * np.sin(2 * np.pi * t))
    pde = CustomPDE(
        "u_t - 0.1*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        inputs={"q": {"path": str(tmp_path / "left.csv"), "value_column": "T_left"}},
        boundary_conditions={"dirichlet": {"left": "q", "right": 0.0}},
        initial_condition="0.5*(1 - x)",
    )
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [32, 32], "layer_norm": False},
        num_collocation_points=256,
        batch_size=256,
        num_boundary_points=128,
        num_initial_points=128,
        learning_rate=3e-3,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    before = pde._boundary_loss(trainer.model).item()
    trainer.train_steps(num_steps=800, resample_every=50)
    after = pde._boundary_loss(trainer.model).item()
    assert after < before / 10
    # the trained left boundary follows the signal (a 2 x 32 net after 800 steps: coarse)
    tt = torch.linspace(0, 1, 50).reshape(-1, 1)
    with torch.no_grad():
        u_left = trainer.model(torch.cat([torch.zeros_like(tt), tt], dim=1))
    assert torch.mean((u_left - pde.inputs["q"](tt)) ** 2).sqrt().item() < 0.2
