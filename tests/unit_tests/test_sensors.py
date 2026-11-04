"""Sensor operators: point, boundary and integral records, delay, the data loss, the layout
Jacobian against closed forms, specs, and training from sensor records (roadmap v0.6 item 4)."""

import math

import numpy as np
import pytest
import torch
import yaml

from pinnrl import CustomPDE
from pinnrl.analysis import fisher_information
from pinnrl.pdes import BoundarySensor, IntegralSensor, PointSensor, sensor_jacobian
from pinnrl.pdes.catalog import from_catalog
from pinnrl.pdes.sensors import sensor_noise_vector
from pinnrl.training.train import build_config_dict, create_pde

torch.manual_seed(0)
ALPHA = 0.1
TIMES = np.linspace(0.05, 1.0, 20)
NET = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))


def _heat(**kwargs):
    return from_catalog(
        "heat",
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": 0.5},
        sensors=[
            PointSensor(0.3, TIMES, noise=0.0, name="probe"),
            BoundarySensor("left", TIMES, "-alpha*u_x", noise=0.0, delay=0.05, name="flux"),
            IntegralSensor(TIMES, "u", noise=0.0, name="mean", normalize=True),
        ],
        **kwargs,
    )


def _decay(t):
    return np.exp(-ALPHA * math.pi**2 * t)


def test_records_come_from_the_true_solution_through_each_operator():
    pde = _heat()
    pde.observe(seed=0)
    probe, flux, mean = (s.values.ravel().numpy() for s in pde.sensors)
    assert np.allclose(probe, _decay(TIMES) * math.sin(math.pi * 0.3), atol=1e-5)
    # the flux record stamped t reflects the state at t - delay
    assert np.allclose(flux, -ALPHA * math.pi * _decay(TIMES - 0.05), atol=1e-5)
    assert np.allclose(mean, 2 / math.pi * _decay(TIMES), atol=2e-4)
    assert pde.get_trainable_parameter_values() == {"alpha": 0.5}  # the guess is untouched
    assert all(s.n_records == 20 for s in pde.sensors)


def test_noise_is_per_sensor_and_seeded():
    pde = _heat()
    for s, sigma in zip(pde.sensors, (0.1, 0.0, 0.01), strict=True):
        s.noise = sigma
    pde.observe(seed=3)
    first = [s.values.clone() for s in pde.sensors]
    clean = _decay(TIMES) * math.sin(math.pi * 0.3)
    assert 0.02 < np.std(first[0].ravel().numpy() - clean) < 0.3
    assert np.allclose(first[1].ravel().numpy(), -ALPHA * math.pi * _decay(TIMES - 0.05), atol=1e-5)
    pde.observe(seed=3)
    assert all(torch.equal(a, s.values) for a, s in zip(first, pde.sensors, strict=True))
    pde.observe(seed=4)
    assert not torch.equal(first[0], pde.sensors[0].values)
    pde.observe(noise=0.0)  # an override silences every sensor
    assert np.allclose(pde.sensors[0].values.ravel().numpy(), clean, atol=1e-5)


def test_data_loss_sums_weighted_sensor_misfits():
    pde = _heat()
    pde.observe(seed=0)
    pde.sensors[2].weight = 3.0
    x, t = torch.rand(10, 1), torch.rand(10, 1)
    total = pde.compute_loss(NET, x, t)["data"]
    expected = sum(s.weight * torch.mean((s.predict(NET) - s.values) ** 2) for s in pde.sensors)
    assert total.item() == pytest.approx(expected.item(), rel=1e-5)
    with pytest.raises(ValueError, match="no records"):
        _heat().compute_loss(NET, x, t)


def test_layout_jacobian_matches_closed_form_sensitivities():
    pde = _heat()
    J = sensor_jacobian(pde, ["alpha"])
    assert J.shape == (60, 1)
    s = TIMES - 0.05
    expected = np.concatenate(
        [
            -(math.pi**2) * TIMES * _decay(TIMES) * math.sin(math.pi * 0.3),
            -math.pi * _decay(s) * (1 - ALPHA * math.pi**2 * s),
            -(math.pi**2) * TIMES * 2 / math.pi * _decay(TIMES),
        ]
    )
    assert np.allclose(J[:, 0].numpy(), expected, atol=3e-4)
    for sensor, sigma in zip(pde.sensors, (0.01, 0.02, 0.01), strict=True):
        sensor.noise = sigma
    summary = fisher_information(J, sensor_noise_vector(pde.sensors), ["alpha"])
    manual = float(np.sum(expected**2 / np.repeat([0.01, 0.02, 0.01], 20) ** 2))
    assert summary.information[0, 0].item() == pytest.approx(manual, rel=1e-3)
    # a subset of sensors
    J_probe = sensor_jacobian(pde, ["alpha"], [pde.sensors[0]])
    assert J_probe.shape == (20, 1)
    with pytest.raises(ValueError, match="not parameters"):
        sensor_jacobian(pde, ["beta"])


def test_two_parameter_layout_conditioning_with_kdv():
    pde = from_catalog(
        "kdv_two_parameter",
        sensors=[PointSensor(x0, np.linspace(0.1, 1.0, 10), noise=0.01) for x0 in (-2.0, 0.0, 2.0)],
    )
    params = list(pde.parameters)[:2]
    J = sensor_jacobian(pde, params)
    assert J.shape == (30, 2) and torch.isfinite(J).all()
    summary = fisher_information(J, sensor_noise_vector(pde.sensors), params)
    assert summary.condition_number >= 1.0


def test_integral_sensor_with_weight_and_boundary_in_two_dimensions():
    pde = from_catalog(
        "heat_2d",
        sensors=[
            IntegralSensor(TIMES[:3], "u", weight_expression="x", quadrature=24, name="moment"),
            BoundarySensor("right", TIMES[:3], "u_x", axis=0, position=[1.0, 0.25], name="edge"),
            PointSensor([0.5, 0.5], TIMES[:3], name="centre"),
        ],
    )
    pde.observe(seed=0)
    alpha = pde.parameters["alpha"]
    decay = np.exp(-alpha * 2 * math.pi**2 * TIMES[:3])
    # ∫∫ x sin(pi x) sin(pi y) dx dy = (1/pi) * (2/pi)
    assert np.allclose(pde.sensors[0].values.ravel().numpy(), decay * 2 / math.pi**2, rtol=2e-3)
    # u_x at x = 1, y = 0.25: pi cos(pi) sin(pi/4)
    assert np.allclose(
        pde.sensors[1].values.ravel().numpy(),
        decay * math.pi * -1 * math.sin(math.pi / 4),
        atol=1e-4,
    )
    assert np.allclose(pde.sensors[2].values.ravel().numpy(), decay, atol=1e-5)
    x, t = pde.sensors[1].points()
    assert torch.allclose(x[0], torch.tensor([1.0, 0.25]))
    with pytest.raises(ValueError, match="left"):
        BoundarySensor("top", TIMES)


def test_spec_round_trip_keeps_records_and_the_cli_path_observes():
    pde = _heat()
    pde.observe(seed=0)
    spec = yaml.safe_load(yaml.safe_dump(pde.to_dict()))
    assert [s["kind"] for s in spec["sensors"]] == ["point", "boundary", "integral"]
    rebuilt = CustomPDE.from_dict(spec)
    for a, b in zip(pde.sensors, rebuilt.sensors, strict=True):
        assert type(a) is type(b) and torch.allclose(a.values, b.values)
        assert a.delay == b.delay and a.noise == b.noise
    # without records, create_pde records them through the operators
    bare = yaml.safe_load(yaml.safe_dump(_heat().to_dict()))
    from pinnrl.config import DEFAULT_CONFIG_PATH

    with open(DEFAULT_CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    cfg = build_config_dict(base, None, "feedforward", custom=bare)
    cfg["training"]["mode"] = "inverse"
    cfg["pde"]["trainable_parameters"] = ["alpha"]
    cfg["pde"]["parameter_initial_guesses"] = {"alpha": 0.5}
    built = create_pde(cfg, torch.device("cpu"))
    assert all(s.values is not None for s in built.sensors)
    assert built.observation_data is None  # sensors replace the scattered observations


def test_alpha_is_recovered_from_sensor_records():
    torch.manual_seed(0)
    pde = from_catalog(
        "heat",
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": 0.3},
        sensors=[
            PointSensor(0.25, np.linspace(0.0, 1.0, 40), noise=0.0),
            PointSensor(0.6, np.linspace(0.0, 1.0, 40), noise=0.0),
        ],
    )
    pde.observe()
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
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 50.0},
    )
    trainer.train_steps(num_steps=600, resample_every=50)
    alpha_hat = pde.get_trainable_parameter_values()["alpha"]
    assert abs(alpha_hat - ALPHA) < 0.1  # moved most of the way from 0.3 toward 0.1
