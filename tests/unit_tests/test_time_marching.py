"""Time-marching windows (H04): the time cutoff on sampling and data terms, and ``train_windows``."""

import numpy as np
import torch

from pinnrl import CustomPDE
from pinnrl.pdes import PointSensor
from pinnrl.pdes.catalog import from_catalog


def test_cutoff_restricts_sampled_times_for_both_samplers():
    pde = from_catalog("logistic")
    pde.time_cutoff = 2.5
    for strategy in ("uniform", "stratified"):
        _x, t = pde.generate_collocation_points(256, strategy=strategy)
        assert float(t.max()) <= 2.5 + 1e-6 and float(t.min()) >= 0.0
    pde.time_cutoff = None
    _x, t = pde.generate_collocation_points(256)
    assert float(t.max()) > 5.0


def test_cutoff_masks_observations_and_sensor_records():
    pde = CustomPDE(
        "u_t + u",
        domain={"t": (0.0, 4.0)},
        initial_condition="1",
        exact_solution="exp(-t)",
        sensors=[PointSensor(0.5, [0.5, 1.5, 2.5, 3.5], name="probe")],
    )
    pde.observe(seed=0)
    pde.generate_synthetic_observations(n_points=40, noise_std=0.0, seed=0)
    trainer = pde.build_trainer("feedforward", model={"hidden_dims": [8, 8]}, num_epochs=1)
    full = float(pde._compute_data_loss(trainer.model))
    pde.time_cutoff = 2.0
    cut = float(pde._compute_data_loss(trainer.model))
    # the masked sensor term equals the mean over the two records inside the window
    sensor = pde.sensors[0]
    pred = sensor.predict(trainer.model).detach().reshape(-1)
    expected_sensor = float(((pred[:2] - sensor.values.reshape(-1)[:2]) ** 2).mean())
    obs = pde.observation_data
    keep = obs["t"].reshape(-1) <= 2.0
    with torch.no_grad():
        p = trainer.model(torch.cat([obs["x"], obs["t"]], dim=1))[:, :1]
    expected_obs = float(((p[keep] - obs["u"][keep]) ** 2).mean())
    assert np.isclose(cut, expected_sensor + expected_obs, rtol=1e-5)
    assert cut != full
    pde.time_cutoff = 10.0  # beyond the domain: everything counts
    assert np.isclose(float(pde._compute_data_loss(trainer.model)), full, rtol=1e-5)
    pde.time_cutoff = 0.1  # before every sensor record: the sensor term is zero
    assert float(pde.sensors[0].loss(trainer.model, pde._apply_loss_fn, time_cutoff=0.1)) == 0.0


def test_train_windows_marches_and_resets_the_cutoff():
    pde = from_catalog("logistic")
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [16, 16]},
        normalize=True,
        num_collocation_points=128,
        batch_size=128,
        num_initial_points=16,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    seen = []
    original = pde.generate_collocation_points

    def spy(n, **kwargs):
        x, t = original(n, **kwargs)
        seen.append(float(t.max()))
        return x, t

    pde.generate_collocation_points = spy
    result = trainer.train_windows([2.5, 5.0, 10.0], steps_per_window=20, resample_every=10)
    assert result["steps"] == 60 and len(result["loss"]) == 60 and len(result["windows"]) == 3
    assert max(seen[:2]) <= 2.5 + 1e-6 and max(seen[2:4]) <= 5.0 + 1e-6 and max(seen[4:]) > 5.0
    assert pde.time_cutoff is None
    assert all(np.isfinite(result["loss"]))
