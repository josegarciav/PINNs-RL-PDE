"""``PDEBase._load_observation_data`` should accept Well-source specs."""

import sys
import types

import numpy as np
import pytest
import torch

from pinnrl.config import (
    AdaptiveWeightsConfig,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    TrainingConfig,
)
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEConfig


def _install_fake_well(monkeypatch):
    class FakeWellDataset:
        def __init__(self, well_base_path, well_dataset_name, well_split_name):
            self._fields = np.zeros((1, 2, 3, 3, 1), dtype=np.float32)

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return {
                "input_fields": torch.from_numpy(self._fields[idx]),
                "time": np.array([0.0, 1.0], dtype=np.float32),
                "space/x": np.linspace(0.0, 1.0, 3, dtype=np.float32),
                "space/y": np.linspace(0.0, 1.0, 3, dtype=np.float32),
            }

    fake_data = types.ModuleType("the_well.data")
    fake_data.WellDataset = FakeWellDataset
    fake_root = types.ModuleType("the_well")
    fake_root.data = fake_data
    monkeypatch.setitem(sys.modules, "the_well", fake_root)
    monkeypatch.setitem(sys.modules, "the_well.data", fake_data)


def _training_cfg(mode="data_only"):
    return TrainingConfig(
        num_epochs=1,
        batch_size=8,
        num_collocation_points=8,
        num_boundary_points=4,
        num_initial_points=4,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=False, patience=999, min_delta=1e-7),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3
        ),
        adaptive_weights=AdaptiveWeightsConfig(enabled=False),
        loss_weights={"residual": 1.0, "boundary": 1.0, "initial": 1.0, "data": 1.0},
        mode=mode,
    )


def test_well_observation_branch_loads_tensors(monkeypatch, tmp_path):
    _install_fake_well(monkeypatch)
    monkeypatch.setenv("PINNRL_WELL_CACHE", str(tmp_path))

    pde_config = PDEConfig(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.05},
        boundary_conditions={"type": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=torch.device("cpu"),
        training=_training_cfg("data_only"),
        observation_data={
            "source": "well",
            "name": "rayleigh_benard",
            "split": "train",
            "n_traj": 1,
            "n_points": 4,
            "seed": 0,
        },
    )
    pde = HeatEquation(config=pde_config)
    assert pde.observation_data is not None
    assert set(pde.observation_data) == {"x", "t", "u"}
    # 4 sub-sampled points, 2 spatial dims (rayleigh_benard is 2D), 1 field
    # in the fake stub.
    assert pde.observation_data["x"].shape == (4, 2)
    assert pde.observation_data["t"].shape == (4, 1)
    assert pde.observation_data["u"].shape == (4, 1)


def test_unknown_well_dataset_propagates_keyerror(monkeypatch, tmp_path):
    _install_fake_well(monkeypatch)
    monkeypatch.setenv("PINNRL_WELL_CACHE", str(tmp_path))

    bad_config = PDEConfig(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.05},
        boundary_conditions={"type": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=torch.device("cpu"),
        training=_training_cfg("data_only"),
        observation_data={"source": "well", "name": "no_such_dataset"},
    )
    # The PDE constructor calls _load_observation_data which delegates to
    # the registry; the unknown name bubbles up as KeyError.
    with pytest.raises(KeyError):
        HeatEquation(config=bad_config)
