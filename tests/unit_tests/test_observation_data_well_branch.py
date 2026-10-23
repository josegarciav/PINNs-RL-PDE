"""``PDEBase._load_observation_data`` should accept Well-source specs."""

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
from tests.unit_tests.test_well_loader import _install_fake_well_module


def _install_fake_well(monkeypatch):
    """One 3-step trajectory on a 3x3 grid with a single field (real sample format)."""
    _install_fake_well_module(monkeypatch, n_traj=1, n_steps=3, hw=3, n_fields=1)


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
        boundary_conditions={"dirichlet": {"value": 0.0}},
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
        boundary_conditions={"dirichlet": {"value": 0.0}},
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
