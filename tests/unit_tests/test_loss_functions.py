"""Tests for configurable loss functions (MSE / MAE / Huber)."""

import math

import pytest
import torch
import torch.nn.functional as F

from pinnrl.config import (
    AdaptiveWeightsConfig,
    Config,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    ModelConfig,
    TrainingConfig,
)
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEConfig
from pinnrl.training.trainer import PDETrainer

DEVICE = torch.device("cpu")


def _make_training_config(loss_function="mse", huber_delta=1.0, num_epochs=2):
    return TrainingConfig(
        num_epochs=num_epochs,
        batch_size=32,
        num_collocation_points=32,
        num_boundary_points=16,
        num_initial_points=16,
        learning_rate=1e-2,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=False, patience=999, min_delta=1e-7),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3
        ),
        collocation_distribution="uniform",
        adaptive_weights=AdaptiveWeightsConfig(enabled=False),
        loss_weights={"residual": 1.0, "boundary": 1.0, "initial": 1.0, "smoothness": 0.0},
        loss_function=loss_function,
        huber_delta=huber_delta,
    )


def _make_heat_pde(training_cfg):
    pde_config = PDEConfig(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.01},
        boundary_conditions={"type": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=DEVICE,
        training=training_cfg,
    )
    return HeatEquation(config=pde_config)


def test_default_loss_is_mse():
    pde = _make_heat_pde(_make_training_config())
    error = torch.tensor([0.5, -1.0, 0.25, -0.75])
    expected = torch.mean(error ** 2)
    assert torch.allclose(pde._apply_loss_fn(error), expected)


def test_mae_matches_torch_mean_abs():
    pde = _make_heat_pde(_make_training_config(loss_function="mae"))
    error = torch.tensor([0.5, -1.0, 0.25, -0.75])
    expected = torch.mean(torch.abs(error))
    assert torch.allclose(pde._apply_loss_fn(error), expected)


def test_huber_matches_torch_huber():
    delta = 0.5
    pde = _make_heat_pde(_make_training_config(loss_function="huber", huber_delta=delta))
    error = torch.tensor([0.1, -2.0, 0.4, -0.6, 1.5])
    expected = F.huber_loss(error, torch.zeros_like(error), reduction="mean", delta=delta)
    assert torch.allclose(pde._apply_loss_fn(error), expected)


def test_invalid_loss_function_raises():
    with pytest.raises(ValueError):
        _make_training_config(loss_function="bogus")


@pytest.mark.parametrize("loss_function", ["mse", "mae", "huber"])
def test_training_runs_with_each_loss(loss_function):
    """End-to-end smoke: 2 epochs heat 1D under each loss must remain finite."""
    torch.manual_seed(0)
    training_cfg = _make_training_config(loss_function=loss_function)
    pde = _make_heat_pde(training_cfg)

    config = Config.__new__(Config)
    config.device = DEVICE
    config.model = ModelConfig(
        input_dim=2,
        hidden_dim=8,
        output_dim=1,
        num_layers=2,
        activation="tanh",
        architecture="feedforward",
    )
    config.training = training_cfg
    model = PINNModel(config=config, device=DEVICE)

    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config={"learning_rate": 1e-2, "weight_decay": 0.0},
        config=config,
        device=DEVICE,
        validation_frequency=5,
    )
    trainer.train(num_epochs=2, batch_size=32, num_points=32)
    losses = trainer.history["train_loss"]
    assert len(losses) == 2
    for val in losses:
        assert math.isfinite(val), f"non-finite loss for {loss_function}: {val}"
