"""Tests for Residual-Adaptive Refinement (RAR) sampling.

Verifies that:
1. RAR concentrates samples in high-residual regions (vs uniform).
2. RAR falls back to uniform when no model is provided.
3. The trainer's RAR path runs end-to-end without errors.
"""

import torch

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


def _make_heat_pde(training_cfg=None):
    pde_config = PDEConfig(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.05},
        boundary_conditions={"type": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=DEVICE,
        training=training_cfg,
    )
    return HeatEquation(config=pde_config)


def _make_model():
    config = Config.__new__(Config)
    config.device = DEVICE
    config.model = ModelConfig(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        num_layers=3,
        activation="tanh",
        architecture="feedforward",
    )
    config.training = TrainingConfig(
        num_epochs=1,
        batch_size=32,
        num_collocation_points=32,
        num_boundary_points=16,
        num_initial_points=16,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=False, patience=999, min_delta=1e-7),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3
        ),
        collocation_distribution="uniform",
        adaptive_weights=AdaptiveWeightsConfig(enabled=False),
        loss_weights={"residual": 1.0, "boundary": 1.0, "initial": 1.0, "smoothness": 0.0},
    )
    return PINNModel(config=config, device=DEVICE), config


def test_rar_falls_back_to_uniform_when_no_model():
    """model=None must return uniform samples without raising."""
    pde = _make_heat_pde()
    x, t = pde._sample_residual_based(100, model=None)
    assert x.shape == (100, 1)
    assert t.shape == (100, 1)


def test_rar_concentrates_on_high_residual_regions():
    """RAR samples must have higher mean |residual| than uniform samples."""
    torch.manual_seed(0)
    pde = _make_heat_pde()
    model, _ = _make_model()

    n = 2000
    x_rar, t_rar = pde.generate_collocation_points(n, strategy="residual_based", model=model)
    x_uni, t_uni = pde.generate_collocation_points(n, strategy="uniform")

    res_rar = pde.compute_residual(model, x_rar.to(DEVICE), t_rar.to(DEVICE))
    res_uni = pde.compute_residual(model, x_uni.to(DEVICE), t_uni.to(DEVICE))

    mean_rar = torch.abs(res_rar.detach()).mean().item()
    mean_uni = torch.abs(res_uni.detach()).mean().item()

    assert mean_rar > mean_uni, (
        f"RAR did not concentrate on high-residual regions: "
        f"mean|res| RAR={mean_rar:.4e}, uniform={mean_uni:.4e}"
    )


def test_trainer_rar_runs_end_to_end():
    """Trainer with collocation_distribution='residual_based' must run without errors."""
    torch.manual_seed(0)
    training_cfg = TrainingConfig(
        num_epochs=2,
        batch_size=32,
        num_collocation_points=32,
        num_boundary_points=16,
        num_initial_points=16,
        learning_rate=5e-3,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=False, patience=999, min_delta=1e-7),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3
        ),
        collocation_distribution="residual_based",
        adaptive_weights=AdaptiveWeightsConfig(enabled=False),
        loss_weights={"residual": 1.0, "boundary": 1.0, "initial": 1.0, "smoothness": 0.0},
    )
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
        optimizer_config={"learning_rate": 5e-3, "weight_decay": 0.0},
        config=config,
        device=DEVICE,
        validation_frequency=5,
    )
    trainer.train(num_epochs=2, batch_size=32, num_points=32)
    assert len(trainer.history["train_loss"]) == 2
