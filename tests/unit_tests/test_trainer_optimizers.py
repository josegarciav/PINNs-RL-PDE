"""Tests for the optimizer branching in PDETrainer (Adam / L-BFGS / two-phase)."""

import torch
import torch.optim as optim

from pinnrl.config import (
    AdaptiveWeightsConfig,
    Config,
    EarlyStoppingConfig,
    LBFGSConfig,
    LearningRateSchedulerConfig,
    ModelConfig,
    TrainingConfig,
)
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEConfig
from pinnrl.training.trainer import PDETrainer

DEVICE = torch.device("cpu")


def _make_training_config(optimizer="adam", num_epochs=3, switch_ratio=0.5):
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
        optimizer=optimizer,
        adam_lbfgs_switch_ratio=switch_ratio,
        lbfgs=LBFGSConfig(history_size=10, max_iter=5),
    )


def _make_config(training_cfg):
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
    return config


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


def _build_trainer(optimizer, num_epochs=3, switch_ratio=0.5):
    training_cfg = _make_training_config(
        optimizer=optimizer, num_epochs=num_epochs, switch_ratio=switch_ratio
    )
    config = _make_config(training_cfg)
    pde = _make_heat_pde(training_cfg)
    model = PINNModel(config=config, device=DEVICE)
    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config={"learning_rate": 1e-2, "weight_decay": 0.0},
        config=config,
        device=DEVICE,
        validation_frequency=1,
    )
    return trainer, training_cfg


def test_lbfgs_optimizer_constructed():
    trainer, _ = _build_trainer("lbfgs")
    assert isinstance(trainer.optimizer, optim.LBFGS)
    assert trainer._is_lbfgs is True
    assert isinstance(trainer.scheduler, optim.lr_scheduler.ReduceLROnPlateau)


def test_lbfgs_runs_and_decreases_loss():
    """A few L-BFGS epochs on Heat should reduce total training loss end-to-end."""
    trainer, _ = _build_trainer("lbfgs", num_epochs=3)
    trainer.train(num_epochs=3, batch_size=32, num_points=32)
    losses = trainer.history["train_loss"]
    assert len(losses) >= 1
    # L-BFGS is full-batch; even 3 outer iterations should bend the curve.
    assert losses[-1] <= losses[0]


def test_adam_lbfgs_two_phase_switch():
    """In adam_lbfgs mode the optimizer should hot-swap to LBFGS at the switch epoch."""
    trainer, training_cfg = _build_trainer(
        "adam_lbfgs", num_epochs=4, switch_ratio=0.5
    )
    assert isinstance(trainer.optimizer, optim.Adam)
    assert trainer._is_lbfgs is False
    assert trainer._switch_epoch == max(1, int(4 * 0.5))

    trainer.train(num_epochs=4, batch_size=32, num_points=32)
    # Trainer should have switched to LBFGS at some point during training.
    assert isinstance(trainer.optimizer, optim.LBFGS)
    assert trainer._is_lbfgs is True


def test_adam_optimizer_unchanged_for_default():
    """Forward-compatibility: optimizer="adam" preserves the existing path."""
    trainer, _ = _build_trainer("adam", num_epochs=2)
    assert isinstance(trainer.optimizer, optim.Adam)
    assert trainer._is_lbfgs is False
    trainer.train(num_epochs=2, batch_size=32, num_points=32)
    assert len(trainer.history["train_loss"]) == 2
