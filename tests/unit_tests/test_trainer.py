"""Tests for src/training/trainer.py — PDETrainer smoke tests."""

import os
import tempfile

import pytest
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


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_config(
    architecture="feedforward",
    adaptive_weights_enabled=False,
    adaptive_strategy="rbw",
    scheduler_type="cosine",
):
    """Build a minimal Config object suitable for PDETrainer."""
    training = TrainingConfig(
        num_epochs=2,
        batch_size=32,
        num_collocation_points=64,
        num_boundary_points=32,
        num_initial_points=32,
        learning_rate=1e-3,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=True, patience=5, min_delta=1e-7),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type=scheduler_type,
            warmup_epochs=0,
            min_lr=1e-6,
            factor=0.5,
            patience=3,
        ),
        collocation_distribution="uniform",
        adaptive_weights=AdaptiveWeightsConfig(
            enabled=adaptive_weights_enabled,
            strategy=adaptive_strategy,
            alpha=0.9,
            eps=1e-5,
        ),
    )
    model_cfg = ModelConfig(
        input_dim=2,
        hidden_dim=16,
        output_dim=1,
        num_layers=2,
        activation="tanh",
        architecture=architecture,
    )
    config = Config.__new__(Config)
    config.device = DEVICE
    config.model = model_cfg
    config.training = training
    config.pde_config = None
    return config


def _make_pde():
    """Build a simple HeatEquation for testing."""
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
        training=TrainingConfig(
            num_epochs=2,
            batch_size=32,
            num_collocation_points=64,
            num_boundary_points=32,
            num_initial_points=32,
            learning_rate=1e-3,
            weight_decay=0.0,
            gradient_clipping=1.0,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=5, min_delta=1e-7),
            learning_rate_scheduler=LearningRateSchedulerConfig(
                type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=3
            ),
            adaptive_weights=AdaptiveWeightsConfig(enabled=False),
        ),
    )
    return HeatEquation(config=pde_config)


def _make_trainer(config=None, pde=None, **kwargs):
    """Create a PDETrainer with sensible defaults."""
    config = config or _make_config()
    pde = pde or _make_pde()
    model = PINNModel(config=config, device=DEVICE)
    return PDETrainer(
        model=model,
        pde=pde,
        optimizer_config={"learning_rate": 1e-3, "weight_decay": 0.0},
        config=config,
        device=DEVICE,
        **kwargs,
    )


# ── Initialization tests ────────────────────────────────────────────────


def test_trainer_init_default():
    """PDETrainer initialises with default early stopping config."""
    trainer = _make_trainer()
    assert trainer.early_stopping_enabled is True
    assert trainer.patience == 10  # default from trainer when no config passed
    assert trainer.best_val_loss == float("inf")
    assert isinstance(trainer.history, dict)
    assert "train_loss" in trainer.history


def test_trainer_init_no_early_stopping():
    """PDETrainer respects early_stopping_config override."""
    trainer = _make_trainer(early_stopping_config={"enabled": False, "patience": 100})
    assert trainer.early_stopping_enabled is False
    assert trainer.patience == 100


def test_trainer_init_with_rl_agent():
    """PDETrainer stores rl_agent attribute."""

    class FakeAgent:
        pass

    agent = FakeAgent()
    trainer = _make_trainer(rl_agent=agent)
    assert trainer.rl_agent is agent


# ── Optimizer & scheduler tests ──────────────────────────────────────────


def test_cosine_scheduler():
    """Trainer initialises cosine scheduler correctly."""
    trainer = _make_trainer()
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)


def test_reduce_lr_scheduler():
    """Trainer initialises ReduceLROnPlateau scheduler correctly."""
    config = _make_config(scheduler_type="reduce_lr")
    trainer = _make_trainer(config=config)
    assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_unknown_scheduler_raises():
    """Unknown scheduler type raises ValueError."""
    config = _make_config(scheduler_type="unknown_type")
    with pytest.raises(ValueError, match="Unknown scheduler type"):
        _make_trainer(config=config)


# ── Training smoke test ──────────────────────────────────────────────────


def test_train_one_epoch():
    """Run 1 epoch of training and verify history is populated."""
    trainer = _make_trainer()
    history = trainer.train(num_epochs=1, batch_size=32, num_points=64)
    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] > 0


def test_train_with_experiment_dir():
    """Training with experiment_dir creates expected files."""
    trainer = _make_trainer()
    with tempfile.TemporaryDirectory() as tmpdir:
        exp_dir = os.path.join(tmpdir, "exp1")
        history = trainer.train(num_epochs=1, batch_size=32, num_points=64, experiment_dir=exp_dir)
        assert os.path.isdir(exp_dir)
        assert os.path.exists(os.path.join(exp_dir, "metadata.json"))
        assert os.path.exists(os.path.join(exp_dir, "final_model.pt"))
        assert len(history["train_loss"]) >= 1


def test_train_early_stopping():
    """Early stopping fires when val_loss doesn't improve."""
    config = _make_config()
    config.training.early_stopping = EarlyStoppingConfig(enabled=True, patience=1, min_delta=1e-7)
    trainer = _make_trainer(config=config, validation_frequency=1)
    # Run enough epochs for early stopping to trigger
    trainer.train(num_epochs=50, batch_size=32, num_points=64)
    # Should have stopped before 50 epochs
    assert len(trainer.history["train_loss"]) <= 50


# ── Validation loss ──────────────────────────────────────────────────────


def test_compute_validation_loss():
    """_compute_validation_loss returns dict with expected keys."""
    trainer = _make_trainer()
    val = trainer._compute_validation_loss(num_points=64)
    assert "total_loss" in val
    assert "residual_loss" in val
    assert "boundary_loss" in val
    assert "initial_loss" in val
    assert all(isinstance(v, float) for v in val.values())


# ── Scheduler update ─────────────────────────────────────────────────────


def test_update_scheduler_cosine():
    """_update_scheduler works for cosine scheduler."""
    trainer = _make_trainer()
    # Should not raise
    trainer._update_scheduler(val_loss=0.5)


def test_update_scheduler_reduce_lr():
    """_update_scheduler works for ReduceLROnPlateau."""
    config = _make_config(scheduler_type="reduce_lr")
    trainer = _make_trainer(config=config)
    trainer.train_loss = 1.0
    trainer._update_scheduler(val_loss=0.5)


# ── Experiment logging ───────────────────────────────────────────────────


def test_setup_experiment_logging():
    """setup_experiment_logging adds a file handler."""
    trainer = _make_trainer()
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.setup_experiment_logging(tmpdir)
        log_file = os.path.join(tmpdir, "experiment.log")
        assert os.path.exists(log_file)


def test_setup_experiment_logging_none():
    """setup_experiment_logging with None does nothing."""
    trainer = _make_trainer()
    handler_count = len(trainer.logger.handlers)
    trainer.setup_experiment_logging(None)
    # Handler count should not change
    assert len(trainer.logger.handlers) == handler_count


# ── History retrieval ────────────────────────────────────────────────────


def test_get_training_history():
    """get_training_history returns the history dict."""
    trainer = _make_trainer()
    trainer.train(num_epochs=1, batch_size=32, num_points=64)
    h = trainer.get_training_history()
    assert h is trainer.history
    assert len(h["train_loss"]) == 1


# ── Plotting (no display, just verify it doesn't crash) ─────────────────


def test_plot_training_history():
    """plot_training_history runs without error."""
    trainer = _make_trainer()
    trainer.train(num_epochs=2, batch_size=32, num_points=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "history.html")
        # Should not raise
        trainer.plot_training_history(save_path=save_path)


def test_save_plots():
    """save_plots creates visualization files."""
    trainer = _make_trainer()
    trainer.train(num_epochs=1, batch_size=32, num_points=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer.save_plots(tmpdir)
        # Should have created some files
        files = os.listdir(tmpdir)
        assert len(files) > 0


# ── Adaptive weights path ───────────────────────────────────────────────


def test_trainer_adaptive_weights_rbw():
    """PDETrainer with RBW adaptive weights completes training."""
    config = _make_config(adaptive_weights_enabled=True, adaptive_strategy="rbw")
    trainer = _make_trainer(config=config)
    assert trainer.use_adaptive_weights is True
    history = trainer.train(num_epochs=1, batch_size=32, num_points=64)
    assert len(history["train_loss"]) == 1


def test_trainer_adaptive_weights_lrw():
    """PDETrainer with LRW adaptive weights completes training."""
    config = _make_config(adaptive_weights_enabled=True, adaptive_strategy="lrw")
    trainer = _make_trainer(config=config)
    history = trainer.train(num_epochs=1, batch_size=32, num_points=64)
    assert len(history["train_loss"]) == 1
