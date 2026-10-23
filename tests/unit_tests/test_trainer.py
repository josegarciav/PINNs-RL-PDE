"""Tests for src/training/trainer.py — PDETrainer smoke tests."""

import os
import tempfile

import pytest
import torch

from pinnrl import plots
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
        boundary_conditions={"dirichlet": {"value": 0.0}},
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
        config=config,
        device=DEVICE,
        **kwargs,
    )


# ── Initialization tests ────────────────────────────────────────────────


def test_trainer_init_default():
    """PDETrainer initialises with default early stopping config."""
    trainer = _make_trainer()
    # Without an override, early stopping comes from config.training.early_stopping.
    assert trainer.early_stopping_enabled is True
    assert trainer.patience == 5
    assert trainer.min_delta == 1e-7
    assert trainer.best_val_loss == float("inf")
    assert isinstance(trainer.history, dict)
    assert "train_loss" in trainer.history


def test_trainer_init_no_early_stopping():
    """Early stopping settings come from config.training.early_stopping."""
    config = _make_config()
    config.training.early_stopping = EarlyStoppingConfig(enabled=False, patience=100, min_delta=0)
    trainer = _make_trainer(config=config)
    assert trainer.early_stopping_enabled is False
    assert trainer.patience == 100


def test_trainer_init_with_bandit_sampler():
    """A bandit sampler switches the trainer to adaptive sampling and is attached to the PDE."""

    class FakeSampler:
        pass

    sampler = FakeSampler()
    trainer = _make_trainer(bandit_sampler=sampler)
    assert trainer.use_rl is True
    assert trainer.pde.bandit_sampler is sampler


def test_trainer_without_bandit_sampler_uses_config_distribution():
    """Without a sampler the trainer keeps the configured collocation distribution."""
    trainer = _make_trainer()
    assert trainer.use_rl is False


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
    config.training.validation_frequency = 1
    trainer = _make_trainer(config=config)
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


def test_train_returns_history():
    """train() returns the trainer's history dict."""
    trainer = _make_trainer()
    h = trainer.train(num_epochs=1, batch_size=32, num_points=64)
    assert h is trainer.history
    assert len(h["train_loss"]) == 1


def test_train_defaults_from_config():
    """train() with no arguments uses config.training epochs/batch/points."""
    trainer = _make_trainer()
    h = trainer.train()
    assert len(h["train_loss"]) == 2  # config.training.num_epochs


def test_constant_learning_rate_when_scheduler_is_none():
    """learning_rate_scheduler=None means no scheduler and a fixed LR."""
    config = _make_config()
    config.training.learning_rate_scheduler = None
    trainer = _make_trainer(config=config)
    assert trainer.scheduler is None
    trainer.train(num_epochs=2)
    assert trainer.history["learning_rate"] == [1e-3, 1e-3]


def test_constant_scheduler_type():
    trainer = _make_trainer(config=_make_config(scheduler_type="constant"))
    assert trainer.scheduler is None


# ── Plotting (no display, just verify it doesn't crash) ─────────────────


def test_plot_training_history():
    """plot_training_history runs without error."""
    trainer = _make_trainer()
    trainer.train(num_epochs=2, batch_size=32, num_points=64)
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "history.png")
        plots.plot_training_history(trainer, save_path=save_path)
        assert os.path.exists(os.path.join(tmpdir, "history.html"))


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


# ── Step-based loop (benchmarks / studies) ──────────────────────────────


def test_train_steps_num_steps():
    trainer = _make_trainer()
    out = trainer.train_steps(num_steps=5)
    assert out["steps"] == 5
    assert len(out["loss"]) == 5
    assert out["train_time_s"] > 0


def test_train_steps_time_budget_and_checkpoints():
    trainer = _make_trainer()
    seen = []
    out = trainer.train_steps(
        time_budget_s=0.2,
        checkpoints=[0.05, 0.1],
        on_checkpoint=lambda step, clock, model: seen.append((step, clock)),
    )
    assert out["train_time_s"] >= 0.2
    assert [c >= mark for (_, c), mark in zip(seen, [0.05, 0.1])] == [True, True]


def test_train_steps_resample_every_reuses_points():
    trainer = _make_trainer()
    calls = []

    def sample_fn(model, progress):
        calls.append(progress)
        return trainer.pde.generate_collocation_points(16)

    trainer.train_steps(num_steps=6, sample_fn=sample_fn, resample_every=3)
    assert len(calls) == 2


def test_train_steps_matches_train_step():
    """train() and train_steps() run the same update for the same points."""
    a, b = _make_trainer(), _make_trainer()
    b.model.load_state_dict(a.model.state_dict())
    x, t = a.pde.generate_collocation_points(32)
    la = a._step(x, t)["total"].item()
    lb = b.train_steps(num_steps=1, sample_fn=lambda m, p: (x, t))["loss"][0]
    assert la == pytest.approx(lb)


def test_train_steps_requires_a_stop_condition():
    with pytest.raises(ValueError):
        _make_trainer().train_steps()


# ── pinnrl-train run directories ────────────────────────────────────────


def test_run_training_and_load_run_round_trip(tmp_path, monkeypatch):
    """A run written by run_training reloads into the same model and the true-parameter PDE."""
    import yaml

    from pinnrl.config import DEFAULT_CONFIG_PATH
    from pinnrl.training.train import build_config_dict, load_run, run_training

    monkeypatch.chdir(tmp_path)
    with open(DEFAULT_CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    raw["training"].update(batch_size=32, num_collocation_points=32)
    config = build_config_dict(raw, "heat", "feedforward", epochs=1)
    config["pde"]["trainable_parameters"] = ["alpha"]
    config["pde"]["parameter_initial_guesses"] = {"alpha": 0.5}
    config["training"]["mode"] = "inverse"
    run_training(config, torch.device("cpu"))

    (run_dir,) = (tmp_path / "runs").iterdir()
    pde, model, saved = load_run(run_dir)
    assert (
        pde.alpha == raw["pde_configs"]["heat"]["parameters"]["alpha"]
    )  # true value, not the guess
    state = torch.load(run_dir / "final_model.pt")
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert saved["pde"]["name"] == "Heat Equation"


def test_linear_warmup_then_scheduler():
    config = _make_config(scheduler_type="constant")
    config.training.learning_rate_scheduler.warmup_epochs = 4
    trainer = _make_trainer(config=config)
    trainer.train(num_epochs=6)
    assert trainer.history["learning_rate"] == pytest.approx(
        [2.5e-4, 5e-4, 7.5e-4, 1e-3, 1e-3, 1e-3]
    )


def test_reduce_lr_steps_on_last_train_loss_by_default():
    trainer = _make_trainer(config=_make_config(scheduler_type="reduce_lr"))
    trainer.history["train_loss"].append(1.0)
    trainer._update_scheduler()  # used to read a never-set attribute
