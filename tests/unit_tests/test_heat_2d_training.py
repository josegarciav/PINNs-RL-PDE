"""Smoke test for 2D heat equation end-to-end training.

Builds a HeatEquation with dimension=2 over [0, 1]^2 with a sine_2d
analytical solution, trains for 3 epochs, and asserts that loss stays
finite and does not blow up. Convergence is not asserted — only that
the 2D path is plumbed correctly through compute_loss / sampling.
"""

import math

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


def _build_2d_setup(num_epochs=3):
    training_cfg = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=64,
        num_collocation_points=64,
        num_boundary_points=32,
        num_initial_points=32,
        learning_rate=5e-3,
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
    pde_config = PDEConfig(
        name="heat_2d",
        domain=[[0.0, 1.0], [0.0, 1.0]],
        time_domain=[0.0, 0.5],
        parameters={"alpha": 0.05},
        boundary_conditions={"dirichlet": {"type": "fixed", "value": 0.0}},
        initial_condition={
            "type": "sine_2d",
            "amplitude": 1.0,
            "frequency_x": 1.0,
            "frequency_y": 1.0,
        },
        exact_solution={
            "type": "sine_2d",
            "amplitude": 1.0,
            "frequency_x": 1.0,
            "frequency_y": 1.0,
        },
        dimension=2,
        device=DEVICE,
        training=training_cfg,
    )
    pde = HeatEquation(config=pde_config)

    config = Config.__new__(Config)
    config.device = DEVICE
    config.model = ModelConfig(
        input_dim=3,
        hidden_dim=16,
        output_dim=1,
        num_layers=3,
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
    return trainer, pde, model


def test_2d_compute_loss_returns_finite_components():
    """compute_loss on a 2D heat PDE must return finite residual/boundary/initial scalars."""
    trainer, pde, model = _build_2d_setup(num_epochs=1)
    x_batch, t_batch = pde.generate_collocation_points(32)
    losses = pde.compute_loss(model, x_batch.to(DEVICE), t_batch.to(DEVICE))
    for key in ("residual", "boundary", "initial", "total"):
        assert key in losses
        val = losses[key].item()
        assert math.isfinite(val), f"{key} loss is not finite: {val}"


def test_2d_collocation_points_have_correct_shape():
    """generate_collocation_points must return x of shape [N, 2] and t of shape [N, 1]."""
    _, pde, _ = _build_2d_setup(num_epochs=1)
    x, t = pde.generate_collocation_points(50)
    assert x.shape == (50, 2)
    assert t.shape == (50, 1)


def test_2d_training_runs_without_blowup():
    """3 epochs of 2D training must not produce NaN/Inf and not increase loss by >50x."""
    torch.manual_seed(0)
    trainer, _, _ = _build_2d_setup(num_epochs=3)
    trainer.train(num_epochs=3, batch_size=64, num_points=64)
    losses = trainer.history["train_loss"]
    assert len(losses) == 3
    for loss_val in losses:
        assert math.isfinite(loss_val), f"non-finite loss: {loss_val}"
    assert (
        losses[-1] <= 50.0 * losses[0]
    ), f"2D training blew up: initial {losses[0]:.4f}, final {losses[-1]:.4f}"
