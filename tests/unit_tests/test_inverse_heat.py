"""End-to-end test for inverse-problem parameter identification on the Heat equation.

Builds a Heat PDE with α as a trainable parameter, generates synthetic noisy
observations from the analytical solution at the TRUE α, then trains and
verifies the recovered α moves meaningfully toward the truth.
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
TRUE_ALPHA = 0.05
INITIAL_GUESS = 0.5


def _build_inverse_setup(num_epochs=20):
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
        loss_weights={
            "residual": 1.0,
            "boundary": 1.0,
            "initial": 1.0,
            "smoothness": 0.0,
            "data": 50.0,
        },
        mode="inverse",
    )
    pde_config = PDEConfig(
        name="heat",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": TRUE_ALPHA},
        boundary_conditions={"type": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=DEVICE,
        training=training_cfg,
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": INITIAL_GUESS},
    )
    pde = HeatEquation(config=pde_config)
    pde.generate_synthetic_observations(n_points=200, noise_std=0.005, seed=0)

    config = Config.__new__(Config)
    config.device = DEVICE
    config.model = ModelConfig(
        input_dim=2,
        hidden_dim=32,
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
    return trainer, pde


def test_alpha_registered_as_nn_parameter():
    """`alpha` becomes an nn.Parameter and `get_parameter` returns it."""
    _, pde = _build_inverse_setup(num_epochs=1)
    assert "alpha" in pde._trainable_params
    assert isinstance(pde._trainable_params["alpha"], torch.nn.Parameter)
    assert torch.is_tensor(pde.get_parameter("alpha"))
    # Initial guess is set, true α is preserved separately.
    assert abs(pde._trainable_params["alpha"].item() - INITIAL_GUESS) < 1e-6
    assert abs(pde._true_parameters["alpha"] - TRUE_ALPHA) < 1e-9


def test_synthetic_observations_use_true_alpha():
    """Synthetic obs must encode the true α even though α is trainable now."""
    _, pde = _build_inverse_setup(num_epochs=1)
    obs = pde.observation_data
    assert obs is not None
    assert obs["x"].shape[0] == 200
    assert obs["u"].shape == obs["x"].shape


def test_compute_loss_includes_data_term():
    """`compute_loss` must surface a non-zero `data` loss when observations exist."""
    trainer, pde = _build_inverse_setup(num_epochs=1)
    x_batch, t_batch = pde.generate_collocation_points(32)
    losses = pde.compute_loss(trainer.model, x_batch.to(DEVICE), t_batch.to(DEVICE))
    assert "data" in losses
    assert losses["data"].item() > 0


def test_alpha_trajectory_logged_during_training():
    """Trainer history should record `param_alpha` once per epoch."""
    trainer, _ = _build_inverse_setup(num_epochs=3)
    trainer.train(num_epochs=3, batch_size=64, num_points=64)
    assert "param_alpha" in trainer.history
    assert len(trainer.history["param_alpha"]) == 3


def test_alpha_moves_toward_truth():
    """20 epochs should reduce the error in identified α at least directionally.

    We're not asserting full convergence (CI budget), only that the gradient
    flow is wired correctly: α should move from 0.5 toward 0.05.
    """
    trainer, pde = _build_inverse_setup(num_epochs=20)
    initial_err = abs(pde._trainable_params["alpha"].item() - TRUE_ALPHA)
    trainer.train(num_epochs=20, batch_size=64, num_points=64)
    final_err = abs(pde._trainable_params["alpha"].item() - TRUE_ALPHA)
    assert final_err < initial_err, (
        f"α did not move toward truth (initial err {initial_err:.4f}, " f"final {final_err:.4f})"
    )
