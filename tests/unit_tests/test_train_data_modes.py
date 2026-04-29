"""``compute_loss`` should respect ``data_only`` and ``data_augmented`` modes."""

import torch

from pinnrl.config import (
    AdaptiveWeightsConfig,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    TrainingConfig,
)
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEConfig

DEVICE = torch.device("cpu")


def _training_cfg(mode):
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
        loss_weights={
            "residual": 1.0,
            "boundary": 10.0,
            "initial": 10.0,
            "data": 1.0,
        },
        mode=mode,
    )


def _build_pde_with_observations(mode):
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
        training=_training_cfg(mode),
    )
    pde = HeatEquation(config=pde_config)
    pde.generate_synthetic_observations(n_points=64, noise_std=0.01, seed=0)
    return pde


class _DummyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 1),
        )

    def forward(self, xt):
        return self.net(xt)


def _losses_for_mode(mode):
    pde = _build_pde_with_observations(mode)
    model = _DummyMLP().to(DEVICE)
    x = torch.linspace(0.05, 0.95, 8, device=DEVICE).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0.05, 0.95, 8, device=DEVICE).reshape(-1, 1).requires_grad_(True)
    return pde.compute_loss(model, x, t)


def test_data_only_total_excludes_physics_terms():
    losses = _losses_for_mode("data_only")
    # Components are still computed for visualization.
    assert losses["residual"].item() >= 0.0
    assert losses["boundary"].item() >= 0.0
    assert losses["initial"].item() >= 0.0
    assert losses["data"].item() > 0.0
    # The total in data_only mode is just the data term (smoothness is 0).
    expected = losses["data"]
    assert torch.allclose(losses["total"], expected, atol=1e-6)


def test_data_augmented_total_includes_data_and_physics():
    losses = _losses_for_mode("data_augmented")
    assert losses["data"].item() > 0.0
    # The physics terms must contribute (residual + IC/BC > 0 once we have a
    # randomly initialised network on a non-trivial domain).
    physics = (
        losses["residual"].item()
        + losses["boundary"].item()
        + losses["initial"].item()
    )
    assert physics > 0.0
    # Total should strictly exceed the data-only assembly.
    assert losses["total"].item() > losses["data"].item()


def test_forward_total_unchanged_when_no_observations():
    """Backward-compat: forward mode without observations behaves as before."""
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
        training=_training_cfg("forward"),
    )
    pde = HeatEquation(config=pde_config)
    model = _DummyMLP().to(DEVICE)
    x = torch.linspace(0.05, 0.95, 8, device=DEVICE).reshape(-1, 1).requires_grad_(True)
    t = torch.linspace(0.05, 0.95, 8, device=DEVICE).reshape(-1, 1).requires_grad_(True)
    losses = pde.compute_loss(model, x, t)
    # No observations attached → data term is zero and total matches the
    # weighted physics sum.
    assert losses["data"].item() == 0.0
    expected = (
        1.0 * losses["residual"]
        + 10.0 * losses["boundary"]
        + 10.0 * losses["initial"]
    )
    assert torch.allclose(losses["total"], expected, atol=1e-6)
