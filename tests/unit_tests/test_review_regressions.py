"""Regression tests for the issues found in the 0.4.0 code review (one test per finding)."""

import copy

import pytest
import torch
import torch.nn as nn

from pinnrl.config import TrainingConfig
from pinnrl.pdes.convection_equation import ConvectionEquation
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.pdes.pde_base import PDEConfig
from pinnrl.rl.bandit_sampler import ResidualBanditSampler
from tests.unit_tests.test_trainer import _make_config, _make_pde, _make_trainer


def test_lrw_does_not_leak_component_gradients():
    """LRW must not leave per-component gradients behind for the optimizer step."""
    torch.manual_seed(0)
    trainer = _make_trainer(
        config=_make_config(adaptive_weights_enabled=True, adaptive_strategy="lrw")
    )
    x, t = trainer.pde.generate_collocation_points(32)

    reference = copy.deepcopy(trainer.model)
    applied = {}
    original_step = trainer.optimizer.step

    def capture_step(*args, **kwargs):
        applied.update({n: p.grad.clone() for n, p in trainer.model.named_parameters()})
        return original_step(*args, **kwargs)

    trainer.optimizer.step = capture_step
    trainer.config.training.gradient_clipping = 0.0
    losses = trainer._step(x, t)

    # Recompute the intended gradient: d(total)/d(theta) with the same weights.
    weights = trainer.adaptive_weights.weights
    ref_losses = trainer.pde.compute_loss(reference, x, t)
    total = trainer.pde.weighted_total(ref_losses, *weights[:3])
    expected = torch.autograd.grad(total, list(reference.parameters()))
    for (name, _), g in zip(reference.named_parameters(), expected):
        torch.testing.assert_close(applied[name], g, rtol=1e-4, atol=1e-6)
    assert torch.isfinite(losses["total"])


def test_lrw_tolerates_components_without_graph():
    """A loss component with no autograd graph counts as a zero gradient norm."""
    trainer = _make_trainer(
        config=_make_config(adaptive_weights_enabled=True, adaptive_strategy="lrw")
    )
    x, t = trainer.pde.generate_collocation_points(16)
    losses = trainer.pde.compute_loss(trainer.model, x, t)
    losses["boundary"] = torch.tensor(0.0)
    trainer._adaptive_total(losses)
    assert torch.isfinite(losses["total"])


def test_exact_solution_is_measured_from_t0():
    """With time_domain = [1, 3] the exact solution still equals the IC at t = t0."""
    pde = _make_pde()
    shifted = HeatEquation(config=PDEConfig(**{**pde.config.__dict__, "time_domain": [1.0, 3.0]}))
    x = torch.linspace(0.0, 1.0, 50).reshape(-1, 1)
    t0 = torch.full_like(x, 1.0)
    torch.testing.assert_close(shifted.exact_solution(x, t0), shifted.initial_condition(x, t0))


def test_from_dict_coerces_lbfgs_and_switch_ratio():
    """YAML-style strings like '1e-7' become floats for L-BFGS and the switch ratio."""
    cfg = TrainingConfig.from_dict(
        {
            "adam_lbfgs_switch_ratio": "7e-1",
            "lbfgs": {"tolerance_grad": "1e-7", "tolerance_change": "1e-9", "max_iter": "5"},
        }
    )
    assert cfg.adam_lbfgs_switch_ratio == pytest.approx(0.7)
    assert isinstance(cfg.lbfgs.tolerance_grad, float) and cfg.lbfgs.tolerance_grad == 1e-7
    assert isinstance(cfg.lbfgs.tolerance_change, float)
    assert cfg.lbfgs.max_iter == 5


def _convection_config(velocity):
    return PDEConfig(
        name="Convection Equation",
        domain=[[0.0, 2.0]],
        time_domain=[0.0, 1.0],
        parameters={"velocity": velocity},
        boundary_conditions={"periodic": {}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={},
        dimension=1,
        device=torch.device("cpu"),
        trainable_parameters=["velocity"],
    )


def test_inverse_mode_unwraps_one_element_parameter_lists():
    """The shipped `velocity: [1.0]` can be identified; vector parameters are rejected clearly."""
    pde = ConvectionEquation(config=_convection_config([1.0]))
    assert pde.get_trainable_parameter_values() == {"velocity": 1.0}
    with pytest.raises(ValueError, match="only scalar parameters"):
        ConvectionEquation(config=_convection_config([1.0, 2.0]))


class _ZeroModel(nn.Module):
    """A model whose output (and hence the heat residual) is identically zero."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))

    def forward(self, xt):
        return self.w * xt[:, :1]


def test_rad_falls_back_to_uniform_on_zero_residual():
    """All-zero residuals must not produce NaN probabilities."""
    pde = _make_pde()
    x, t = pde.generate_collocation_points(64, strategy="residual_based", model=_ZeroModel(), c=0.0)
    assert x.shape == (64, 1) and t.shape == (64, 1)


def test_rad_handles_a_single_point_pool():
    """num_points * pool_factor == 1 must still return one point."""
    pde = _make_pde()
    model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
    x, t = pde.generate_collocation_points(1, strategy="residual_based", model=model, pool_factor=1)
    assert x.shape == (1, 1) and t.shape == (1, 1)


class _ConstantResidualPDE:
    def compute_residual(self, model, x, t):
        return torch.ones_like(x)


@pytest.mark.parametrize("policy_fraction, n", [(0.0, 10), (0.5, 1)])
def test_bandit_with_zero_policy_points(policy_fraction, n):
    """When the policy share rounds to zero points, sampling is purely uniform."""
    sampler = ResidualBanditSampler(
        (0.0, 1.0),
        (0.0, 1.0),
        n_x_cells=2,
        n_t_cells=2,
        probe_cells=2,
        probe_points=2,
        policy_fraction=policy_fraction,
    )
    x, t = sampler.sample(_ConstantResidualPDE(), model=None, num_points=n)
    assert x.shape == (n, 1) and t.shape == (n, 1)


def test_cosine_schedule_spans_post_warmup_epochs():
    """With warmup, the cosine schedule still reaches min_lr at the last epoch."""
    config = _make_config(scheduler_type="cosine")
    config.training.num_epochs = 20
    config.training.learning_rate_scheduler.warmup_epochs = 5
    trainer = _make_trainer(config=config)
    assert trainer.scheduler.T_max == 15


def test_early_stopping_patience_counts_epochs():
    """patience is in epochs: with validation every 2 epochs and patience 4, stop by epoch ~6."""
    config = _make_config()
    config.training.validation_frequency = 2
    config.training.early_stopping.patience = 4
    config.training.early_stopping.min_delta = 1e9  # nothing ever counts as an improvement
    trainer = _make_trainer(config=config)
    trainer.validation_frequency = 2
    trainer.train(num_epochs=50, batch_size=16, num_points=32, progress=False)
    assert len(trainer.history["train_loss"]) <= 7
