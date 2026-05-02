"""Sampling-strategy benchmark: RL vs uniform / stratified / residual_based.

Trains a small PINN on a chosen 1-D PDE under each requested collocation
sampling strategy with the same model, optimizer, and seed; reports the
final L2 error against the analytical solution and wall-clock time.

The harness is deliberately lightweight — it skips the dashboard plumbing,
experiment dirs, and adaptive weighting — so a comparison run finishes in
seconds per strategy on CPU. The same trainer machinery used in production
:class:`pinnrl.training.trainer.PDETrainer` is not reused: that one is
built around per-experiment side effects we want to avoid here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
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
from pinnrl.pdes.wave_equation import WaveEquation
from pinnrl.rl.rl_agent import RLAgent

# Strategies the benchmark accepts. ``adaptive`` triggers an RL agent —
# everything else is a pure sampler attached to the PDE base class.
SUPPORTED_STRATEGIES = ("uniform", "stratified", "residual_based", "adaptive")


@dataclass
class SamplingResult:
    """One row of the sampling benchmark table.

    Attributes:
        strategy: Sampling strategy name.
        l2_error: Mean-squared error against the analytical solution
            evaluated on a fixed validation grid.
        max_error: Inf-norm error against the analytical solution.
        final_loss: Total training loss at the last epoch.
        wall_time_s: Wall-clock training time in seconds.
        epochs: Number of epochs actually run.
        seed: RNG seed used for this run.
    """

    strategy: str
    l2_error: float
    max_error: float
    final_loss: float
    wall_time_s: float
    epochs: int
    seed: int
    history: List[float] = field(default_factory=list)


def _build_heat_pde(device: torch.device) -> HeatEquation:
    """Heat equation on ``[0, 1] × [0, 1]`` with periodic BCs and ``α = 0.1``."""
    cfg = PDEConfig(
        name="Heat Equation",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"alpha": 0.1},
        boundary_conditions={"periodic": {}},
        initial_condition={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        exact_solution={"type": "sin_exp_decay", "amplitude": 1.0, "frequency": 1.0},
        dimension=1,
        device=device,
    )
    return HeatEquation(config=cfg)


def _build_wave_pde(device: torch.device) -> WaveEquation:
    """Wave equation on ``[0, 1] × [0, 1]`` with periodic BCs and ``c = 1``."""
    cfg = PDEConfig(
        name="Wave Equation",
        domain=[[0.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"c": 1.0},
        boundary_conditions={"periodic": {}},
        initial_condition={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
        exact_solution={"type": "sine", "amplitude": 1.0, "frequency": 2.0},
        dimension=1,
        device=device,
    )
    return WaveEquation(config=cfg)


_PDE_BUILDERS: Dict[str, Callable[[torch.device], "object"]] = {
    "heat": _build_heat_pde,
    "wave": _build_wave_pde,
}


def _build_model(device: torch.device) -> PINNModel:
    """A modest Fourier-feature MLP — adequate to discriminate strategies."""
    config = Config()
    config.device = device
    config.model = ModelConfig(
        input_dim=2,
        hidden_dim=64,
        output_dim=1,
        num_layers=3,
        activation="tanh",
        architecture="fourier",
    )
    config.model.mapping_size = 32
    config.model.scale = 4.0
    return PINNModel(config=config, device=device).to(device)


def _build_rl_agent(device: torch.device) -> RLAgent:
    """Default DQN agent. ``state_dim`` matches the (x, t) coordinate space."""
    return RLAgent(
        state_dim=2,
        action_dim=1,
        hidden_dim=64,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        memory_size=2000,
        batch_size=32,
        target_update=10,
        reward_weights=None,
        device=device,
    )


def _validation_grid(pde, n_points: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed Cartesian grid for fair L2-error comparisons across runs."""
    nx = int(np.sqrt(n_points))
    nt = nx
    x_lo, x_hi = float(pde.domain[0][0]), float(pde.domain[0][1])
    t_lo, t_hi = float(pde.time_domain[0]), float(pde.time_domain[1])
    xs = torch.linspace(x_lo, x_hi, nx, device=device)
    ts = torch.linspace(t_lo, t_hi, nt, device=device)
    xx, tt = torch.meshgrid(xs, ts, indexing="ij")
    return xx.reshape(-1, 1), tt.reshape(-1, 1)


def _evaluate(model, pde, device: torch.device, n_points: int = 2500) -> tuple[float, float]:
    """L2/max error on a fixed validation grid against the analytical solution."""
    x, t = _validation_grid(pde, n_points, device)
    model.eval()
    with torch.no_grad():
        u_pred = model(torch.cat([x, t], dim=1)).reshape(-1)
        u_exact = pde.exact_solution(x, t).reshape(-1)
        err = (u_pred - u_exact).abs()
    return float(torch.mean(err**2).item()), float(torch.max(err).item())


def _train_one(
    pde,
    strategy: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> SamplingResult:
    """Train a fresh model under one sampling strategy and return its metrics."""
    if strategy not in SUPPORTED_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Supported: {SUPPORTED_STRATEGIES}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_model(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rl_agent: Optional[RLAgent] = None
    if strategy == "adaptive":
        rl_agent = _build_rl_agent(device)
        pde.rl_agent = rl_agent

    history: List[float] = []
    start = time.perf_counter()
    for _ in range(epochs):
        sampling_kwargs: Dict[str, object] = {}
        if strategy == "residual_based":
            sampling_kwargs["model"] = model
        x, t = pde.generate_collocation_points(batch_size, strategy=strategy, **sampling_kwargs)
        x = x.to(device)
        t = t.to(device)

        optimizer.zero_grad()
        losses = pde.compute_loss(model, x, t)
        total = losses["total"]
        total.backward()
        optimizer.step()
        history.append(float(total.item()))
    wall = time.perf_counter() - start

    l2, max_err = _evaluate(model, pde, device)
    return SamplingResult(
        strategy=strategy,
        l2_error=l2,
        max_error=max_err,
        final_loss=history[-1] if history else float("nan"),
        wall_time_s=wall,
        epochs=epochs,
        seed=seed,
        history=history,
    )


def run_sampling_benchmark(
    pde_name: str,
    strategies: Sequence[str] = SUPPORTED_STRATEGIES,
    epochs: int = 200,
    batch_size: int = 256,
    learning_rate: float = 5e-3,
    seed: int = 0,
    device: Optional[torch.device] = None,
) -> List[SamplingResult]:
    """Train a fresh PINN under each strategy and return one result per strategy.

    Args:
        pde_name: Either ``"heat"`` or ``"wave"`` — the only PDEs the
            benchmark currently scaffolds. Adding more is just a matter of
            registering a builder in ``_PDE_BUILDERS``.
        strategies: Subset of :data:`SUPPORTED_STRATEGIES` to compare.
        epochs: Training epochs per strategy. Each strategy runs from a
            fresh model with the same ``seed`` so the comparison is
            single-trajectory rather than averaged.
        batch_size: Collocation points per epoch.
        learning_rate: Adam learning rate.
        seed: Master RNG seed.
        device: Torch device. Defaults to CPU since the harness is sized
            for that.

    Returns:
        List of :class:`SamplingResult`, one per requested strategy, in the
        order they were requested.

    Raises:
        ValueError: If ``pde_name`` or any strategy is unknown.
    """
    if pde_name not in _PDE_BUILDERS:
        raise ValueError(f"Unknown pde_name {pde_name!r}. Supported: {tuple(_PDE_BUILDERS.keys())}")

    device = device or torch.device("cpu")
    results: List[SamplingResult] = []
    for strategy in strategies:
        # Rebuild the PDE per strategy so the RL agent attached for the
        # ``adaptive`` run does not bleed into the next one.
        pde = _PDE_BUILDERS[pde_name](device)
        # Trainer-style training config so PDEBase.compute_loss knows the
        # mode, optimizer hyper-parameters, and loss weights.
        pde.config.training = TrainingConfig(
            num_epochs=epochs,
            batch_size=batch_size,
            num_collocation_points=batch_size,
            num_boundary_points=max(32, batch_size // 4),
            num_initial_points=max(32, batch_size // 4),
            learning_rate=learning_rate,
            weight_decay=0.0,
            gradient_clipping=1.0,
            early_stopping=EarlyStoppingConfig(enabled=False, patience=100, min_delta=1e-4),
            learning_rate_scheduler=LearningRateSchedulerConfig(
                type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=50
            ),
            adaptive_weights=AdaptiveWeightsConfig(enabled=False),
            loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 1.0},
        )
        results.append(
            _train_one(
                pde=pde,
                strategy=strategy,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                seed=seed,
                device=device,
            )
        )
    return results
