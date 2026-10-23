"""Equal-wall-clock, multi-seed sampling study against numerical references.

Where :mod:`pinnrl.benchmarks.sampling` runs one seed for a fixed number of
epochs on PDEs with closed-form solutions, this study answers the question a
sampling method actually has to win: *given the same compute time, which
sampler gives the lowest error?*

* **PDEs** — the standard Burgers (``nu = 0.01/pi``, ``u0 = -sin(pi x)``) and
  Allen-Cahn (``u_t = 0.001 u_xx + 5(u - u³)``, ``u0 = x² cos(pi x)``)
  benchmarks, scored against :mod:`pinnrl.benchmarks.reference`.
* **Strategies** — ``uniform`` (fresh uniform random points every step),
  ``rad`` (residual-based adaptive distribution, Wu et al. 2023) and
  ``rl_bandit`` (:class:`pinnrl.rl.bandit_sampler.ResidualBanditSampler`).
* **Budget** — each run trains until its *training* wall-clock reaches
  ``time_budget_s``; evaluation time is excluded. Sampling overhead counts
  against the budget, so an expensive sampler has to earn its cost.
* **Seeds** — a seed fixes the initial network weights, so strategies are
  compared pairwise on identical initializations.
"""

from __future__ import annotations

import csv
import json
import math
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pinnrl.benchmarks.reference import (
    ReferenceSolution,
    allen_cahn_reference,
    burgers_reference,
)
from pinnrl.config import (
    AdaptiveWeightsConfig,
    Config,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    ModelConfig,
    TrainingConfig,
)
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes.allen_cahn import AllenCahnEquation
from pinnrl.pdes.burgers_equation import BurgersEquation
from pinnrl.pdes.pde_base import PDEBase, PDEConfig
from pinnrl.rl.bandit_sampler import ResidualBanditSampler

STUDY_PDES = ("burgers", "allen_cahn")
STUDY_STRATEGIES = ("uniform", "rad", "rl_bandit")


@dataclass
class StudySettings:
    """Shared knobs for every run in a study.

    Attributes:
        time_budget_s: Training wall-clock per run.
        n_collocation: Collocation points per training batch.
        resample_every: Steps between resampling for ``rad`` and
            ``rl_bandit`` (``uniform`` draws fresh points every step).
        learning_rate: Adam learning rate.
        hidden_dims: Widths of the tanh MLP.
        n_checkpoints: Evaluations along the budget for error-vs-time curves.
        rad_k: RAD exponent ``k``.
        rad_c: RAD uniform floor ``c``.
        rad_pool_factor: RAD candidate pool size as a multiple of ``n_collocation``.
    """

    time_budget_s: float = 60.0
    n_collocation: int = 1000
    resample_every: int = 100
    learning_rate: float = 1e-3
    hidden_dims: Tuple[int, ...] = (64, 64, 64, 64)
    n_checkpoints: int = 10
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_pool_factor: int = 20


@dataclass
class StudyRun:
    """Outcome of one ``(pde, strategy, seed)`` run.

    Attributes:
        pde: PDE key.
        strategy: Sampling strategy.
        seed: Seed for weights and sampling.
        rel_l2: Final relative L2 error against the reference.
        steps: Optimizer steps completed within the budget.
        train_time_s: Training wall-clock actually used.
        curve_time_s: Training time at each checkpoint.
        curve_rel_l2: Relative L2 error at each checkpoint.
    """

    pde: str
    strategy: str
    seed: int
    rel_l2: float
    steps: int
    train_time_s: float
    curve_time_s: List[float] = field(default_factory=list)
    curve_rel_l2: List[float] = field(default_factory=list)


def _training_config(settings: StudySettings) -> TrainingConfig:
    return TrainingConfig(
        num_epochs=1,
        batch_size=settings.n_collocation,
        num_collocation_points=settings.n_collocation,
        num_boundary_points=200,
        num_initial_points=100,
        learning_rate=settings.learning_rate,
        weight_decay=0.0,
        gradient_clipping=1.0,
        early_stopping=EarlyStoppingConfig(enabled=False, patience=100, min_delta=1e-4),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=50
        ),
        adaptive_weights=AdaptiveWeightsConfig(enabled=False),
        loss_weights={"residual": 1.0, "boundary": 10.0, "initial": 10.0, "data": 1.0},
    )


def _build_burgers(device: torch.device) -> PDEBase:
    """Burgers on ``[-1, 1] × [0, 1]``, ``nu = 0.01/pi``, zero Dirichlet BCs."""
    cfg = PDEConfig(
        name="Burgers Equation",
        domain=[[-1.0, 1.0]],
        time_domain=[0.0, 1.0],
        parameters={"nu": 0.01 / math.pi},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        initial_condition={"type": "sine", "amplitude": -1.0, "frequency": 1.0},
        exact_solution={},
        dimension=1,
        device=device,
    )
    return BurgersEquation(config=cfg)


def _build_allen_cahn(device: torch.device) -> PDEBase:
    """Allen-Cahn benchmark in rescaled time ``τ = 5 t`` (see :func:`allen_cahn_reference`)."""
    cfg = PDEConfig(
        name="Allen-Cahn Equation",
        domain=[[-1.0, 1.0]],
        time_domain=[0.0, 5.0],
        parameters={"epsilon": math.sqrt(0.0002)},
        boundary_conditions={"dirichlet": {"value": -1.0}},
        initial_condition={"type": "x2_cos"},
        exact_solution={},
        dimension=1,
        device=device,
    )
    return AllenCahnEquation(config=cfg)


_PDE_BUILDERS: Dict[str, Callable[[torch.device], PDEBase]] = {
    "burgers": _build_burgers,
    "allen_cahn": _build_allen_cahn,
}

_REFERENCES: Dict[str, Callable[[], ReferenceSolution]] = {
    "burgers": burgers_reference,
    "allen_cahn": allen_cahn_reference,
}


def build_reference(pde_name: str) -> ReferenceSolution:
    """Compute the numerical reference for a study PDE."""
    if pde_name not in _REFERENCES:
        raise ValueError(f"Unknown pde {pde_name!r}. Supported: {STUDY_PDES}")
    return _REFERENCES[pde_name]()


def _build_model(settings: StudySettings, device: torch.device) -> PINNModel:
    """Plain tanh MLP without layer norm — the standard PINN baseline network."""
    config = Config()
    config.device = device
    config.model = ModelConfig(
        input_dim=2,
        hidden_dim=settings.hidden_dims[0],
        output_dim=1,
        num_layers=len(settings.hidden_dims),
        activation="tanh",
        architecture="feedforward",
        layer_norm=False,
        dropout=0.0,
    )
    config.model.hidden_dims = list(settings.hidden_dims)
    return PINNModel(config=config, device=device).to(device)


def _relative_l2(model: torch.nn.Module, reference: ReferenceSolution, device) -> float:
    x, t = reference.flat_points()
    inputs = torch.tensor(np.hstack([x, t]), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        u_pred = model(inputs).cpu().numpy()
    model.train()
    return reference.relative_l2(u_pred)


def _make_sampler(
    strategy: str, pde: PDEBase, settings: StudySettings, device: torch.device
) -> Callable[[torch.nn.Module, float], Tuple[torch.Tensor, torch.Tensor]]:
    """Return ``sample(model, progress) -> (x, t)`` for a strategy."""
    n = settings.n_collocation
    (x_lo, x_hi), (t_lo, t_hi) = pde.domain[0], pde.time_domain

    if strategy == "uniform":

        def sample_uniform(_model, _progress):
            x = x_lo + torch.rand(n, 1, device=device) * (x_hi - x_lo)
            t = t_lo + torch.rand(n, 1, device=device) * (t_hi - t_lo)
            return x, t

        return sample_uniform

    if strategy == "rad":

        def sample_rad(model, _progress):
            return pde.generate_collocation_points(
                n,
                strategy="residual_based",
                model=model,
                k=settings.rad_k,
                c=settings.rad_c,
                pool_factor=settings.rad_pool_factor,
                replacement=False,
            )

        return sample_rad

    if strategy == "rl_bandit":
        bandit = ResidualBanditSampler((x_lo, x_hi), (t_lo, t_hi), device=device)

        def sample_bandit(model, progress):
            return bandit.sample(pde, model, n, progress=progress)

        return sample_bandit

    raise ValueError(f"Unknown strategy {strategy!r}. Supported: {STUDY_STRATEGIES}")


def run_study_case(
    pde_name: str,
    strategy: str,
    seed: int,
    settings: StudySettings,
    reference: ReferenceSolution,
    device: Optional[torch.device] = None,
) -> StudyRun:
    """Train one PINN under one sampler until the time budget runs out."""
    if pde_name not in _PDE_BUILDERS:
        raise ValueError(f"Unknown pde {pde_name!r}. Supported: {STUDY_PDES}")
    if strategy not in STUDY_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Supported: {STUDY_STRATEGIES}")
    device = device or torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = _build_model(settings, device)
    torch.manual_seed(seed + 10_000)  # sampling stream, independent of the init

    pde = _PDE_BUILDERS[pde_name](device)
    pde.config.training = _training_config(settings)
    sample = _make_sampler(strategy, pde, settings, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)

    budget = settings.time_budget_s
    checkpoints = [budget * (i + 1) / settings.n_checkpoints for i in range(settings.n_checkpoints)]
    curve_t: List[float] = []
    curve_e: List[float] = []
    clock = 0.0
    step = 0
    x = t = None
    while clock < budget:
        start = time.perf_counter()
        if strategy == "uniform" or step % settings.resample_every == 0:
            x, t = sample(model, clock / budget)
        optimizer.zero_grad()
        loss = pde.compute_loss(model, x, t)["total"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        clock += time.perf_counter() - start
        step += 1
        while checkpoints and clock >= checkpoints[0]:
            checkpoints.pop(0)
            curve_t.append(clock)
            curve_e.append(_relative_l2(model, reference, device))

    return StudyRun(
        pde=pde_name,
        strategy=strategy,
        seed=seed,
        rel_l2=curve_e[-1],
        steps=step,
        train_time_s=clock,
        curve_time_s=curve_t,
        curve_rel_l2=curve_e,
    )


def _worker(task: tuple) -> StudyRun:
    """Process-pool entry point; one thread per worker keeps runs comparable."""
    torch.set_num_threads(1)
    return run_study_case(*task)


def run_sampling_study(
    pdes: Sequence[str] = STUDY_PDES,
    strategies: Sequence[str] = STUDY_STRATEGIES,
    seeds: Sequence[int] = tuple(range(10)),
    settings: Optional[StudySettings] = None,
    workers: int = 1,
    on_result: Optional[Callable[[StudyRun], None]] = None,
) -> List[StudyRun]:
    """Run every ``(pde, strategy, seed)`` combination.

    Tasks are ordered seed-major so the strategies for one seed run side by
    side under similar machine load. ``workers > 1`` uses a spawn-based
    process pool; ``workers = 1`` runs inline.

    Args:
        pdes: Subset of :data:`STUDY_PDES`.
        strategies: Subset of :data:`STUDY_STRATEGIES`.
        seeds: Seeds to run; each fixes the initial weights.
        settings: Shared study settings.
        workers: Parallel worker processes.
        on_result: Optional callback invoked as each run finishes.

    Returns:
        One :class:`StudyRun` per combination, in completion order.
    """
    settings = settings or StudySettings()
    for name in pdes:
        if name not in _PDE_BUILDERS:
            raise ValueError(f"Unknown pde {name!r}. Supported: {STUDY_PDES}")
    for name in strategies:
        if name not in STUDY_STRATEGIES:
            raise ValueError(f"Unknown strategy {name!r}. Supported: {STUDY_STRATEGIES}")

    references = {name: build_reference(name) for name in pdes}
    tasks = [
        (pde, strategy, seed, settings, references[pde])
        for pde in pdes
        for seed in seeds
        for strategy in strategies
    ]

    results: List[StudyRun] = []
    if workers <= 1:
        for task in tasks:
            run = run_study_case(*task)
            results.append(run)
            if on_result:
                on_result(run)
        return results

    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        for run in pool.map(_worker, tasks):
            results.append(run)
            if on_result:
                on_result(run)
    return results


@dataclass
class StudySummary:
    """Aggregate over seeds for one ``(pde, strategy)`` pair.

    Attributes:
        pde: PDE key.
        strategy: Sampling strategy.
        n: Number of seeds.
        median_rel_l2: Median final relative L2 error.
        q25_rel_l2: 25th percentile of the final error.
        q75_rel_l2: 75th percentile of the final error.
        median_steps: Median optimizer steps within the budget.
        wins_vs_uniform: Seeds on which this strategy beat ``uniform``.
        p_vs_uniform: Two-sided Wilcoxon signed-rank p-value against
            ``uniform`` on paired seeds (``nan`` if not computable).
    """

    pde: str
    strategy: str
    n: int
    median_rel_l2: float
    q25_rel_l2: float
    q75_rel_l2: float
    median_steps: float
    wins_vs_uniform: int
    p_vs_uniform: float


def summarize(runs: Sequence[StudyRun]) -> List[StudySummary]:
    """Median/IQR per ``(pde, strategy)`` plus paired comparison against ``uniform``."""
    from scipy.stats import wilcoxon

    by_key: Dict[Tuple[str, str], Dict[int, StudyRun]] = {}
    for run in runs:
        by_key.setdefault((run.pde, run.strategy), {})[run.seed] = run

    summaries: List[StudySummary] = []
    for (pde, strategy), seed_runs in sorted(by_key.items()):
        errors = np.array([r.rel_l2 for r in seed_runs.values()])
        baseline = by_key.get((pde, "uniform"), {})
        paired = sorted(set(seed_runs) & set(baseline))
        wins = sum(seed_runs[s].rel_l2 < baseline[s].rel_l2 for s in paired)
        p_value = float("nan")
        if strategy != "uniform" and len(paired) >= 5:
            diffs = [seed_runs[s].rel_l2 - baseline[s].rel_l2 for s in paired]
            if any(d != 0 for d in diffs):
                p_value = float(wilcoxon(diffs).pvalue)
        summaries.append(
            StudySummary(
                pde=pde,
                strategy=strategy,
                n=len(errors),
                median_rel_l2=float(np.median(errors)),
                q25_rel_l2=float(np.percentile(errors, 25)),
                q75_rel_l2=float(np.percentile(errors, 75)),
                median_steps=float(np.median([r.steps for r in seed_runs.values()])),
                wins_vs_uniform=wins if strategy != "uniform" else 0,
                p_vs_uniform=p_value,
            )
        )
    return summaries


RUN_FIELDS = ("pde", "strategy", "seed", "rel_l2", "steps", "train_time_s")


def write_study_outputs(
    out_dir: Path, runs: Sequence[StudyRun], settings: StudySettings
) -> List[StudySummary]:
    """Write ``runs.csv``, ``curves.json``, ``summary.csv`` and ``settings.json``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "runs.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RUN_FIELDS)
        for r in runs:
            writer.writerow([getattr(r, f) for f in RUN_FIELDS])
    (out_dir / "curves.json").write_text(json.dumps([asdict(r) for r in runs]))
    (out_dir / "settings.json").write_text(json.dumps(asdict(settings), indent=2))

    summaries = summarize(runs)
    fields = list(asdict(summaries[0]).keys()) if summaries else []
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for s in summaries:
            writer.writerow([getattr(s, f) for f in fields])
    return summaries
