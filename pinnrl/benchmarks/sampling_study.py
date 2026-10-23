"""Multi-seed sampling study at equal training budget, scored against numerical references.

Answers the question an adaptive sampler has to win: *for the same training
budget, does it give a lower error, and does it still win once its own
overhead is charged?*

* **PDEs** — the standard Burgers (``nu = 0.01/pi``, ``u0 = -sin(pi x)``) and
  Allen-Cahn (``u_t = 0.001 u_xx + 5(u - u³)``, ``u0 = x² cos(pi x)``)
  benchmarks, scored against :mod:`pinnrl.benchmarks.reference`.
* **Strategies** — ``uniform`` and ``stratified`` (fresh points every step),
  ``uniform_periodic`` (uniform, refreshed every ``resample_every`` steps; the
  control that separates resampling frequency from adaptivity), ``rad``
  (residual-based adaptive distribution, Wu et al. 2023) and ``rl_bandit``
  (:class:`pinnrl.rl.bandit_sampler.ResidualBanditSampler`).
* **Budget** — every run takes ``num_steps`` optimizer steps on
  ``n_collocation`` points (equal training points). Each checkpoint records the
  training wall-clock and the residual evaluations spent so far, *including* the
  sampler's own (RAD candidate pools, bandit probes), so the same run can be
  compared at equal steps (:func:`summarize` with ``budget="steps"``) or at
  equal total evaluations (``budget="evals"``).
* **Training** — :meth:`pinnrl.training.trainer.PDETrainer.train_steps`, the
  same per-step update as ``PDETrainer.train``.
* **Seeds** — a seed fixes the initial network weights, so strategies are
  compared pairwise on identical initializations.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import multiprocessing as mp
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

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
from pinnrl.training.trainer import PDETrainer

STUDY_PDES = ("burgers", "allen_cahn")
STUDY_STRATEGIES = ("uniform", "stratified", "uniform_periodic", "rad", "rl_bandit")
_EVERY_STEP = ("uniform", "stratified")  # fresh points on every optimizer step


@dataclass
class StudySettings:
    """Shared knobs for every run in a study.

    Attributes:
        num_steps: Optimizer steps per run (the equal training budget).
        n_collocation: Collocation points per step.
        resample_every: Steps between resampling for ``uniform_periodic``,
            ``rad`` and ``rl_bandit``.
        learning_rate: Adam learning rate (constant).
        hidden_dims: Widths of the tanh MLP.
        n_checkpoints: Evaluations along the run; ``num_steps / n_checkpoints``
            must be a multiple of ``resample_every``.
        rad_k: RAD exponent ``k``.
        rad_c: RAD uniform floor ``c``.
        rad_pool_factor: RAD candidate pool size as a multiple of ``n_collocation``.
    """

    num_steps: int = 15_000
    n_collocation: int = 1000
    resample_every: int = 100
    learning_rate: float = 1e-3
    hidden_dims: tuple[int, ...] = (64, 64, 64, 64)
    n_checkpoints: int = 50
    rad_k: float = 1.0
    rad_c: float = 1.0
    rad_pool_factor: int = 20

    @property
    def chunk(self) -> int:
        """Steps between checkpoints."""
        return self.num_steps // self.n_checkpoints


@dataclass
class StudyRun:
    """Outcome of one ``(pde, strategy, seed)`` run.

    Attributes:
        pde: PDE key.
        strategy: Sampling strategy.
        seed: Seed for the initial weights; ``seed + 10_000`` seeds sampling.
        rel_l2: Final relative L2 error against the reference.
        steps: Optimizer steps taken.
        train_time_s: Training wall-clock (sampling + steps, evaluation excluded).
        residual_evals: Residual evaluations, training points plus sampler overhead.
        curve_step: Steps at each checkpoint.
        curve_time_s: Training time at each checkpoint.
        curve_evals: Residual evaluations at each checkpoint.
        curve_rel_l2: Relative L2 error at each checkpoint.
    """

    pde: str
    strategy: str
    seed: int
    rel_l2: float
    steps: int
    train_time_s: float
    residual_evals: int
    curve_step: list[int] = field(default_factory=list)
    curve_time_s: list[float] = field(default_factory=list)
    curve_evals: list[int] = field(default_factory=list)
    curve_rel_l2: list[float] = field(default_factory=list)


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
            type="constant", warmup_epochs=0, min_lr=1e-6, factor=0.5, patience=50
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


_PDE_BUILDERS: dict[str, Callable[[torch.device], PDEBase]] = {
    "burgers": _build_burgers,
    "allen_cahn": _build_allen_cahn,
}

_REFERENCES: dict[str, Callable[[], ReferenceSolution]] = {
    "burgers": burgers_reference,
    "allen_cahn": allen_cahn_reference,
}


def build_reference(pde_name: str) -> ReferenceSolution:
    """Compute the numerical reference for a study PDE."""
    if pde_name not in _REFERENCES:
        raise ValueError(f"Unknown pde {pde_name!r}. Supported: {STUDY_PDES}")
    return _REFERENCES[pde_name]()


def _build_trainer(pde: PDEBase, settings: StudySettings, device: torch.device) -> PDETrainer:
    """Plain tanh MLP without layer norm (the standard PINN baseline) and its trainer."""
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
    config.training = _training_config(settings)
    pde.config.training = config.training
    model = PINNModel(config=config, device=device).to(device)
    return PDETrainer(model=model, pde=pde, config=config, device=device)


def _relative_l2(model: torch.nn.Module, reference: ReferenceSolution, device) -> float:
    x, t = reference.flat_points()
    inputs = torch.tensor(np.hstack([x, t]), dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        u_pred = model(inputs).cpu().numpy()
    model.train()
    return reference.relative_l2(u_pred)


def _make_sampler(
    strategy: str,
    pde: PDEBase,
    settings: StudySettings,
    device: torch.device,
    overhead: dict[str, int],
) -> Callable[[torch.nn.Module, float], tuple[torch.Tensor, torch.Tensor]]:
    """Return ``sample(model, progress) -> (x, t)``; extra residual evaluations go to ``overhead``."""
    n = settings.n_collocation
    (x_lo, x_hi), (t_lo, t_hi) = pde.domain[0], pde.time_domain

    if strategy in ("uniform", "uniform_periodic"):

        def sample_uniform(_model, _progress):
            x = x_lo + torch.rand(n, 1, device=device) * (x_hi - x_lo)
            t = t_lo + torch.rand(n, 1, device=device) * (t_hi - t_lo)
            return x, t

        return sample_uniform

    if strategy == "stratified":

        def sample_stratified(_model, _progress):
            return pde.generate_collocation_points(n, strategy="stratified")

        return sample_stratified

    if strategy == "rad":

        def sample_rad(model, _progress):
            overhead["evals"] += n * settings.rad_pool_factor
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
            # Default probe_every=1: every call probes.
            overhead["evals"] += min(bandit.probe_cells, bandit.n_cells) * bandit.probe_points
            return bandit.sample(pde, model, n, progress=progress)

        return sample_bandit

    raise ValueError(f"Unknown strategy {strategy!r}. Supported: {STUDY_STRATEGIES}")


def run_study_case(
    pde_name: str,
    strategy: str,
    seed: int,
    settings: StudySettings,
    reference: ReferenceSolution,
    device: torch.device | None = None,
) -> StudyRun:
    """Train one PINN for ``settings.num_steps`` steps under one sampler."""
    if pde_name not in _PDE_BUILDERS:
        raise ValueError(f"Unknown pde {pde_name!r}. Supported: {STUDY_PDES}")
    if strategy not in STUDY_STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}. Supported: {STUDY_STRATEGIES}")
    chunk = settings.chunk
    if chunk * settings.n_checkpoints != settings.num_steps or chunk % settings.resample_every:
        raise ValueError(
            "num_steps must split into n_checkpoints chunks that are each a multiple "
            f"of resample_every (got {settings.num_steps}, {settings.n_checkpoints}, "
            f"{settings.resample_every})."
        )
    device = device or torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)
    pde = _PDE_BUILDERS[pde_name](device)
    trainer = _build_trainer(pde, settings, device)
    torch.manual_seed(seed + 10_000)  # sampling stream, independent of the init

    overhead = {"evals": 0}
    progress = {"value": 0.0}
    sampler = _make_sampler(strategy, pde, settings, device, overhead)
    resample_every = 1 if strategy in _EVERY_STEP else settings.resample_every

    def sample_fn(model, _chunk_progress):
        return sampler(model, progress["value"])

    run = StudyRun(pde_name, strategy, seed, float("nan"), 0, 0.0, 0)
    for _ in range(settings.n_checkpoints):
        out = trainer.train_steps(
            num_steps=chunk, sample_fn=sample_fn, resample_every=resample_every
        )
        run.steps += out["steps"]
        run.train_time_s += out["train_time_s"]
        progress["value"] = run.steps / settings.num_steps
        run.residual_evals = run.steps * settings.n_collocation + overhead["evals"]
        run.curve_step.append(run.steps)
        run.curve_time_s.append(run.train_time_s)
        run.curve_evals.append(run.residual_evals)
        run.curve_rel_l2.append(_relative_l2(trainer.model, reference, device))
    run.rel_l2 = run.curve_rel_l2[-1]
    return run


def _worker(task: tuple) -> StudyRun:
    """Process-pool entry point; one thread per worker keeps runs comparable."""
    torch.set_num_threads(1)
    logging.getLogger("pinnrl").setLevel(logging.WARNING)
    return run_study_case(*task)


def run_sampling_study(
    pdes: Sequence[str] = STUDY_PDES,
    strategies: Sequence[str] = STUDY_STRATEGIES,
    seeds: Sequence[int] = tuple(range(10)),
    settings: StudySettings | None = None,
    workers: int = 1,
    on_result: Callable[[StudyRun], None] | None = None,
) -> list[StudyRun]:
    """Run every ``(pde, strategy, seed)`` combination.

    Tasks are ordered by PDE, then seed, then strategy, so the strategies for
    one seed run side by side under similar machine load. ``workers > 1`` uses a spawn-based
    process pool; ``workers = 1`` runs inline.

    Args:
        pdes: Subset of :data:`STUDY_PDES`.
        strategies: Subset of :data:`STUDY_STRATEGIES`.
        seeds: Seeds to run; each fixes the initial weights.
        settings: Shared study settings.
        workers: Parallel worker processes.
        on_result: Optional callback invoked as each run finishes.

    Returns:
        One :class:`StudyRun` per combination, in task order.
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

    results: list[StudyRun] = []
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
    """Aggregate over seeds for one ``(pde, strategy)`` pair at one budget.

    Attributes:
        budget: ``"steps"`` (equal training points) or ``"evals"`` (equal
            residual evaluations including sampler overhead).
        pde: PDE key.
        strategy: Sampling strategy.
        n: Number of seeds.
        median_rel_l2: Median relative L2 error at the budget.
        q25_rel_l2: 25th percentile.
        q75_rel_l2: 75th percentile.
        median_time_s: Median training wall-clock at the budget.
        wins_vs_uniform: Seeds on which this strategy beat ``uniform`` (0 for
            ``uniform`` itself).
        p_vs_uniform: Two-sided Wilcoxon signed-rank p-value against
            ``uniform`` on paired seeds; ``nan`` for ``uniform`` itself, with
            fewer than 5 paired seeds, or when every paired difference is 0.
    """

    budget: str
    pde: str
    strategy: str
    n: int
    median_rel_l2: float
    q25_rel_l2: float
    q75_rel_l2: float
    median_time_s: float
    wins_vs_uniform: int
    p_vs_uniform: float


def _at_budget(run: StudyRun, max_evals: int | None) -> tuple[float, float]:
    """``(rel_l2, time_s)`` at the last checkpoint within ``max_evals``.

    ``max_evals=None`` means the end of the run; if no checkpoint fits the
    cap, the first checkpoint is used.
    """
    if max_evals is None:
        return run.rel_l2, run.train_time_s
    idx = [i for i, e in enumerate(run.curve_evals) if e <= max_evals]
    i = idx[-1] if idx else 0
    return run.curve_rel_l2[i], run.curve_time_s[i]


def summarize(runs: Sequence[StudyRun], budget: str = "steps") -> list[StudySummary]:
    """Median/IQR per ``(pde, strategy)`` plus a paired comparison against ``uniform``.

    ``budget="steps"`` compares final errors (equal training points).
    ``budget="evals"`` compares each run at the last checkpoint whose residual
    evaluations (training plus sampler overhead) do not exceed the smallest
    total spent by any run on that PDE.
    """
    from scipy.stats import wilcoxon

    if budget not in ("steps", "evals"):
        raise ValueError(f"budget must be 'steps' or 'evals', got {budget!r}")
    cap: dict[str, int] = {}
    if budget == "evals":
        for run in runs:
            cap[run.pde] = min(cap.get(run.pde, run.residual_evals), run.residual_evals)

    by_key: dict[tuple[str, str], dict[int, tuple[float, float]]] = {}
    for run in runs:
        point = _at_budget(run, cap.get(run.pde))
        by_key.setdefault((run.pde, run.strategy), {})[run.seed] = point

    summaries: list[StudySummary] = []
    for (pde, strategy), seed_vals in sorted(by_key.items()):
        errors = np.array([v[0] for v in seed_vals.values()])
        baseline = by_key.get((pde, "uniform"), {})
        paired = sorted(set(seed_vals) & set(baseline))
        wins = sum(seed_vals[s][0] < baseline[s][0] for s in paired)
        p_value = float("nan")
        if strategy != "uniform" and len(paired) >= 5:
            diffs = [seed_vals[s][0] - baseline[s][0] for s in paired]
            if any(d != 0 for d in diffs):
                p_value = float(wilcoxon(diffs).pvalue)
        summaries.append(
            StudySummary(
                budget=budget,
                pde=pde,
                strategy=strategy,
                n=len(errors),
                median_rel_l2=float(np.median(errors)),
                q25_rel_l2=float(np.percentile(errors, 25)),
                q75_rel_l2=float(np.percentile(errors, 75)),
                median_time_s=float(np.median([v[1] for v in seed_vals.values()])),
                wins_vs_uniform=wins if strategy != "uniform" else 0,
                p_vs_uniform=p_value,
            )
        )
    return summaries


RUN_FIELDS = ("pde", "strategy", "seed", "rel_l2", "steps", "train_time_s", "residual_evals")


def write_study_outputs(
    out_dir: Path, runs: Sequence[StudyRun], settings: StudySettings
) -> list[StudySummary]:
    """Write ``runs.csv``, ``curves.json``, ``summary.csv`` and ``settings.json``.

    ``summary.csv`` holds both budgets (column ``budget``); the returned list
    is the same rows.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "runs.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(RUN_FIELDS)
        for r in runs:
            writer.writerow([getattr(r, f) for f in RUN_FIELDS])
    (out_dir / "curves.json").write_text(json.dumps([asdict(r) for r in runs]))
    (out_dir / "settings.json").write_text(json.dumps(asdict(settings), indent=2))

    summaries = summarize(runs, "steps") + summarize(runs, "evals")
    fields = list(asdict(summaries[0]).keys()) if summaries else []
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(fields)
        for s in summaries:
            writer.writerow([getattr(s, f) for f in fields])
    return summaries
