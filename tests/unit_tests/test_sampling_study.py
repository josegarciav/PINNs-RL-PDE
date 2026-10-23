"""Smoke tests for the equal-wall-clock sampling study."""

import csv

import numpy as np
import pytest

from pinnrl.benchmarks.reference import ReferenceSolution
from pinnrl.benchmarks.sampling_study import (
    StudyRun,
    StudySettings,
    run_sampling_study,
    run_study_case,
    summarize,
    write_study_outputs,
)

TINY = StudySettings(time_budget_s=0.3, n_collocation=64, resample_every=5, n_checkpoints=2)


def _toy_reference():
    x = np.linspace(-1.0, 1.0, 9)
    t = np.linspace(0.0, 1.0, 5)
    return ReferenceSolution(x=x, t=t, u=-np.sin(np.pi * x)[:, None] * np.ones((1, 5)))


@pytest.mark.parametrize("strategy", ["uniform", "rad", "rl_bandit"])
def test_run_study_case_respects_budget(strategy):
    """Each sampler trains until the budget and records one point per checkpoint."""
    run = run_study_case("burgers", strategy, 0, TINY, _toy_reference())
    assert run.steps > 0
    assert run.train_time_s >= TINY.time_budget_s
    assert len(run.curve_rel_l2) == TINY.n_checkpoints
    assert run.rel_l2 == run.curve_rel_l2[-1] and np.isfinite(run.rel_l2)


def test_allen_cahn_case_runs():
    """The Allen-Cahn builder (x2_cos IC, rescaled time) trains without errors."""
    run = run_study_case("allen_cahn", "uniform", 0, TINY, _toy_reference())
    assert run.steps > 0


def test_unknown_names_raise():
    """Unknown PDEs or strategies are rejected up front."""
    with pytest.raises(ValueError, match="Unknown pde"):
        run_sampling_study(pdes=["schrodinger"], settings=TINY)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_sampling_study(pdes=["burgers"], strategies=["mystery"], settings=TINY)


def test_summary_pairs_seeds_against_uniform(tmp_path):
    """Wins and the Wilcoxon p-value are computed on seed-paired runs."""
    runs = []
    for seed in range(6):
        runs.append(StudyRun("burgers", "uniform", seed, 0.2, 100, 1.0))
        runs.append(StudyRun("burgers", "rad", seed, 0.1, 90, 1.0))
    summaries = {s.strategy: s for s in summarize(runs)}
    assert summaries["rad"].wins_vs_uniform == 6
    assert summaries["rad"].p_vs_uniform < 0.05
    assert np.isnan(summaries["uniform"].p_vs_uniform)

    write_study_outputs(tmp_path, runs, TINY)
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert {r["strategy"] for r in rows} == {"uniform", "rad"}
    assert (tmp_path / "curves.json").exists() and (tmp_path / "settings.json").exists()
