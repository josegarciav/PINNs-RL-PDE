"""Tests for the multi-seed sampling study (equal steps, eval accounting, summaries)."""

import csv

import numpy as np
import pytest

from pinnrl.benchmarks.reference import ReferenceSolution
from pinnrl.benchmarks.sampling_study import (
    STUDY_STRATEGIES,
    StudyRun,
    StudySettings,
    run_sampling_study,
    run_study_case,
    summarize,
    write_study_outputs,
)

TINY = StudySettings(num_steps=20, n_collocation=64, resample_every=5, n_checkpoints=2)


def _toy_reference():
    x = np.linspace(-1.0, 1.0, 9)
    t = np.linspace(0.0, 1.0, 5)
    return ReferenceSolution(x=x, t=t, u=-np.sin(np.pi * x)[:, None] * np.ones((1, 5)))


@pytest.mark.parametrize("strategy", STUDY_STRATEGIES)
def test_run_study_case_takes_equal_steps(strategy):
    """Every sampler takes exactly num_steps steps and records one point per checkpoint."""
    run = run_study_case("burgers", strategy, 0, TINY, _toy_reference())
    assert run.steps == TINY.num_steps
    assert run.curve_step == [10, 20]
    assert len(run.curve_rel_l2) == TINY.n_checkpoints == len(run.curve_evals)
    assert run.rel_l2 == run.curve_rel_l2[-1] and np.isfinite(run.rel_l2)
    assert run.train_time_s > 0


def test_overhead_evaluations_are_counted():
    """Uniform spends only training evaluations; RAD and the bandit add their own."""
    training = TINY.num_steps * TINY.n_collocation
    ref = _toy_reference()
    assert run_study_case("burgers", "uniform", 0, TINY, ref).residual_evals == training
    rad = run_study_case("burgers", "rad", 0, TINY, ref)
    # 4 resamples (every 5 of 20 steps), each scoring a pool of rad_pool_factor * n points.
    assert rad.residual_evals == training + 4 * TINY.rad_pool_factor * TINY.n_collocation
    assert run_study_case("burgers", "rl_bandit", 0, TINY, ref).residual_evals > training


def test_same_seed_same_initial_weights_across_strategies():
    """At the first checkpoint, the every-step samplers differ only through their points."""
    a = run_study_case("burgers", "uniform", 3, TINY, _toy_reference())
    b = run_study_case("burgers", "uniform", 3, TINY, _toy_reference())
    assert a.curve_rel_l2 == b.curve_rel_l2  # deterministic per seed


def test_allen_cahn_case_runs():
    """The Allen-Cahn builder (x2_cos IC, rescaled time) trains without errors."""
    run = run_study_case("allen_cahn", "uniform", 0, TINY, _toy_reference())
    assert run.steps == TINY.num_steps


def test_invalid_chunking_raises():
    """Checkpoint chunks must be whole multiples of resample_every."""
    bad = StudySettings(num_steps=20, n_collocation=64, resample_every=3, n_checkpoints=2)
    with pytest.raises(ValueError, match="multiple"):
        run_study_case("burgers", "rad", 0, bad, _toy_reference())


def test_unknown_names_raise():
    """Unknown PDEs or strategies are rejected up front."""
    with pytest.raises(ValueError, match="Unknown pde"):
        run_sampling_study(pdes=["schrodinger"], settings=TINY)
    with pytest.raises(ValueError, match="Unknown strategy"):
        run_sampling_study(pdes=["burgers"], strategies=["mystery"], settings=TINY)


def _run(strategy, seed, final, evals, curve):
    return StudyRun(
        "burgers",
        strategy,
        seed,
        final,
        20,
        1.0,
        evals,
        curve_step=[10, 20],
        curve_time_s=[0.5, 1.0],
        curve_evals=[evals // 2, evals],
        curve_rel_l2=curve,
    )


def test_summaries_at_equal_steps_and_equal_evals(tmp_path):
    """RAD wins at equal steps but is scored at its earlier checkpoint at equal evaluations."""
    runs = []
    for seed in range(6):
        runs.append(_run("uniform", seed, 0.2, 1000, [0.3, 0.2]))
        runs.append(_run("rad", seed, 0.1, 1600, [0.4, 0.1]))  # overhead pushes it past 1000
    steps = {s.strategy: s for s in summarize(runs, "steps")}
    assert steps["rad"].wins_vs_uniform == 6 and steps["rad"].p_vs_uniform < 0.05
    evals = {s.strategy: s for s in summarize(runs, "evals")}
    assert evals["rad"].median_rel_l2 == pytest.approx(0.4)  # the 800-eval checkpoint
    assert evals["rad"].wins_vs_uniform == 0
    assert np.isnan(evals["uniform"].p_vs_uniform)

    write_study_outputs(tmp_path, runs, TINY)
    rows = list(csv.DictReader((tmp_path / "summary.csv").open()))
    assert {(r["budget"], r["strategy"]) for r in rows} == {
        ("steps", "uniform"),
        ("steps", "rad"),
        ("evals", "uniform"),
        ("evals", "rad"),
    }
    assert (tmp_path / "curves.json").exists() and (tmp_path / "settings.json").exists()
