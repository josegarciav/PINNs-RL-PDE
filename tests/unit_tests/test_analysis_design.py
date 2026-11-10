"""Linear-Gaussian measurement design: the posterior formula, the criteria and greedy selection."""

import itertools
import math

import numpy as np
import pytest

from pinnrl.analysis import (
    DesignResult,
    greedy_design,
    information,
    information_gain,
    posterior_covariance,
    target_std,
)


def test_posterior_of_a_mean_and_its_information_gain():
    J = np.ones((25, 1))
    F = information(J, 0.5)
    assert F[0, 0] == pytest.approx(100.0)
    cov = posterior_covariance(F, [2.0])
    assert cov[0, 0] == pytest.approx(1.0 / (100.0 + 0.25))
    assert posterior_covariance(F)[0, 0] == pytest.approx(0.01)  # flat prior
    assert posterior_covariance(F, [math.inf])[0, 0] == pytest.approx(0.01)
    assert information_gain(np.array([[4.0]]), cov) == pytest.approx(0.5 * math.log(1 + 400.0))
    # per-row noise
    assert information(J, np.full(25, 0.5))[0, 0] == pytest.approx(100.0)
    assert target_std(np.diag([4.0, 9.0]), np.array([1.0, 1.0])) == pytest.approx(math.sqrt(13))
    assert target_std(np.diag([4.0, 9.0]), np.eye(2)) == pytest.approx(math.sqrt(13))


def test_validation_and_undetermined_directions():
    J = np.array([[1.0, 1.0], [2.0, 2.0]])  # only the sum is seen
    with pytest.raises(np.linalg.LinAlgError, match="undetermined"):
        posterior_covariance(information(J, 1.0))
    cov = posterior_covariance(information(J, 1.0), [1.0, math.inf])  # a prior on one closes it
    assert np.all(np.isfinite(cov))
    with pytest.raises(ValueError, match="one entry per row"):
        information(J, np.ones(3))
    with pytest.raises(ValueError, match="positive"):
        information(J, 0.0)
    with pytest.raises(ValueError, match="one entry per parameter"):
        posterior_covariance(np.eye(2), [1.0])
    with pytest.raises(ValueError, match="without repeat"):
        greedy_design([J[0], J[1]], 1.0, 3)
    with pytest.raises(ValueError, match="at least one"):
        greedy_design([], 1.0, 1)
    with pytest.raises(ValueError, match="one entry per candidate"):
        greedy_design([J[0], J[1]], [1.0], 1)


def test_greedy_picks_the_ends_of_a_line_and_matches_exhaustive_search():
    t = np.linspace(0.0, 1.0, 11)
    rows = [np.array([1.0, ti]) for ti in t]  # y = a + b t
    d = greedy_design(rows, 0.1, 2, prior_std=[10.0, 10.0])
    assert isinstance(d, DesignResult) and sorted(d.chosen) == [0, 10]  # D-optimal: the two ends
    assert d.criterion[0] > d.criterion[1] > d.criterion[2]
    # the value at t = 1.5 (an extrapolation), three samples on top of a weak record
    target = np.array([1.0, 1.5])
    base = information(np.array([[1.0, 0.5]]), 0.5)
    prior = [1.0, 1.0]
    d = greedy_design(rows, 0.1, 3, prior_std=prior, base_information=base, target=target)
    best = min(
        target_std(
            posterior_covariance(base + sum(information(rows[i], 0.1) for i in combo), prior),
            target,
        )
        for combo in itertools.combinations(range(11), 3)
    )
    assert d.criterion[-1] == pytest.approx(best, rel=0.02)
    assert d.criterion[-1] == pytest.approx(target_std(d.covariance, target))
    assert 10 in d.chosen  # the sample nearest the target's time
    # replicates: with repeat the far end may be taken again
    rep = greedy_design(rows, 0.1, 3, prior_std=prior, target=target, repeat=True)
    assert len(rep.chosen) == 3 and rep.criterion[-1] <= d.criterion[0]
    # groups of rows and per-candidate noise: the precise sensor is picked first
    groups = [np.stack(rows[:5]), np.stack(rows[5:])]
    assert greedy_design(groups, [1.0, 0.01], 1, prior_std=prior).chosen == [1]
