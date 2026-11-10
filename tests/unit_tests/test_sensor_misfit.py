"""The sensor misfit a twin can compute on itself, and the restart rule built on it."""

import math

import numpy as np
import pytest
import torch
from scipy.stats import chi2

from pinnrl.pdes import PointSensor, SensorMisfit, sensor_misfit
from pinnrl.pdes.catalog import from_catalog
from pinnrl.training.restarts import train_with_restarts

TIMES = np.linspace(0.05, 1.0, 25)


class _Exact(torch.nn.Module):
    """The heat solution with ``alpha``, as a network would be called."""

    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, inp):
        x, t = inp[:, :1], inp[:, 1:]
        return torch.exp(-self.alpha * math.pi**2 * t) * torch.sin(math.pi * x)


def _heat(noise=0.01):
    pde = from_catalog(
        "heat",
        sensors=[
            PointSensor(0.3, TIMES, noise=noise, name="probe"),
            PointSensor(0.7, TIMES, "u", noise=noise, name="probe 2"),
        ],
    )
    pde.observe(seed=0)
    return pde


def test_misfit_is_about_n_for_the_truth_and_large_for_a_wrong_model():
    pde = _heat()
    good = pde.sensor_misfit(_Exact(0.1))
    assert isinstance(good, SensorMisfit)
    assert good.n == 50 and set(good.per_sensor) == {"probe", "probe 2"}
    assert 20 < good.chi2 < 90 and good.accepted
    assert good.limit == pytest.approx(chi2.ppf(0.999, 50))
    assert 0.4 < good.reduced < 1.8
    assert "accepted" in str(good)
    bad = pde.sensor_misfit(_Exact(0.3), level=0.99)
    assert bad.chi2 > 10 * good.chi2 and not bad.accepted
    assert bad.limit == pytest.approx(chi2.ppf(0.99, 50)) and "rejected" in str(bad)
    # per sensor, and the sum of the parts
    assert sum(bad.per_sensor.values()) == pytest.approx(bad.chi2)
    assert pde.sensors[0].misfit(_Exact(0.3)) == pytest.approx(bad.per_sensor["probe"])


def test_misfit_validation():
    pde = _heat()
    with pytest.raises(ValueError, match="noise > 0"):
        PointSensor(0.3, TIMES, noise=0.0).bind(pde).set_values(np.zeros(25)).misfit(_Exact(0.1))
    with pytest.raises(ValueError, match="no records"):
        PointSensor(0.3, TIMES, noise=0.1).bind(pde).misfit(_Exact(0.1))
    with pytest.raises(ValueError, match="at least one sensor"):
        sensor_misfit([PointSensor(0.3, TIMES, noise=0.1).bind(pde)], _Exact(0.1))
    with pytest.raises(ValueError, match="level"):
        pde.sensor_misfit(_Exact(0.1), level=1.0)


class _Stub:
    def __init__(self, model):
        self.model = model


def test_restarts_stop_at_the_first_accepted_run_and_keep_the_best_otherwise():
    pde = _heat()
    seeds_seen = []

    def make_trainer(seed):
        seeds_seen.append(seed)
        return _Stub(_Exact({7: 0.3, 8: 0.1, 9: 0.2}[seed]))  # the second attempt is right

    trained = []
    result = train_with_restarts(pde, make_trainer, trained.append, attempts=3, seed=7)
    assert seeds_seen == [7, 8] and result.seeds == [7, 8] and len(trained) == 2
    assert result.accepted and result.n_attempts == 2
    assert result.trainer.model.alpha == 0.1
    assert [m.accepted for m in result.attempts] == [False, True]
    assert result.train_time_s >= 0.0
    # none accepted: all attempts run, the lowest chi-square is returned
    result = train_with_restarts(
        pde, lambda s: _Stub(_Exact({0: 0.5, 1: 0.2, 2: 0.4}[s])), lambda tr: None, attempts=3
    )
    assert not result.accepted and result.n_attempts == 3
    assert result.trainer.model.alpha == 0.2
    assert result.misfit.chi2 == min(m.chi2 for m in result.attempts)
    with pytest.raises(ValueError, match="attempts"):
        train_with_restarts(pde, make_trainer, trained.append, attempts=0)
