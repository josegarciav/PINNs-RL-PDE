"""Restart a twin's training on the one test it can make itself: the sensor misfit.

A PINN twin that identifies an unknown function from sensor records can fail
to train in ways a loss curve does not show. In experiment J06 22 of 36 runs
collapsed to a constant fouling or stuck at a bound; in J07b one closure in
three left the basin its pointwise start had put it in. In every one of those
failures the fields also missed the records by far more than their noise, so
the chi-square of the records under the trained network, which the run can
compute without knowing the truth, caught each of them.

:func:`train_with_restarts` turns that check into a rule: train, test the
misfit against its chi-square limit, and if it fails start again from a new
seed, up to a number of attempts, keeping the best run. The acceptance test is
:meth:`CustomPDE.sensor_misfit <pinnrl.pdes.custom.CustomPDE.sensor_misfit>`.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch

from pinnrl.pdes.sensors import SensorMisfit

logger = logging.getLogger(__name__)


@dataclass
class RestartResult:
    """What :func:`train_with_restarts` returns.

    ``trainer`` is the best attempt's trainer (the accepted one, or the one
    with the lowest chi-square when none was accepted); ``misfit`` its
    :class:`~pinnrl.pdes.sensors.SensorMisfit`; ``attempts`` the misfit of
    every attempt in order; ``seeds`` the seed each attempt started from;
    ``train_time_s`` the summed wall-clock of all attempts.
    """

    trainer: Any
    misfit: SensorMisfit
    attempts: list[SensorMisfit] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    train_time_s: float = 0.0

    @property
    def accepted(self) -> bool:
        """Whether the returned run passed the misfit test."""
        return self.misfit.accepted

    @property
    def n_attempts(self) -> int:
        """How many runs were trained."""
        return len(self.attempts)


def train_with_restarts(
    pde: Any,
    make_trainer: Callable[[int], Any],
    train: Callable[[Any], Any],
    *,
    attempts: int = 3,
    level: float = 0.999,
    seed: int = 0,
) -> RestartResult:
    """Train until the sensor misfit passes its chi-square test, restarting from new seeds.

    Args:
        pde: The problem; its sensors carry the records the fit is judged on.
        make_trainer: ``make_trainer(seed) -> trainer`` builds a fresh trainer
            (typically ``pde.build_trainer(...)``); ``torch.manual_seed(seed)``
            has been called just before, so the network's initialization and
            the sampler follow the seed.
        train: ``train(trainer)`` runs the training schedule in place.
        attempts: The most runs to train.
        level: The chi-square quantile a run must stay under; see
            :func:`~pinnrl.pdes.sensors.sensor_misfit`.
        seed: The first seed; attempt ``k`` uses ``seed + k``.

    Returns:
        :class:`RestartResult` with the accepted run, or the best of
        ``attempts`` when none passed (``result.accepted`` says which).

    Raises:
        ValueError: For ``attempts < 1``.
    """
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    best: RestartResult | None = None
    misfits: list[SensorMisfit] = []
    seeds: list[int] = []
    total = 0.0
    for k in range(int(attempts)):
        s = int(seed) + k
        torch.manual_seed(s)
        tic = time.perf_counter()
        trainer = make_trainer(s)
        train(trainer)
        total += time.perf_counter() - tic
        misfit = pde.sensor_misfit(trainer.model, level)
        misfits.append(misfit)
        seeds.append(s)
        logger.info("attempt %d (seed %d): %s", k + 1, s, misfit)
        if best is None or misfit.chi2 < best.misfit.chi2:
            best = RestartResult(trainer, misfit)
        if misfit.accepted:
            break
    assert best is not None
    best.attempts = misfits
    best.seeds = seeds
    best.train_time_s = total
    return best
