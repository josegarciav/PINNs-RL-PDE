"""Process models for the industrial twin: reference simulators, their PINN forms, and classical estimators.

* :mod:`pinnrl.process.evaporator` — a falling-film evaporator tube (J01).
* :mod:`pinnrl.process.ekf` — an extended Kalman filter and RTS smoother (J05).
"""

from pinnrl.process.ekf import ExtendedKalmanFilter, FilterResult, augment_random_walk
from pinnrl.process.evaporator import EvaporatorReference, FallingFilmEvaporator

__all__ = [
    "EvaporatorReference",
    "ExtendedKalmanFilter",
    "FallingFilmEvaporator",
    "FilterResult",
    "augment_random_walk",
]
