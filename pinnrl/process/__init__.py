"""Process models for the industrial twin: reference simulators, their PINN forms, and classical estimators.

* :mod:`pinnrl.process.evaporator` — a falling-film evaporator tube (J01).
* :mod:`pinnrl.process.droplet` — a single drying droplet with a drying-rate closure (J07).
* :mod:`pinnrl.process.ekf` — an extended Kalman filter and RTS smoother (J05).
"""

from pinnrl.process.droplet import DropletReference, DryingDroplet
from pinnrl.process.ekf import ExtendedKalmanFilter, FilterResult, augment_random_walk
from pinnrl.process.evaporator import EvaporatorReference, FallingFilmEvaporator

__all__ = [
    "DropletReference",
    "DryingDroplet",
    "EvaporatorReference",
    "ExtendedKalmanFilter",
    "FallingFilmEvaporator",
    "FilterResult",
    "augment_random_walk",
]
