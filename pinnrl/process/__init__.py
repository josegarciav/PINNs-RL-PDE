"""Process models for the industrial twin: reference simulators, their PINN forms, and classical estimators.

* :mod:`pinnrl.process.evaporator` — a falling-film evaporator tube (J01).
* :mod:`pinnrl.process.droplet` — a single drying droplet with a drying-rate closure (J07).
* :mod:`pinnrl.process.dryer` — a co-current spray-drying tower: the droplet coupled to its air (J11).
* :mod:`pinnrl.process.ekf` — an extended Kalman filter and RTS smoother (J05).
* :mod:`pinnrl.process.mhe` — moving-horizon estimation with bounds on states and disturbances.
* :mod:`pinnrl.process.historian` — plant-historian exports and a tag map to inputs and sensors.
* :mod:`pinnrl.process.monitor` — the evaporator's fouling monitor: bands and the time to clean (J06).
"""

from pinnrl.process.droplet import DropletReference, DryingDroplet
from pinnrl.process.dryer import DryerReference, SprayDryer
from pinnrl.process.ekf import (
    ExtendedKalmanFilter,
    FilterResult,
    ThresholdTime,
    augment_local_linear_trend,
    augment_random_walk,
    time_to_threshold,
)
from pinnrl.process.evaporator import EvaporatorReference, FallingFilmEvaporator
from pinnrl.process.historian import Historian, TwinData
from pinnrl.process.mhe import MovingHorizonEstimator
from pinnrl.process.monitor import (
    FoulingMonitor,
    monitor_fouling,
    monitor_from_files,
    monitor_from_historian,
    write_demo_export,
)

__all__ = [
    "DropletReference",
    "DryerReference",
    "SprayDryer",
    "DryingDroplet",
    "EvaporatorReference",
    "ExtendedKalmanFilter",
    "FallingFilmEvaporator",
    "FilterResult",
    "FoulingMonitor",
    "Historian",
    "MovingHorizonEstimator",
    "ThresholdTime",
    "TwinData",
    "augment_local_linear_trend",
    "augment_random_walk",
    "monitor_fouling",
    "monitor_from_files",
    "monitor_from_historian",
    "write_demo_export",
    "time_to_threshold",
]
