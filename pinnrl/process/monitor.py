"""A fouling monitor for the evaporator: a soft sensor with its band and the time to the next cleaning.

This is the estimator experiment J06 arrives at, packaged for use on a
record: an extended Kalman filter on the method-of-lines model with the
fouling resistance as a local linear trend (a level and a rate) and,
optionally, the offset of the steam-temperature record as one more constant
state under its prior. It returns the fouling with its interval at every
record, the hindsight (smoothed) estimate, the outlet records against the
filter's one-step predictions, and the time until the fouling reaches a
cleaning threshold. The dashboard's Process view plots it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from pinnrl.process.ekf import ExtendedKalmanFilter, ThresholdTime, time_to_threshold
from pinnrl.process.evaporator import FallingFilmEvaporator, _as_fn
from pinnrl.process.historian import Historian, TwinData


@dataclass
class FoulingMonitor:
    """The fouling of an evaporator tube over a record, with its uncertainty.

    Attributes:
        t: Record times, s.
        fouling: The filtered fouling resistance at each record (what was known then), m²K/W.
        lower: Lower end of its central interval at ``level``.
        upper: Upper end.
        smoothed: The hindsight estimate from the whole record (RTS smoother).
        smoothed_lower: Lower end of the smoothed interval.
        smoothed_upper: Upper end.
        rate: The filtered fouling rate, m²K/W per second.
        rate_std: Its standard deviation.
        records: The outlet records ``(K, 2)``, solids fraction and flow.
        predicted: The filter's prediction of each record before seeing it.
        misfit: Mean squared prediction error of each sensor in units of its
            noise variance, after the first tenth of the record: near 1 when
            the model explains the records, well above when it does not.
        steam_bias: Estimated offset of the steam record and its standard
            deviation, K, or ``None`` when it was not estimated.
        threshold: The cleaning threshold, or ``None``.
        time_to_clean: Time from the last record until the fouling reaches the
            threshold, or ``None`` without a threshold.
        level: Nominal level of the intervals.
    """

    t: np.ndarray
    fouling: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    smoothed: np.ndarray
    smoothed_lower: np.ndarray
    smoothed_upper: np.ndarray
    rate: np.ndarray
    rate_std: np.ndarray
    records: np.ndarray
    predicted: np.ndarray
    misfit: tuple[float, float]
    steam_bias: tuple[float, float] | None
    threshold: float | None
    time_to_clean: ThresholdTime | None
    level: float

    def summary(self) -> str:
        """One paragraph: the fouling now, its rate, the misfit and the time to clean."""
        lines = [
            f"Fouling at t = {self.t[-1]:.0f} s: {self.fouling[-1]:.3e} m²K/W, "
            f"{100 * self.level:.0f} % interval [{self.lower[-1]:.3e}, {self.upper[-1]:.3e}].",
            f"Rate: {3600 * self.rate[-1]:.3e} ± {3600 * self.rate_std[-1]:.1e} m²K/W per hour.",
            f"Prediction misfit (1 = sensor noise): solids {self.misfit[0]:.2f}, flow {self.misfit[1]:.2f}.",
        ]
        if self.steam_bias is not None:
            lines.append(
                f"Steam record offset: {self.steam_bias[0]:+.2f} ± {self.steam_bias[1]:.2f} K."
            )
        if self.time_to_clean is not None:
            tt = self.time_to_clean
            if np.isfinite(tt.median):
                upper = f"{tt.upper / 60:.0f}" if np.isfinite(tt.upper) else "no limit"
                lines.append(
                    f"Cleaning threshold {self.threshold:.2e} reached in {tt.median / 60:.0f} min "
                    f"({100 * tt.level:.0f} % interval {tt.lower / 60:.0f} to {upper}; "
                    f"reached at the present rate with probability {tt.probability:.2f})."
                )
            else:
                lines.append(
                    f"Cleaning threshold {self.threshold:.2e}: not reached at the present rate "
                    f"(probability {tt.probability:.2f})."
                )
        return " ".join(lines)


def monitor_fouling(
    model: FallingFilmEvaporator,
    times: np.ndarray,
    records: np.ndarray,
    *,
    nz: int = 51,
    noise: tuple[float, float] = (0.002, 0.001),
    fouling_std: float = 1e-4,
    rate_std: float = 1e-6,
    rate_walk: float = 1e-9,
    steam_bias_std: float | None = None,
    threshold: float | None = None,
    level: float = 0.9,
) -> FoulingMonitor:
    """Estimate the fouling resistance of ``model`` from outlet records.

    Args:
        model: The evaporator with its recorded signals (``T_s``, ``W_in``,
            ``X_in`` as functions of time); its ``R_f`` at ``t = 0`` is the
            prior mean of the fouling (0 after a cleaning).
        times: Record times, s, increasing, after ``t = 0``.
        records: ``(K, 2)``, the outlet solids fraction and the outlet flow.
        nz: Axial cells of the filter's model. A coarse grid biases the
            estimate by a few percent (3 % at 11 cells, under 0.5 % at 51),
            which matters once the interval is that narrow (J06).
        noise: Standard deviations of the two sensors.
        fouling_std: Prior standard deviation of the fouling at ``t = 0``.
        rate_std: Prior standard deviation of its rate, per second.
        rate_walk: Random-walk intensity of the rate, per second^1.5: how fast
            the rate itself may change. A rate ``r`` that may change by its
            own size over a time ``tau`` corresponds to ``r / sqrt(tau)``.
            Too small a value on a fouling curve that bends gives a narrow
            band around a biased estimate (J06: coverage 7 % instead of 98 %
            on a saturating curve); a larger value is safe and widens the
            band and, more so, the time to clean.
        steam_bias_std: Prior standard deviation of a constant offset of the
            steam record, K. ``None`` trusts the record. A steam offset mimics
            fouling (0.5 K for 30 %, J04), so an interval that ignores a
            plausible offset is too narrow.
        threshold: Fouling at which the tube is cleaned; gives ``time_to_clean``.
        level: Nominal level of the intervals.
    """
    times = np.asarray(times, dtype=float)
    records = np.asarray(records, dtype=float).reshape(times.size, 2)
    with_bias = steam_bias_std is not None
    f, h, s0 = model.fouling_state_space(nz, steam_bias=with_bias)
    n = s0.size
    k_rate, k_bias = n + 1, n + 2
    dim = n + 2 + int(with_bias)

    def dynamics(t: float, x: np.ndarray) -> np.ndarray:
        theta = x[[n, k_bias]] if with_bias else x[n : n + 1]
        out = np.zeros(dim)
        out[:n] = f(t, x[:n], theta)
        out[n] = x[k_rate]
        return out

    x0, P0, Q = np.zeros(dim), np.zeros(dim), np.zeros(dim)
    x0[:n], P0[:n], Q[:n] = s0, 1e-6, 1e-12
    x0[n] = max(float(_as_fn(model.R_f)(0.0)), 0.0)
    P0[n], P0[k_rate], Q[k_rate] = fouling_std**2, rate_std**2, rate_walk**2
    if with_bias:
        assert steam_bias_std is not None  # with_bias is exactly that
        P0[k_bias] = float(steam_bias_std) ** 2
    ekf = ExtendedKalmanFilter(
        dynamics, lambda t, x: h(t, x[:n]), Q=Q, R=np.array([noise[0] ** 2, noise[1] ** 2])
    )
    # The filter's state is not clipped at zero: clipping the fouling alone leaves the
    # states it is correlated with (the steam offset) inconsistent and costs coverage (J06).
    # The reported estimate and band are cut at zero instead.
    res = ekf.run(times, records, x0, np.diag(P0), smooth=True)
    z = float(norm.ppf(0.5 + level / 2))
    std, std_s = res.std(), res.std(smoothed=True)
    smoothed = res.smoothed
    assert smoothed is not None  # the run above asks for the smoother
    predicted = records - res.innovations
    start = max(1, times.size // 10)
    misfit = np.mean((res.innovations[start:] / np.asarray(noise)[None, :]) ** 2, axis=0)
    tt = None
    if threshold is not None:
        idx = [n, k_rate]
        tt = time_to_threshold(
            res.filtered[-1, n],
            res.filtered[-1, k_rate],
            res.filtered_cov[-1][np.ix_(idx, idx)],
            threshold,
            level=level,
            rising=True,
        )
    return FoulingMonitor(
        t=times,
        fouling=np.maximum(res.filtered[:, n], 0.0),
        lower=np.maximum(res.filtered[:, n] - z * std[:, n], 0.0),
        upper=np.maximum(res.filtered[:, n] + z * std[:, n], 0.0),
        smoothed=np.maximum(smoothed[:, n], 0.0),
        smoothed_lower=np.maximum(smoothed[:, n] - z * std_s[:, n], 0.0),
        smoothed_upper=np.maximum(smoothed[:, n] + z * std_s[:, n], 0.0),
        rate=res.filtered[:, k_rate],
        rate_std=std[:, k_rate],
        records=records,
        predicted=predicted,
        misfit=(float(misfit[0]), float(misfit[1])),
        steam_bias=(float(res.filtered[-1, k_bias]), float(std[-1, k_bias])) if with_bias else None,
        threshold=threshold,
        time_to_clean=tt,
        level=level,
    )


def monitor_from_historian(
    data: TwinData,
    model: Mapping[str, Any] | None = None,
    **options: Any,
) -> FoulingMonitor:
    """Run :func:`monitor_fouling` on a historian window (:meth:`Historian.apply`).

    The tag map's inputs ``T_s``, ``W_in`` and ``X_in`` become the model's
    signals (those it does not give keep the model's constants); the two
    records are the sensors reporting ``X`` and ``W``, which must share their
    timestamps, and their ``noise`` is the filter's. ``model`` overrides
    fields of :class:`FallingFilmEvaporator` (``length``, ``U``, ...);
    ``options`` go to :func:`monitor_fouling`.

    Raises:
        ValueError: If a sensor for ``X`` or ``W`` is missing or has no noise level.
    """
    fields = dict(model or {})
    for name in ("T_s", "W_in", "X_in"):
        if name in data.inputs:
            fields[name] = data.inputs[name].numpy
    by_expression = {s.expression: s for s in data.sensors}
    missing = [e for e in ("X", "W") if e not in by_expression]
    if missing:
        raise ValueError(f"the tag map needs sensors reporting {missing} at the outlet")
    sensors = [by_expression["X"], by_expression["W"]]
    if any(s.noise <= 0 for s in sensors):
        raise ValueError("give each outlet sensor its noise level in the tag map")
    times, records = data.records([s.name for s in sensors])
    options.setdefault("noise", (sensors[0].noise, sensors[1].noise))
    return monitor_fouling(FallingFilmEvaporator(**fields), times, records, **options)


def write_demo_export(folder: str | Path, seed: int = 0) -> tuple[Path, Path]:
    """Write a synthetic historian export of a fouling evaporator and its tag map.

    Twenty minutes of the default tube with the steam wandering by 1 K and the
    fouling growing to 2e-4 m²K/W, read every 5 s in plant units (degC,
    percent solids, kg/h) with sensor noise and dated timestamps. Returns the
    paths of ``evaporator_export.csv`` and ``evaporator_tags.yaml``: the input
    of the dashboard's Process view and a template for a real export.
    """
    import yaml

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    t_max, r_end = 1200.0, 2e-4

    def steam(t: float) -> float:
        return 355.15 + math.sin(2 * math.pi * t / 400.0)

    plant = FallingFilmEvaporator(T_s=steam, R_f=lambda t: r_end * t / t_max)
    ref = plant.reference(t_max=t_max, nz=201, nt=241)
    rng = np.random.default_rng(seed)
    start = 1772344800.0  # 2026-03-01T06:00:00Z
    lines = ["timestamp,TT205.PV,DT310.PV,FT320.PV"]
    for k, t in enumerate(ref.t):
        stamp = datetime.fromtimestamp(start + t, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        solids = 100.0 * (ref.X.u[-1, k] + 0.002 * rng.standard_normal())
        flow = 3600.0 * (ref.W.u[-1, k] + 0.001 * rng.standard_normal())
        lines.append(f"{stamp},{steam(t) - 273.15:.3f},{solids:.4f},{flow:.3f}")
    export = folder / "evaporator_export.csv"
    export.write_text("\n".join(lines) + "\n")
    tag_map = {
        "inputs": {"T_s": {"tag": "TT205.PV", "offset": 273.15}},
        "sensors": [
            {
                "tag": "DT310.PV",
                "name": "outlet solids",
                "location": 10.0,
                "expression": "X",
                "gain": 0.01,
                "noise": 0.002,
            },
            {
                "tag": "FT320.PV",
                "name": "outlet flow",
                "location": 10.0,
                "expression": "W",
                "gain": 1 / 3600.0,
                "noise": 0.001,
            },
        ],
    }
    tags = folder / "evaporator_tags.yaml"
    tags.write_text(yaml.safe_dump(tag_map, sort_keys=False))
    return export, tags


def monitor_from_files(export: str | Path, tag_map: str | Path, **options: Any) -> FoulingMonitor:
    """Read a historian export and a YAML tag map and run the fouling monitor.

    A ``model`` mapping in the tag map overrides fields of the evaporator.
    """
    import yaml

    spec = yaml.safe_load(Path(tag_map).read_text())
    data = Historian.from_file(export).apply(spec)
    return monitor_from_historian(data, spec.get("model"), **options)


__all__ = [
    "FoulingMonitor",
    "monitor_fouling",
    "monitor_from_files",
    "monitor_from_historian",
    "write_demo_export",
]
