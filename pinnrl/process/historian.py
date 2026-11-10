"""Plant-historian exports as model inputs and sensor records.

A historian export is a table of tag readings against time: one column per
tag (*wide*), or one row per reading with the tag's name in a column
(*long*), in plant units, with irregular timestamps, gaps and bad-quality
readings. :class:`Historian` reads such a file (CSV, or Parquet when
``pyarrow`` is installed), and a *tag map* says which tag is which signal of
the model: the measured inputs become
:class:`~pinnrl.pdes.inputs.TimeSeries` and the measurements become sensors
(:mod:`pinnrl.pdes.sensors`) with their records attached, in the model's
units and on its time axis. The same records, on a common time grid, feed the
classical estimators (:meth:`TwinData.records`).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pinnrl.pdes.inputs import TimeSeries
from pinnrl.pdes.sensors import Sensor

if TYPE_CHECKING:
    import torch

_TIME_UNITS = {"s": 1.0, "ms": 1e-3, "min": 60.0, "h": 3600.0, "d": 86400.0}


def _parse_time(text: Any, time_format: str | None) -> tuple[float, bool]:
    """A timestamp as ``(seconds, is_date)``: a number as it stands, a date as POSIX seconds.

    Dates without a zone are read as UTC; a cell that is neither is NaN.
    """
    if isinstance(text, datetime):
        stamp = text
    else:
        if isinstance(text, (int, float, np.integer, np.floating)):
            return float(text), False
        text = str(text).strip()
        try:
            return float(text), False
        except ValueError:
            pass
        try:
            if time_format is not None:
                stamp = datetime.strptime(text, time_format)
            else:
                stamp = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return math.nan, False
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.timestamp(), True


def _number(text: Any) -> float:
    """A reading as a float; anything that is not a number (``Bad``, an empty cell) is NaN."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return math.nan


def _read_table(path: Path) -> tuple[list[str], list[list[Any]]]:
    """Column names and columns of a CSV or Parquet file."""
    suffix = path.suffix.lower()
    if suffix in (".parquet", ".pq"):
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError(
                f"{path}: reading Parquet needs pyarrow (pip install pyarrow)"
            ) from exc
        table = pq.read_table(path)
        return list(table.column_names), [c.to_pylist() for c in table.columns]
    if suffix in (".csv", ".txt"):
        with open(path, newline="") as f:
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
            except csv.Error:
                dialect = csv.excel
            rows = [r for r in csv.reader(f, dialect) if r and any(c.strip() for c in r)]
        if len(rows) < 2:
            raise ValueError(f"{path}: no data rows")
        names = [c.strip() for c in rows[0]]
        width = len(names)
        columns = [[r[i] if i < len(r) else "" for r in rows[1:]] for i in range(width)]
        return names, columns
    raise ValueError(f"{path}: use a .csv or .parquet file")


@dataclass
class TwinData:
    """A historian window in the model's terms.

    Attributes:
        inputs: Measured inputs by model name, for ``CustomPDE(inputs=...)`` or,
            through ``series.numpy``, as the signals of a process model.
        sensors: Sensors with their records attached, for ``CustomPDE(sensors=...)``.
        t_max: Length of the window, s (the end of the model's time domain).
        t_start: The window's start on the historian's clock, s.
    """

    inputs: dict[str, TimeSeries] = field(default_factory=dict)
    sensors: list[Sensor] = field(default_factory=list)
    t_max: float = 0.0
    t_start: float = 0.0

    def records(self, names: Sequence[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        """The sensors' records on their common timestamps: ``(times (K,), values (K, m))``.

        The form the filters take (``ExtendedKalmanFilter.run(times, records, ...)``).
        ``names`` selects and orders the sensors; the default is all of them.

        Raises:
            ValueError: If a name is unknown or the sensors do not share their
                timestamps (give them the same ``every`` in the tag map).
        """
        by_name = {s.name: s for s in self.sensors}
        chosen = list(by_name) if names is None else list(names)
        missing = [n for n in chosen if n not in by_name]
        if missing:
            raise ValueError(f"no sensor named {missing}; sensors are {sorted(by_name)}")
        if not chosen:
            raise ValueError("the tag map defines no sensors")
        empty = [n for n in chosen if by_name[n].values is None]
        if empty:
            raise ValueError(f"sensors {empty} carry no records")
        times = by_name[chosen[0]].times
        for n in chosen[1:]:
            other = by_name[n].times
            if other.shape != times.shape or not np.allclose(other, times):
                raise ValueError(
                    f"sensors {chosen[0]!r} and {n!r} have different timestamps; "
                    "resample them to a common grid with the same 'every'"
                )
        values = [
            cast("torch.Tensor", by_name[n].values).detach().cpu().numpy().ravel() for n in chosen
        ]
        return times.copy(), np.stack(values, axis=1)


class Historian:
    """Tag readings from a plant historian, on a clock in seconds.

    Args:
        tags: ``{tag: (times, values)}`` with the times in seconds. Readings
            that are not finite are dropped, the rest sorted in time, and
            readings sharing a timestamp averaged.
        origin: The clock's zero on the source's own time axis (POSIX seconds
            for dated exports), kept for reference.
    """

    def __init__(
        self,
        tags: Mapping[str, tuple[Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]],
        origin: float = 0.0,
    ):
        self.origin = float(origin)
        self._tags: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, (t, v) in tags.items():
            t = np.asarray(t, dtype=float).reshape(-1)
            v = np.asarray(v, dtype=float).reshape(-1)
            if t.shape != v.shape:
                raise ValueError(f"tag {name!r}: {t.size} times but {v.size} values")
            keep = np.isfinite(t) & np.isfinite(v)
            t, v = t[keep], v[keep]
            order = np.argsort(t, kind="stable")
            t, v = t[order], v[order]
            if t.size and np.any(np.diff(t) == 0):  # repeated stamps: one reading, the mean
                unique, inverse = np.unique(t, return_inverse=True)
                v = np.bincount(inverse, weights=v) / np.bincount(inverse)
                t = unique
            self._tags[str(name)] = (t, v)

    # ----------------------------------------------------------------- reading
    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        time_column: str = "timestamp",
        layout: str | None = None,
        tag_column: str = "tag",
        value_column: str = "value",
        quality_column: str | None = None,
        good: Sequence[str] = ("good", "192", "0", "ok", "true", "1"),
        time_unit: str = "s",
        time_format: str | None = None,
        origin: float | str | None = None,
    ) -> Historian:
        """Read a wide or long export from a ``.csv`` or ``.parquet`` file.

        Args:
            path: The file. CSV delimiters ``,``, ``;`` and tab are detected.
            time_column: Name of the timestamp column: numbers (in
                ``time_unit``) or dates (ISO 8601, or ``time_format`` for
                ``datetime.strptime``; dates without a zone are read as UTC).
            layout: ``"wide"`` (a column per tag), ``"long"`` (a row per
                reading) or ``None`` to detect: long when ``tag_column`` and
                ``value_column`` are both present.
            tag_column: Long layout: the column holding the tag names.
            value_column: Long layout: the column holding the readings.
            quality_column: Long layout: a quality flag per reading; rows whose
                flag is not in ``good`` are dropped.
            good: Quality flags that count as good (compared in lower case).
            time_unit: Unit of numeric timestamps: ``s``, ``ms``, ``min``, ``h`` or ``d``.
            time_format: ``strptime`` format of dated timestamps that are not ISO 8601.
            origin: The clock's zero: a timestamp in the file's own terms. The
                default is the first timestamp of the file.

        Raises:
            ValueError: If a named column is missing or the layout is unknown.
        """
        path = Path(path)
        if time_unit not in _TIME_UNITS:
            raise ValueError(f"time_unit {time_unit!r}: use one of {sorted(_TIME_UNITS)}")
        names, columns = _read_table(path)
        table = dict(zip(names, columns, strict=True))
        if time_column not in table:
            raise ValueError(f"{path}: no column {time_column!r}; columns are {names}")
        parsed = [_parse_time(cell, time_format) for cell in table[time_column]]
        seconds = np.array([p[0] for p in parsed])
        dated = any(p[1] for p in parsed)
        unit = 1.0 if dated else _TIME_UNITS[time_unit]  # dates are already in seconds
        seconds = seconds * unit
        if not np.isfinite(seconds).any():
            raise ValueError(f"{path}: no timestamp in column {time_column!r} could be read")
        if origin is None:
            zero = float(np.nanmin(seconds))
        else:
            zero = _parse_time(origin, time_format)[0] * unit
        seconds = seconds - zero
        if layout is None:
            layout = "long" if tag_column in table and value_column in table else "wide"
        tags: dict[str, tuple[list[float], list[float]]] = {}
        if layout == "long":
            for needed in (tag_column, value_column):
                if needed not in table:
                    raise ValueError(f"{path}: no column {needed!r}; columns are {names}")
            quality = table.get(quality_column) if quality_column else None
            if quality_column and quality is None:
                raise ValueError(f"{path}: no column {quality_column!r}; columns are {names}")
            accepted = {g.lower() for g in good}
            for i, (tag, value) in enumerate(
                zip(table[tag_column], table[value_column], strict=True)
            ):
                if quality is not None and str(quality[i]).strip().lower() not in accepted:
                    continue
                t_list, v_list = tags.setdefault(str(tag).strip(), ([], []))
                t_list.append(seconds[i])
                v_list.append(_number(value))
        elif layout == "wide":
            for name in names:
                if name == time_column:
                    continue
                tags[name] = (list(seconds), [_number(c) for c in table[name]])
        else:
            raise ValueError(f"layout {layout!r}: use 'wide' or 'long'")
        return cls(tags, origin=zero)

    # ----------------------------------------------------------------- access
    @property
    def tags(self) -> list[str]:
        """The tag names."""
        return list(self._tags)

    def readings(self, tag: str) -> tuple[np.ndarray, np.ndarray]:
        """``(times, values)`` of one tag: the good readings, in seconds from the origin."""
        if tag not in self._tags:
            raise KeyError(f"no tag {tag!r}; tags are {self.tags}")
        t, v = self._tags[tag]
        return t.copy(), v.copy()

    def span(self, tag: str | None = None) -> tuple[float, float]:
        """First and last reading of a tag, or the interval every tag covers."""
        if tag is not None:
            t, _ = self.readings(tag)
            return float(t[0]), float(t[-1])
        spans = [(t[0], t[-1]) for t, _ in self._tags.values() if t.size]
        return float(max(s[0] for s in spans)), float(min(s[1] for s in spans))

    def resample(
        self, tag: str, times: Sequence[float] | np.ndarray, aggregate: str = "interp"
    ) -> np.ndarray:
        """The tag at ``times``.

        ``"interp"`` interpolates linearly between readings (held at the ends);
        ``"mean"`` averages the readings in the interval ending at each time
        (its length the spacing to the previous time), which is what a slower
        record of a fast, noisy tag should be, and falls back to interpolation
        where an interval holds no reading; ``"last"`` takes the latest reading
        at or before each time (a sample-and-hold signal, a lab value).
        """
        t, v = self.readings(tag)
        if t.size == 0:
            raise ValueError(f"tag {tag!r} has no good readings")
        times = np.asarray(times, dtype=float).reshape(-1)
        interp = np.interp(times, t, v)
        if aggregate == "interp":
            return interp
        if aggregate == "last":
            idx = np.searchsorted(t, times, side="right") - 1
            return np.where(idx >= 0, v[np.clip(idx, 0, None)], v[0])
        if aggregate == "mean":
            steps = np.diff(times)
            first = steps[0] if steps.size else 0.0
            starts = times - np.concatenate([[first], steps])
            lo = np.searchsorted(t, starts, side="right")
            hi = np.searchsorted(t, times, side="right")
            csum = np.concatenate([[0.0], np.cumsum(v)])
            count = hi - lo
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = (csum[hi] - csum[lo]) / count
            return np.where(count > 0, mean, interp)
        raise ValueError(f"aggregate {aggregate!r}: use 'interp', 'mean' or 'last'")

    def series(
        self,
        tag: str,
        *,
        gain: float = 1.0,
        offset: float = 0.0,
        start: float | None = None,
        end: float | None = None,
        every: float | None = None,
        aggregate: str = "interp",
        name: str | None = None,
    ) -> TimeSeries:
        """One tag as a :class:`~pinnrl.pdes.inputs.TimeSeries` in model units, ``gain * reading + offset``.

        ``start`` and ``end`` cut a window and shift its start to ``t = 0``;
        ``every`` resamples to a regular grid (see :meth:`resample`), otherwise
        the readings inside the window are kept, with the interpolated values
        at its two ends.
        """
        lo, hi = self.span(tag)
        t0 = lo if start is None else float(start)
        t1 = hi if end is None else float(end)
        if t1 <= t0:
            raise ValueError(f"tag {tag!r}: empty window [{t0}, {t1}]")
        if every is not None:
            times = np.arange(t0, t1 + 1e-9 * max(1.0, abs(t1)), float(every))
        else:
            t, _ = self.readings(tag)
            inside = t[(t > t0) & (t < t1)]
            times = np.concatenate([[t0], inside, [t1]])
        values = self.resample(tag, times, aggregate if every is not None else "interp")
        return TimeSeries(times - t0, gain * values + offset, name=name or tag)

    # ----------------------------------------------------------------- the tag map
    def apply(self, tag_map: Mapping[str, Any] | str | Path) -> TwinData:
        """Turn the readings into a model's inputs and sensors with a tag map.

        The map is a dict or a YAML file::

            window: [600, 1800]          # optional, s on the historian's clock; shifted to t = 0
            inputs:
              T_s:  {tag: TT205.PV, offset: 273.15}        # degC -> K
              W_in: {tag: FT101.PV, gain: 2.7778e-4}       # kg/h -> kg/s
            sensors:
              - {tag: DT310.PV, name: outlet X, kind: point, location: 10.0,
                 expression: X, gain: 0.01, noise: 0.002, every: 5.0, aggregate: mean}

        An input entry takes ``tag`` and optionally ``gain``, ``offset``,
        ``every`` and ``aggregate``. A sensor entry takes ``tag``, the same
        conversion keys, and the keys of the sensor it describes (``kind``:
        ``point``, ``boundary`` or ``integral``; ``expression``, ``location``
        or ``side``, ``noise``, ``delay``, ``weight``, ``scale``, ``name``);
        its timestamps are the tag's readings in the window, or a regular grid
        with ``every``, which starts one period after the window's start.
        Without ``window`` the interval covered by every mapped tag is used.

        Raises:
            ValueError: If the map names a tag the export does not have.
        """
        if not isinstance(tag_map, Mapping):
            import yaml

            tag_map = cast("Mapping[str, Any]", yaml.safe_load(Path(tag_map).read_text()))
        inputs_map = dict(tag_map.get("inputs") or {})
        sensors_map = list(tag_map.get("sensors") or [])
        used = [spec["tag"] for spec in inputs_map.values()] + [s["tag"] for s in sensors_map]
        unknown = sorted({t for t in used if t not in self._tags})
        if unknown:
            raise ValueError(f"the tag map names tags the export does not have: {unknown}")
        if not used:
            raise ValueError("the tag map has neither inputs nor sensors")
        if tag_map.get("window") is not None:
            t0, t1 = (float(v) for v in tag_map["window"])
        else:
            spans = [self.span(t) for t in used]
            t0, t1 = max(s[0] for s in spans), min(s[1] for s in spans)
        if t1 <= t0:
            raise ValueError(f"the mapped tags share no interval (window [{t0}, {t1}])")
        inputs = {
            name: self.series(
                spec["tag"],
                gain=float(spec.get("gain", 1.0)),
                offset=float(spec.get("offset", 0.0)),
                start=t0,
                end=t1,
                every=spec.get("every"),
                aggregate=spec.get("aggregate", "interp"),
                name=name,
            )
            for name, spec in inputs_map.items()
        }
        sensors = []
        for spec in sensors_map:
            spec = dict(spec)
            tag = spec.pop("tag")
            gain, offset = float(spec.pop("gain", 1.0)), float(spec.pop("offset", 0.0))
            every, aggregate = spec.pop("every", None), spec.pop("aggregate", "interp")
            spec.setdefault("kind", "point")
            spec.setdefault("name", tag)
            if every is not None:
                times = np.arange(t0 + float(every), t1 + 1e-9 * max(1.0, abs(t1)), float(every))
                values = self.resample(tag, times, aggregate)
            else:
                t, v = self.readings(tag)
                keep = (t >= t0) & (t <= t1)
                times, values = t[keep], v[keep]
            if times.size == 0:
                raise ValueError(
                    f"sensor {spec['name']!r}: tag {tag!r} has no readings in the window"
                )
            sensor = Sensor.from_spec({**spec, "times": (times - t0).tolist()})
            sensor.set_values(gain * values + offset)
            sensors.append(sensor)
        return TwinData(inputs=inputs, sensors=sensors, t_max=t1 - t0, t_start=t0)


__all__ = ["Historian", "TwinData"]
