"""Measured inputs: a time series that drives a boundary condition or a forcing term.

A plant model is driven by what was measured: the steam temperature at the
inlet, the feed rate, the ambient conditions. :class:`TimeSeries` wraps such a
record, a table of times and values from a CSV or NPZ file or from arrays, and
evaluates it at any batch of times by linear interpolation, holding the end
values outside the record. Inside a :class:`~pinnrl.pdes.custom.CustomPDE` an
input is named like a parameter (``inputs={"q": TimeSeries(...)}``) and can
appear in the equations (``"u_t - alpha*u_xx - q"``) and in boundary values
(``dirichlet: {left: "q", right: 0.0}``), where it is read at the time of each
collocation or boundary point.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch


class TimeSeries:
    """A sampled signal ``value(t)``, interpolated linearly and held constant outside its range.

    Args:
        t: Sample times, increasing (duplicates are not allowed).
        values: Sample values, same length as ``t``.
        name: A label used in error messages and specs.
        path: Where the data came from, kept so :meth:`to_spec` can refer to it.
        time_column: Column name for ``t`` when the data came from a file.
        value_column: Column name for the values when the data came from a file.

    Raises:
        ValueError: If the lengths differ, there are fewer than two samples, or
            the times are not strictly increasing.
    """

    def __init__(
        self,
        t: Sequence[float] | np.ndarray,
        values: Sequence[float] | np.ndarray,
        *,
        name: str = "input",
        path: str | None = None,
        time_column: str | None = None,
        value_column: str | None = None,
    ):
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
        v_arr = np.asarray(values, dtype=np.float64).reshape(-1)
        if t_arr.shape != v_arr.shape:
            raise ValueError(f"{name}: {t_arr.size} times but {v_arr.size} values")
        if t_arr.size < 2:
            raise ValueError(f"{name}: a time series needs at least two samples")
        if not np.all(np.diff(t_arr) > 0):
            raise ValueError(f"{name}: sample times must be strictly increasing")
        self.name = name
        self.t = t_arr
        self.values = v_arr
        self.path = path
        self.time_column = time_column
        self.value_column = value_column
        self._t_torch = torch.as_tensor(self.t)
        self._v_torch = torch.as_tensor(self.values)

    @property
    def span(self) -> tuple[float, float]:
        """First and last sample time."""
        return float(self.t[0]), float(self.t[-1])

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        time_column: str = "t",
        value_column: str | None = None,
        name: str | None = None,
    ) -> TimeSeries:
        """Read a ``.csv`` (header row) or ``.npz`` file.

        ``value_column`` defaults to the first column that is not the time
        column. Rows that do not parse as numbers are skipped.

        Raises:
            ValueError: If a column is missing or the extension is not supported.
        """
        path = Path(path)
        if path.suffix.lower() == ".npz":
            data = np.load(path)
            columns = {k: np.asarray(data[k]).reshape(-1) for k in data.files}
        elif path.suffix.lower() in (".csv", ".txt"):
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = [r for r in reader]
            if not rows:
                raise ValueError(f"{path}: empty file")
            columns = {}
            for key in rows[0]:
                vals = []
                for r in rows:
                    try:
                        vals.append(float(r[key]))
                    except (TypeError, ValueError):
                        vals.append(np.nan)
                columns[key.strip()] = np.asarray(vals)
        else:
            raise ValueError(f"{path}: use a .csv or .npz file")
        if time_column not in columns:
            raise ValueError(f"{path}: no column {time_column!r}; columns are {sorted(columns)}")
        if value_column is None:
            others = [k for k in columns if k != time_column]
            if not others:
                raise ValueError(f"{path}: no value column besides {time_column!r}")
            value_column = others[0]
        if value_column not in columns:
            raise ValueError(f"{path}: no column {value_column!r}; columns are {sorted(columns)}")
        t, v = columns[time_column], columns[value_column]
        keep = ~(np.isnan(t) | np.isnan(v))
        order = np.argsort(t[keep], kind="stable")
        return cls(
            t[keep][order],
            v[keep][order],
            name=name or value_column,
            path=str(path),
            time_column=time_column,
            value_column=value_column,
        )

    @classmethod
    def from_spec(cls, name: str, spec: Any, base_dir: str | Path | None = None) -> TimeSeries:
        """Build from a :class:`TimeSeries`, a file spec or inline samples.

        ``spec`` is a :class:`TimeSeries`, ``{"path", "time_column", "value_column"}``
        or ``{"t": [...], "values": [...]}``.

        Raises:
            ValueError: For a spec of another shape.
        """
        if isinstance(spec, TimeSeries):
            return spec
        if isinstance(spec, Mapping):
            if "path" in spec:
                path = Path(spec["path"])
                if base_dir is not None and not path.is_absolute():
                    path = Path(base_dir) / path
                return cls.from_file(
                    path,
                    time_column=spec.get("time_column", "t"),
                    value_column=spec.get("value_column"),
                    name=name,
                )
            if "t" in spec and "values" in spec:
                return cls(spec["t"], spec["values"], name=name)
        raise ValueError(
            f"input {name!r}: give a TimeSeries, {{'path': ..., 'time_column': ..., "
            f"'value_column': ...}} or {{'t': [...], 'values': [...]}}"
        )

    def to_spec(self) -> dict[str, Any]:
        """The dict that rebuilds this series: by file when it came from one, else the samples."""
        if self.path is not None:
            spec: dict[str, Any] = {"path": self.path}
            if self.time_column is not None:
                spec["time_column"] = self.time_column
            if self.value_column is not None:
                spec["value_column"] = self.value_column
            return spec
        return {"t": self.t.tolist(), "values": self.values.tolist()}

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """The signal at ``t`` (any shape), with ``t``'s dtype and device; no gradient flows."""
        tt = self._t_torch.to(device=t.device, dtype=torch.float64)
        vv = self._v_torch.to(device=t.device, dtype=torch.float64)
        q = t.detach().to(torch.float64).reshape(-1)
        idx = torch.searchsorted(tt, q, right=True).clamp(1, tt.numel() - 1)
        t0, t1 = tt[idx - 1], tt[idx]
        v0, v1 = vv[idx - 1], vv[idx]
        w = ((q - t0) / (t1 - t0)).clamp(0.0, 1.0)
        out = v0 + w * (v1 - v0)
        return out.reshape(t.shape).to(dtype=t.dtype)

    def numpy(self, t: np.ndarray) -> np.ndarray:
        """The signal at NumPy times, held constant outside the record."""
        return np.interp(np.asarray(t, dtype=float), self.t, self.values)

    def __repr__(self) -> str:
        lo, hi = self.span
        return f"TimeSeries({self.name!r}, {self.t.size} samples on [{lo:g}, {hi:g}])"
