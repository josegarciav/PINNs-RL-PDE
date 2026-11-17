"""A process line: a multi-pass falling-film evaporator feeding a spray dryer (J10).

The twin of a line is more than its units side by side: the evaporator's
concentrate is the dryer's feed, so an error in the fouling estimate of the
evaporator becomes an error in the solids the dryer receives, and from there
in the powder's moisture and its stickiness margin. :class:`ProcessLine`
chains :class:`~pinnrl.process.evaporator.FallingFilmEvaporator` passes in
series (each pass's outlet flow and solids are the next pass's feed) and
hands the last pass's concentrate to a :class:`~pinnrl.process.dryer.SprayDryer`
(or a :class:`~pinnrl.process.counter_dryer.CounterCurrentDryer`), whose
droplets then have the concentrate's solids fraction and whose solids flow is
the concentrate's. The dryer is taken quasi-steady: its residence time is
seconds against the evaporator's hours of fouling.

:meth:`ProcessLine.steady` is the line at one instant, :meth:`ProcessLine.reference`
the dynamic solution over a record (the passes by the method of lines, the
dryer at every output time), :meth:`ProcessLine.records` what the line's
instruments report, and :meth:`ProcessLine.propagate` the linear and
Monte-Carlo propagation of each unit's uncertainty to the powder: which
unit's error the line's product inherits.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from .counter_dryer import CounterCurrentDryer
from .dryer import DryerReference, SprayDryer
from .evaporator import EvaporatorReference, FallingFilmEvaporator

Signal = float | Callable[[float], float]
Dryer = SprayDryer | CounterCurrentDryer


def _as_fn(value: Signal) -> Callable[[float], float]:
    if callable(value):
        return value
    return lambda _t, v=float(value): v  # type: ignore[misc]


def _interp(t: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    return lambda tau: float(np.interp(tau, t, y))


def _shifted(fn: Callable[[float], float], delta: float) -> Callable[[float], float]:
    return lambda tau: fn(tau) + delta


@dataclass
class LineReference:
    """The dynamic solution of a line.

    Attributes:
        t: Output times, s.
        passes: The reference fields of each evaporator pass.
        W: Concentrate flow leaving the last pass, kg/s, at each time.
        X: Concentrate solids fraction at each time.
        powder: Powder moisture (dry basis) from the quasi-steady dryer at each time.
        margin: The stickiness margin of the powder leaving the dryer, K.
        T_air: The dryer's air temperature at the probe heights, ``(n_probes, nt)``, K.
        Y_out: The dryer's outlet humidity ratio at each time.
        probe_heights: The probe heights, m.
        dryer: The dryer's profiles at the last time.
    """

    t: np.ndarray
    passes: list[EvaporatorReference]
    W: np.ndarray
    X: np.ndarray
    powder: np.ndarray
    margin: np.ndarray
    T_air: np.ndarray
    Y_out: np.ndarray
    probe_heights: tuple[float, ...]
    dryer: Any


@dataclass
class ProcessLine:
    """A multi-pass evaporator feeding a spray dryer.

    Args:
        passes: The evaporator tubes in series. The feed (``W_in``, ``X_in``)
            of the first is the line's; those of the others are overwritten
            by the previous pass's outlet.
        dryer: The tower. Its droplet's ``solids_fraction`` and its
            ``solids_flow`` are set from the concentrate at every evaluation;
            the droplet's diameter and temperature are the atomiser's.
    """

    passes: Sequence[FallingFilmEvaporator]
    dryer: Dryer = field(default_factory=SprayDryer)

    @classmethod
    def default(
        cls,
        n_passes: int = 3,
        fouling: Sequence[Signal] | None = None,
        steam: Signal | Sequence[Signal] = 353.15,
        dryer: Dryer | None = None,
        **tube: Any,
    ) -> ProcessLine:
        """A dairy line: three standard tubes in series and the pilot tower.

        The feed of 0.1 kg/s at 10 % solids, with steam 10 K above the
        solvent's saturation temperature in every pass, leaves the third pass
        at 41 % solids and 0.024 kg/s (the evaporation accelerates from pass
        to pass as the film thins) and is atomised into the co-current tower,
        which dries it to a powder moisture of 0.077. ``fouling`` gives each
        pass its ``R_f`` (a number or a function of time); ``steam`` one
        temperature for all passes or one per pass; ``tube`` overrides the
        tubes' other attributes.
        """
        steams = list(steam) if isinstance(steam, Sequence) else [steam] * n_passes
        foulings = list(fouling) if fouling is not None else [0.0] * n_passes
        if len(steams) != n_passes or len(foulings) != n_passes:
            raise ValueError("steam and fouling need one entry per pass")
        passes = [
            FallingFilmEvaporator(T_s=steams[k], R_f=foulings[k], **tube) for k in range(n_passes)
        ]
        return cls(passes=passes, dryer=dryer if dryer is not None else SprayDryer())

    # ----------------------------------------------------------------- coupling
    def dryer_for(self, W: float, X: float) -> Dryer:
        """The dryer fed with a concentrate of flow ``W`` and solids fraction ``X``."""
        X = float(min(max(X, 1e-3), 0.95))
        droplet = replace(self.dryer.droplet, solids_fraction=X)
        return replace(self.dryer, droplet=droplet, solids_flow=float(W) * X)

    def _dryer_outlet(
        self, dryer: Dryer, probes: Sequence[float] = ()
    ) -> tuple[float, float, Any, np.ndarray, float]:
        """``(powder moisture, margin, profiles, air temperatures at the probes, outlet humidity)`` of a steady dryer."""
        z = np.asarray(probes, dtype=float)
        if isinstance(dryer, CounterCurrentDryer):
            prof = dryer.steady()
            T_air = np.interp(z, prof.z, prof.T_a)
            return (
                float(prof.X[-1]),
                float(dryer.stickiness_margin(prof)[-1]),
                prof,
                T_air,
                float(prof.Y[0]),
            )
        ref: DryerReference = dryer.reference()
        T_air = np.interp(z, ref.z, ref.T_a)
        return (
            float(ref.X[-1]),
            float(dryer.stickiness_margin(ref)[-1]),
            ref,
            T_air,
            float(ref.Y[-1]),
        )

    def steady(self, t: float = 0.0, nz: int = 201) -> dict[str, Any]:
        """The line at the inputs' values of time ``t``.

        Returns ``{"W": [feed, after pass 1, ...], "X": [...], "powder": the
        powder's moisture, "margin": its stickiness margin (K), "dryer": the
        dryer's profiles}``.
        """
        W = [float(_as_fn(self.passes[0].W_in)(t))]
        X = [float(_as_fn(self.passes[0].X_in)(t))]
        for k, tube in enumerate(self.passes):
            unit = tube if k == 0 else replace(tube, W_in=W[-1], X_in=X[-1])
            ss = unit.steady_state(nz, t=t)
            W.append(float(ss["W"][-1]))
            X.append(float(ss["X"][-1]))
        powder, margin, prof, _T, _Y = self._dryer_outlet(self.dryer_for(W[-1], X[-1]))
        return {"W": W, "X": X, "powder": powder, "margin": margin, "dryer": prof}

    def reference(
        self,
        t_max: float,
        nz: int = 51,
        nt: int | None = None,
        method: str = "auto",
        dryer_every: int = 1,
        probe_heights: Sequence[float] = (1.0, 2.0, 3.5, 5.0, 6.5, 8.0),
    ) -> LineReference:
        """The dynamic solution: the passes by the method of lines, the dryer quasi-steady.

        Args:
            t_max: Final time, s.
            nz: Axial points of each pass.
            nt: Output times (default one every 5 s, the spacing the compiled
                integrator samples its inputs at).
            method: ``"auto"`` (the compiled integrator of I16 where it builds,
                SciPy's BDF otherwise), or any method ``FallingFilmEvaporator.reference`` takes.
            dryer_every: Evaluate the dryer at every so-many output times (the
                others are interpolated), to save its 5 ms where the record is long.
            probe_heights: Heights of the dryer's air thermocouples, m.
        """
        nt = int(round(t_max / 5.0)) + 1 if nt is None else nt
        if method == "auto":
            from pinnrl.native import evaporator_available

            method = "native" if evaporator_available() else "BDF"
        refs: list[EvaporatorReference] = []
        W_in: Signal = self.passes[0].W_in
        X_in: Signal = self.passes[0].X_in
        for tube in self.passes:
            unit = replace(tube, W_in=W_in, X_in=X_in)
            ref = unit.reference(t_max, nz=nz, nt=nt, method=method)
            refs.append(ref)
            W_in, X_in = _interp(ref.t, ref.W.u[-1]), _interp(ref.t, ref.X.u[-1])
        t = refs[-1].t
        W, X = refs[-1].W.u[-1], refs[-1].X.u[-1]
        idx = np.unique(np.concatenate([np.arange(0, t.size, max(dryer_every, 1)), [t.size - 1]]))
        powder_k, margin_k, T_k, Y_k, last = [], [], [], [], None
        for k in idx:
            p, m, last, T_air, Y_out = self._dryer_outlet(self.dryer_for(W[k], X[k]), probe_heights)
            powder_k.append(p)
            margin_k.append(m)
            T_k.append(T_air)
            Y_k.append(Y_out)
        powder = np.interp(t, t[idx], powder_k)
        margin = np.interp(t, t[idx], margin_k)
        T_arr = np.asarray(T_k)  # (len(idx), n_probes)
        T_air_t = np.stack([np.interp(t, t[idx], T_arr[:, j]) for j in range(T_arr.shape[1])])
        Y_out_t = np.interp(t, t[idx], Y_k)
        return LineReference(
            t=t,
            passes=refs,
            W=W,
            X=X,
            powder=powder,
            margin=margin,
            T_air=T_air_t,
            Y_out=Y_out_t,
            probe_heights=tuple(float(z) for z in probe_heights),
            dryer=last,
        )

    # ------------------------------------------------------------------ records
    def records(self, ref: LineReference) -> dict[str, np.ndarray]:
        """What the line's instruments report over a reference run, without noise.

        ``{"t", "X_pass": (n_passes, nt) outlet solids of each pass, "W_pass":
        their outlet flows, "T_air": the dryer's air temperatures at the probe
        heights ``(n_probes, nt)``, "Y_out": its outlet humidity, "X_powder":
        the powder's moisture at every time}``. A plant samples the powder in
        the lab; the experiment chooses when.
        """
        return {
            "t": ref.t,
            "X_pass": np.stack([r.X.u[-1] for r in ref.passes]),
            "W_pass": np.stack([r.W.u[-1] for r in ref.passes]),
            "T_air": ref.T_air,
            "Y_out": ref.Y_out,
            "X_powder": ref.powder,
        }

    # -------------------------------------------------------------- propagation
    def _with(self, theta: Mapping[str, float], t: float) -> ProcessLine:
        """The line with the perturbations ``theta`` applied: ``R_f_k`` per pass, dryer attributes, droplet attributes, feed."""
        passes = list(self.passes)
        for k, tube in enumerate(passes):
            if f"R_f_{k}" in theta:
                passes[k] = replace(tube, R_f=_shifted(_as_fn(tube.R_f), float(theta[f"R_f_{k}"])))
        if "W_feed" in theta:
            passes[0] = replace(passes[0], W_in=float(_as_fn(passes[0].W_in)(t)) + theta["W_feed"])
        if "X_feed" in theta:
            passes[0] = replace(passes[0], X_in=float(_as_fn(passes[0].X_in)(t)) + theta["X_feed"])
        dryer = self.dryer
        droplet = dryer.droplet
        for name, value in theta.items():
            if name.startswith("R_f_") or name in ("W_feed", "X_feed"):
                continue
            if hasattr(droplet, name):
                droplet = replace(droplet, **{name: getattr(droplet, name) + value})
            elif hasattr(dryer, name):
                dryer = replace(dryer, **{name: getattr(dryer, name) + value})
            else:
                raise ValueError(
                    f"{name!r} is neither a pass fouling, a feed, a dryer nor a droplet attribute"
                )
        return replace(self, passes=passes, dryer=replace(dryer, droplet=droplet))

    def propagate(
        self,
        std: Mapping[str, float],
        t: float = 0.0,
        samples: int = 0,
        seed: int = 0,
        steps: Mapping[str, float] | None = None,
    ) -> dict[str, Any]:
        """Propagate independent uncertainties of the units to the concentrate and the powder.

        ``std`` maps each uncertain quantity to its standard deviation:
        ``"R_f_k"`` the fouling of pass ``k`` (m²K/W), ``"W_feed"`` and
        ``"X_feed"`` the line's feed, and any attribute of the dryer
        (``"U_wall"``) or of its droplet (``"X_cr"``, ``"n"``, ``"d0"``), as
        absolute deviations. The linear propagation uses central differences
        of the line's steady map at time ``t``; with ``samples > 0`` a Monte
        Carlo run through the same map checks it.

        Returns ``{"targets": {"X_concentrate", "powder", "margin"}, "linear":
        {target: {"std", "shares": {source: share of the variance}}},
        "sensitivity": {target: {source: d target / d source}}, "monte_carlo":
        {target: {"std", "mean_shift"}} (when sampled)}``.
        """
        names = list(std)
        base = self.steady(t=t)
        targets = ("X_concentrate", "powder", "margin")

        def outputs(theta: Mapping[str, float]) -> np.ndarray:
            s = self._with(theta, t).steady(t=t)
            return np.array([s["X"][-1], s["powder"], s["margin"]])

        y0 = outputs({})
        sens: dict[str, dict[str, float]] = {k: {} for k in targets}
        contributions = np.zeros((len(targets), len(names)))
        for j, name in enumerate(names):
            h = float((steps or {}).get(name, 0.5 * std[name]))
            if h == 0.0:
                continue
            grad = (outputs({name: h}) - outputs({name: -h})) / (2 * h)
            for i, target in enumerate(targets):
                sens[target][name] = float(grad[i])
                contributions[i, j] = grad[i] * std[name]
        var = np.sum(contributions**2, axis=1)
        linear = {
            target: {
                "std": float(math.sqrt(var[i])),
                "shares": {
                    name: float(contributions[i, j] ** 2 / var[i]) if var[i] > 0 else 0.0
                    for j, name in enumerate(names)
                },
            }
            for i, target in enumerate(targets)
        }
        out: dict[str, Any] = {
            "targets": dict(zip(targets, y0.tolist(), strict=True)),
            "linear": linear,
            "sensitivity": sens,
            "base": {"W": base["W"], "X": base["X"]},
        }
        if samples > 0:
            rng = np.random.default_rng(seed)
            draws = np.stack(
                [
                    outputs({n: float(std[n] * rng.standard_normal()) for n in names})
                    for _ in range(samples)
                ]
            )
            out["monte_carlo"] = {
                target: {
                    "std": float(draws[:, i].std(ddof=1)),
                    "mean_shift": float(draws[:, i].mean() - y0[i]),
                }
                for i, target in enumerate(targets)
            }
        return out
