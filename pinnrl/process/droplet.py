"""A single drying droplet (J07): moisture and temperature of a droplet in hot air.

The droplet (solids plus water, ideal mixing of volumes) exchanges heat and
water vapour with the air through Ranz–Marshall transfer coefficients. The
drying rate is the wet-surface rate times a **drying-rate closure** ``f(X)``
of the dry-basis moisture ``X``, the characteristic drying curve of the
material: 1 above the critical moisture, falling to 0 at the equilibrium
moisture. That closure is what a plant cannot measure and the twin is to
identify (``custom_pde(unknown_rate=...)``).

Units are SI; moisture is dry basis (kg water per kg solids).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from pinnrl.benchmarks.reference import ReferenceSolution

Signal = float | Callable[[float], float]
M_W = 0.018015  # kg/mol
R_GAS = 8.314


def _as_fn(value: Signal) -> Callable[[float], float]:
    if callable(value):
        return value
    v = float(value)
    return lambda t: v


def saturation_pressure(T: np.ndarray) -> np.ndarray:
    """Water vapour saturation pressure, Pa (Magnus form, ``T`` in K)."""
    Tc = np.asarray(T, dtype=float) - 273.15
    return 610.94 * np.exp(17.625 * Tc / (Tc + 243.04))


@dataclass
class DropletReference:
    """The reference trajectory of one droplet: ``X`` (dry-basis moisture), ``T`` (K), ``m`` (kg), ``d`` (m)."""

    t: np.ndarray
    X: np.ndarray
    T: np.ndarray
    m: np.ndarray
    d: np.ndarray

    def as_reference_solutions(self) -> dict[str, ReferenceSolution]:
        """``{"X", "T"}`` as :class:`ReferenceSolution` on the dummy coordinate (for ``CustomPDE``)."""
        x = np.array([0.0, 1.0])
        return {
            "X": ReferenceSolution(x=x, t=self.t, u=np.tile(self.X, (2, 1))),
            "T": ReferenceSolution(x=x, t=self.t, u=np.tile(self.T, (2, 1))),
        }


@dataclass
class DryingDroplet:
    """A droplet of a solids solution drying in hot air (a single-droplet rig, or one dryer trajectory).

    Defaults: a 1.5 mm droplet of a 20 % solids solution (skim-milk-like) at
    25 °C, in air at 90 °C with a humidity ratio of 0.01 and a relative
    velocity of 1 m/s. The characteristic drying curve is
    ``f(X) = min(1, ((X - X_eq) / (X_cr - X_eq))^n)``.
    """

    d0: float = 1.5e-3  # m, initial diameter
    solids_fraction: float = 0.2  # initial mass fraction of solids
    T0: float = 298.15  # K
    rho_s: float = 1400.0  # kg/m³, solids
    rho_w: float = 1000.0  # kg/m³, water
    c_s: float = 1500.0  # J/kg/K, solids
    c_w: float = 4180.0  # J/kg/K, water
    latent_heat: float = 2.35e6  # J/kg
    T_air: Signal = 363.15  # K
    Y_air: float = 0.01  # kg water / kg dry air
    velocity: float = 1.0  # m/s, relative air velocity
    rho_air: float = 0.97  # kg/m³
    mu_air: float = 2.15e-5  # Pa s
    k_air: float = 0.031  # W/m/K
    cp_air: float = 1010.0  # J/kg/K
    D_v: float = 3.3e-5  # m²/s, vapour diffusivity in air
    X_eq: float = 0.05  # equilibrium moisture (dry basis)
    X_cr: float = 1.0  # critical moisture (dry basis)
    n: float = 1.5  # exponent of the falling-rate curve
    rate_closure: Callable[[np.ndarray], np.ndarray] | None = None  # overrides the CDC form
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- geometry
    @property
    def m0(self) -> float:
        """Initial droplet mass, kg."""
        X0 = self.X0
        return self.m_s * (1.0 + X0)

    @property
    def X0(self) -> float:
        """Initial dry-basis moisture."""
        return (1.0 - self.solids_fraction) / self.solids_fraction

    @property
    def m_s(self) -> float:
        """Solids mass, kg (from the initial diameter and composition)."""
        X0 = self.X0
        volume_per_solids = 1.0 / self.rho_s + X0 / self.rho_w
        return math.pi * self.d0**3 / 6.0 / volume_per_solids

    def diameter(self, X: np.ndarray) -> np.ndarray:
        """Droplet diameter at moisture ``X`` (ideal mixing of volumes), m."""
        return (6.0 * self.m_s * (1.0 / self.rho_s + np.asarray(X) / self.rho_w) / math.pi) ** (
            1 / 3
        )

    # -------------------------------------------------------------- physics
    def drying_curve(self, X: np.ndarray) -> np.ndarray:
        """The closure ``f(X)`` in ``[0, 1]``."""
        if self.rate_closure is not None:
            return np.asarray(self.rate_closure(np.asarray(X, dtype=float)), dtype=float)
        s = np.maximum(np.asarray(X, dtype=float) - self.X_eq, 0.0) / (self.X_cr - self.X_eq)
        return np.minimum(1.0, s**self.n)

    def transfer(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(area, h, h_m)``: surface, heat- and mass-transfer coefficients at moisture ``X``."""
        d = self.diameter(X)
        Re = self.rho_air * self.velocity * d / self.mu_air
        Sc = self.mu_air / (self.rho_air * self.D_v)
        Pr = self.cp_air * self.mu_air / self.k_air
        Sh = 2.0 + 0.6 * np.sqrt(Re) * Sc ** (1 / 3)
        Nu = 2.0 + 0.6 * np.sqrt(Re) * Pr ** (1 / 3)
        return math.pi * d**2, Nu * self.k_air / d, Sh * self.D_v / d

    def rhs(self, t: float, s: np.ndarray) -> np.ndarray:
        """``d[X, T]/dt``."""
        X, T = s
        area, h, h_m = self.transfer(X)
        rho_vs = saturation_pressure(T) * M_W / (R_GAS * T)
        rho_va = self.Y_air * self.rho_air
        rate = h_m * area * (rho_vs - rho_va) * self.drying_curve(X) / self.m_s  # -dX/dt
        dT = (h * area * (_as_fn(self.T_air)(t) - T) - self.latent_heat * self.m_s * rate) / (
            self.m_s * (self.c_s + self.c_w * X)
        )
        return np.array([-rate, dT])

    def reference(self, t_max: float, nt: int = 301, method: str = "LSODA") -> DropletReference:
        """Integrate the droplet from its initial state over ``[0, t_max]``."""
        t = np.linspace(0.0, t_max, nt)
        sol = solve_ivp(
            self.rhs,
            (0.0, t_max),
            [self.X0, self.T0],
            t_eval=t,
            method=method,
            rtol=1e-9,
            atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(f"droplet reference failed: {sol.message}")
        X, T = sol.y
        return DropletReference(t=t, X=X, T=T, m=self.m_s * (1.0 + X), d=self.diameter(X))

    def closure_from_record(
        self,
        times: np.ndarray,
        mass: np.ndarray,
        temperature: np.ndarray,
        window: int = 21,
        order: int = 3,
        margin: float = 0.02,
    ) -> tuple[np.ndarray, np.ndarray]:
        """The drying-rate closure read off a mass-and-temperature record, point by point.

        ``f_i = -dX/dt m_s / (h_m A (rho_vs(T_i) - rho_va))`` with the moisture
        and its rate from a Savitzky–Golay filter of the mass record (window
        and polynomial order as given) and the transfer coefficients of this
        model. Returns ``(X_i, f_i)`` for the readings with ``X_i > X_eq +
        margin``, ``f_i`` clipped to ``[0, 1]``: the start for an
        :class:`~pinnrl.pdes.unknowns.UnknownFunction` (``fn.fit(...)``).
        """
        from scipy.signal import savgol_filter

        times = np.asarray(times, dtype=float)
        X_rec = np.asarray(mass, dtype=float) / self.m_s - 1.0
        dt = float(times[1] - times[0])
        X = savgol_filter(X_rec, window, order)
        dX = savgol_filter(X_rec, window, order, deriv=1, delta=dt)
        T = savgol_filter(np.asarray(temperature, dtype=float), window, order)
        area, _h, h_m = self.transfer(np.maximum(X, 0.0))
        rho_vs = saturation_pressure(T) * M_W / (R_GAS * T)
        drive = h_m * area * (rho_vs - self.Y_air * self.rho_air) / self.m_s
        f = np.clip(-dX / np.maximum(drive, 1e-12), 0.0, 1.0)
        keep = X > self.X_eq + margin
        return X[keep], f[keep]

    def record_misfit(
        self,
        times: np.ndarray,
        mass: np.ndarray,
        temperature: np.ndarray,
        sigma_mass: float,
        sigma_temperature: float,
        closure: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> dict[str, float]:
        """Reduced chi-square of a mass-and-temperature record under this model, solved by the ODE reference.

        ``closure`` replaces the drying-rate curve ``f(X)`` (an identified
        :class:`~pinnrl.pdes.unknowns.UnknownFunction` wrapped as a NumPy
        function, a pointwise estimate interpolated). Returns ``{"mass",
        "temperature", "total"}``, each the mean squared residual in units of
        its noise: about 1 when the model explains the record. This is the
        check a learned closure has to pass and the guard on a refinement
        (J07c): a closure whose misfit is above its starting estimate's is
        rejected. The reference solver makes the check independent of the
        network that produced the closure.
        """
        model = self if closure is None else replace(self, rate_closure=closure)
        times = np.asarray(times, dtype=float)
        # A closure that does not vanish at the equilibrium moisture dries the droplet past
        # zero volume; such a solve fails or returns NaN, and the misfit is infinite.
        try:
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                ref = model.reference(float(times[-1]), nt=max(2001, 4 * times.size + 1))
        except RuntimeError:
            return {"mass": math.inf, "temperature": math.inf, "total": math.inf}
        r_m = (np.interp(times, ref.t, ref.m) - np.asarray(mass, dtype=float)) / sigma_mass
        r_T = (
            np.interp(times, ref.t, ref.T) - np.asarray(temperature, dtype=float)
        ) / sigma_temperature
        chi_m, chi_T = (float(np.mean(r**2)) for r in (r_m, r_T))
        chi_m, chi_T = (c if math.isfinite(c) else math.inf for c in (chi_m, chi_T))
        return {"mass": chi_m, "temperature": chi_T, "total": 0.5 * (chi_m + chi_T)}

    # ------------------------------------------------------------- the PINN
    def expressions(self) -> dict[str, str]:
        """The text sub-expressions of the model in the fields ``X`` and ``T`` (and ``f`` for the closure)."""
        # max(X, 0): a negative moisture from an untrained network would give a complex cube root
        d = "(6*m_s*(1/rho_s + max(X, 0)/rho_w)/pi)^(1/3)"
        Re = f"(rho_air*u*{d}/mu_air)"
        Sh = f"(2 + 0.6*sqrt({Re})*Sc^(1/3))"
        Nu = f"(2 + 0.6*sqrt({Re})*Pr^(1/3))"
        area = f"(pi*({d})^2)"
        h = f"({Nu}*k_air/{d})"
        h_m = f"({Sh}*D_v/{d})"
        # max(T, 273.15) keeps the Magnus exponent finite while the network's T is untrained
        Tg = "max(T, 273.15)"
        p_sat = f"(610.94*exp(17.625*({Tg} - 273.15)/({Tg} - 273.15 + 243.04)))"
        rho_vs = f"({p_sat}*M_w/(R*{Tg}))"
        rate = f"({h_m}*{area}*({rho_vs} - Y_air*rho_air)*f/m_s)"
        return {"d": d, "area": area, "h": h, "h_m": h_m, "rho_vs": rho_vs, "rate": rate}

    def closure_expression(self) -> str:
        """The characteristic drying curve as text in ``X``."""
        return f"min(1, (max(X - {self.X_eq}, 0)/({self.X_cr} - {self.X_eq}))^{self.n})"

    def custom_pde(
        self,
        t_max: float,
        *,
        inputs_grid: int = 201,
        unknown_rate: Mapping[str, Any] | None = None,
        sensors: list | None = None,
        name: str = "Drying droplet",
    ):
        """The same model as a :class:`~pinnrl.pdes.custom.CustomPDE` (an ODE in ``t``).

        Fields ``X`` and ``T``; the air temperature as a :class:`~pinnrl.pdes.inputs.TimeSeries`
        input ``Tair``; the closure ``f`` either substituted as text (the true
        curve) or, with ``unknown_rate``, an :class:`~pinnrl.pdes.unknowns.UnknownFunction`
        of ``X`` bounded to ``[0, 1]`` and increasing in ``X`` by default.
        """
        from pinnrl.pdes.custom import CustomPDE
        from pinnrl.pdes.inputs import TimeSeries

        ts = np.linspace(0.0, t_max, inputs_grid)
        series = {"Tair": TimeSeries(ts, [_as_fn(self.T_air)(x) for x in ts], name="Tair")}
        params = {
            "m_s": self.m_s,
            "rho_s": self.rho_s,
            "rho_w": self.rho_w,
            "c_s": self.c_s,
            "c_w": self.c_w,
            "lam": self.latent_heat,
            "Y_air": self.Y_air,
            "u": self.velocity,
            "rho_air": self.rho_air,
            "mu_air": self.mu_air,
            "k_air": self.k_air,
            "D_v": self.D_v,
            "Sc": self.mu_air / (self.rho_air * self.D_v),
            "Pr": self.cp_air * self.mu_air / self.k_air,
            "M_w": M_W,
            "R": R_GAS,
        }
        ex = self.expressions()
        unknown_functions = None
        if unknown_rate is None:
            rate = ex["rate"].replace("*f/", f"*{self.closure_expression()}/")
        else:
            spec = {
                "inputs": ["X"],
                "input_ranges": {"X": [0.0, self.X0]},
                "bounds": [0.0, 1.0],
                "monotone": {"X": "increasing"},
                "initial": 0.5,
                "hidden": [16, 16],
                "truth": self.closure_expression(),
            }
            spec.update(unknown_rate)
            unknown_functions = {"f": spec}
            rate = ex["rate"]
        equations = [
            f"X_t + {rate}",
            f"T_t - ({ex['h']}*{ex['area']}*(Tair - T) - lam*m_s*{rate})/(m_s*(c_s + c_w*X))",
        ]
        T_mid = 0.5 * (self.T0 + _as_fn(self.T_air)(0.0))
        return CustomPDE(
            equations,
            domain={"t": (0.0, t_max)},
            fields=["X", "T"],
            parameters=params,
            inputs=series,
            unknown_functions=unknown_functions,
            initial_condition={"X": str(self.X0), "T": str(self.T0)},
            scales={
                "X": (0.5 * self.X0, 0.5 * self.X0),
                "T": (T_mid, 0.5 * abs(_as_fn(self.T_air)(0.0) - self.T0)),
            },
            sensors=sensors,
            name=name,
        )
