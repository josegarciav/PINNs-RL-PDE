"""A co-current spray dryer at steady state: the drying droplet coupled to the air it dries in.

The single droplet of :mod:`pinnrl.process.droplet` sees air of fixed
humidity. In a dryer the air is part of the problem: the water the droplets
lose humidifies it and the heat they take cools it, so the driving forces
fall along the tower. This module is the smallest model with that coupling: a
plug flow of equal droplets travelling down the tower with the air
(co-current), followed along their residence time ``tau = z / v_p``::

    dX/dtau   = -r(X, T_p, Y, T_a)
    dT_p/dtau = (h A (T_a - T_p) - lambda m_s r) / (m_s (c_s + c_w X))
    dY/dtau   = (F_s / G) r
    dT_a/dtau = -[(F_s / m_s) h A (T_a - T_p) + G c_v (T_a - T_p) dY/dtau
                  + U_w pi D v_p (T_a - T_amb)] / (G (c_a + c_v Y))

with the drying rate per unit solids ``r = h_m A (rho_vs(T_p) - Y rho_a) f(X) / m_s``
and the same closure ``f(X)`` (the characteristic drying curve) as the
droplet. ``F_s`` is the solids flow, ``G`` the dry-air flow, ``U_w`` the wall
loss coefficient. The water and energy balances close exactly (see
:meth:`SprayDryer.balances`).

What a plant measures on such a tower is far less than a droplet rig gives:
air temperatures at a few heights, the outlet air's humidity and the powder's
moisture from the lab. :meth:`SprayDryer.records` returns those for an
operating point; experiment J11 asks how much of the closure they determine.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from .droplet import M_W, R_GAS, DryingDroplet, saturation_pressure


def _dryer_droplet() -> DryingDroplet:
    """A 250 µm droplet of a 40 % solids concentrate at 50 °C with a slip velocity of 0.3 m/s."""
    return DryingDroplet(d0=250e-6, solids_fraction=0.4, T0=323.15, velocity=0.3)


@dataclass
class DryerReference:
    """The steady profiles along the tower.

    Attributes:
        tau: Residence time of the droplets, s.
        z: Distance below the atomiser, m.
        X: Droplet moisture (dry basis).
        T_p: Droplet temperature, K.
        Y: Air humidity ratio, kg water / kg dry air.
        T_a: Air temperature, K.
    """

    tau: np.ndarray
    z: np.ndarray
    X: np.ndarray
    T_p: np.ndarray
    Y: np.ndarray
    T_a: np.ndarray

    def outlet(self) -> dict[str, float]:
        """The outlet state ``{"X", "T_p", "Y", "T_a"}``."""
        return {k: float(getattr(self, k)[-1]) for k in ("X", "T_p", "Y", "T_a")}


@dataclass
class SprayDryer:
    """A co-current spray-drying tower with mono-disperse droplets (SI units, steady state).

    Defaults are a pilot tower: 8 m of drying height and 2.5 m across, 0.36 kg/s
    of dry air entering at 180 °C with a humidity ratio of 0.01, and 0.01 kg/s
    of solids as 250 µm droplets of a 40 % concentrate that travel at 1.5 m/s
    (5.3 s of residence). The droplets dry over the whole height; the air
    leaves at 74 °C, the powder at a moisture of 0.085 (dry basis), and the
    wall loses 13 % of the heat the air gives up.
    """

    droplet: DryingDroplet = field(default_factory=_dryer_droplet)
    height: float = 8.0  # m
    diameter: float = 2.5  # m
    particle_velocity: float = 1.5  # m/s, downward velocity of the droplets
    solids_flow: float = 0.01  # kg solids / s
    air_flow: float = 0.36  # kg dry air / s
    T_in: float = 453.15  # K, inlet air
    Y_in: float = 0.01  # inlet humidity ratio
    T_ambient: float = 293.15  # K
    U_wall: float = 1.0  # W/m²K, wall loss coefficient
    cp_vapour: float = 1880.0  # J/kg/K
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- physics
    def residence_time(self) -> float:
        """``H / v_p``, s."""
        return self.height / self.particle_velocity

    def rhs(self, tau: float, s: np.ndarray) -> np.ndarray:
        """``d[X, T_p, Y, T_a] / dtau``."""
        d = self.droplet
        X, T_p, Y, T_a = s
        area, h, h_m = d.transfer(np.maximum(X, 0.0))
        rho_vs = saturation_pressure(T_p) * M_W / (R_GAS * T_p)
        rate = h_m * area * (rho_vs - Y * d.rho_air) * float(d.drying_curve(X)) / d.m_s
        heat = h * area * (T_a - T_p)  # W per droplet
        dT_p = (heat - d.latent_heat * d.m_s * rate) / (d.m_s * (d.c_s + d.c_w * X))
        dY = self.solids_flow / self.air_flow * rate
        wall = (
            self.U_wall * math.pi * self.diameter * self.particle_velocity * (T_a - self.T_ambient)
        )
        dT_a = -(
            self.solids_flow / d.m_s * heat
            + self.air_flow * self.cp_vapour * (T_a - T_p) * dY
            + wall
        ) / (self.air_flow * (d.cp_air + self.cp_vapour * Y))
        return np.array([-rate, dT_p, dY, dT_a])

    def reference(self, n: int = 201, method: str = "LSODA") -> DryerReference:
        """Integrate the profiles from the atomiser to the outlet on ``n`` points.

        Raises:
            RuntimeError: When the integration fails.
        """
        d = self.droplet
        tau = np.linspace(0.0, self.residence_time(), n)
        sol = solve_ivp(
            self.rhs,
            (0.0, tau[-1]),
            [d.X0, d.T0, self.Y_in, self.T_in],
            t_eval=tau,
            method=method,
            rtol=1e-9,
            atol=1e-11,
        )
        if not sol.success:
            raise RuntimeError(f"dryer reference failed: {sol.message}")
        X, T_p, Y, T_a = sol.y
        return DryerReference(tau=tau, z=self.particle_velocity * tau, X=X, T_p=T_p, Y=Y, T_a=T_a)

    def balances(self, ref: DryerReference | None = None) -> dict[str, float]:
        """Relative closure of the water and energy balances over the tower.

        ``water``: the water the droplets lost against the water the air
        gained. ``energy``: the change of the air's and the droplets' enthalpy
        plus the wall loss, relative to the heat the air gave up. The model
        takes the latent heat as constant, so its enthalpy is conserved up to
        the term ``F_s r (c_v - c_w) (T_p - T_0)`` (about 5 % of the latent
        heat), which is included; ``latent_inconsistency`` reports its size.
        All are zero for the exact solution of the model; the reference closes
        them to the quadrature error of its output grid.
        """
        ref = self.reference() if ref is None else ref
        d = self.droplet
        Fs, G, lam = self.solids_flow, self.air_flow, d.latent_heat
        lost = Fs * (ref.X[0] - ref.X[-1])
        gained = G * (ref.Y[-1] - ref.Y[0])
        T0 = 273.15  # reference temperature of the enthalpies, water liquid

        def enthalpy(k: int) -> float:
            air = G * ((d.cp_air + self.cp_vapour * ref.Y[k]) * (ref.T_a[k] - T0) + lam * ref.Y[k])
            return float(air + Fs * (d.c_s + d.c_w * ref.X[k]) * (ref.T_p[k] - T0))

        wall = float(
            np.trapezoid(self.U_wall * math.pi * self.diameter * (ref.T_a - self.T_ambient), ref.z)
        )
        rate = -np.gradient(ref.X, ref.tau, edge_order=2)
        latent = float(np.trapezoid(Fs * rate * (self.cp_vapour - d.c_w) * (ref.T_p - T0), ref.tau))
        given_up = G * d.cp_air * (ref.T_a[0] - ref.T_a[-1])
        return {
            "water": float((lost - gained) / lost),
            "energy": float((enthalpy(-1) - enthalpy(0) + wall - latent) / given_up),
            "latent_inconsistency": float(latent / given_up),
            "wall_loss_fraction": float(wall / given_up),
        }

    # ---------------------------------------------------------------- records
    def records(
        self,
        probe_heights: Sequence[float] = (1.0, 2.0, 3.5, 5.0, 6.5, 8.0),
        ref: DryerReference | None = None,
    ) -> dict[str, np.ndarray]:
        """What a plant measures at this operating point, without noise.

        ``{"T_air": air temperature at the probe heights (K), "Y_out": outlet
        humidity ratio, "X_out": powder moisture}``.
        """
        ref = self.reference() if ref is None else ref
        z = np.asarray(probe_heights, dtype=float)
        if np.any(z < 0) or np.any(z > self.height):
            raise ValueError(f"probe heights must lie in [0, {self.height}]")
        return {
            "T_air": np.interp(z, ref.z, ref.T_a),
            "Y_out": np.array([ref.Y[-1]]),
            "X_out": np.array([ref.X[-1]]),
        }

    def with_closure(self, closure) -> SprayDryer:
        """The same tower with the droplets' drying-rate closure replaced by ``closure(X)``."""
        return replace(self, droplet=replace(self.droplet, rate_closure=closure))

    def operating_point(self, **changes: Any) -> SprayDryer:
        """The same tower at other operating conditions (``T_in``, ``solids_flow``, ``air_flow``, ...)."""
        return replace(self, **changes)

    # ------------------------------------------------------------------ PINN
    def custom_pde(
        self,
        *,
        unknown_rate: Mapping[str, Any] | None = None,
        sensors: list | None = None,
        name: str = "Spray dryer",
    ):
        """The same model as a :class:`~pinnrl.pdes.custom.CustomPDE`: an ODE system in ``t = tau``.

        Fields ``X``, ``T`` (droplet), ``Y`` and ``Ta`` (air) along the
        residence time; a sensor at height ``z`` reads at ``t = z / v_p``. The
        closure ``f`` is the droplet's curve as text or, with ``unknown_rate``,
        an :class:`~pinnrl.pdes.unknowns.UnknownFunction` of ``X``, by default
        monotone and anchored at zero at the equilibrium moisture.
        """
        from pinnrl.pdes.custom import CustomPDE

        d = self.droplet
        params = {
            "m_s": d.m_s,
            "rho_s": d.rho_s,
            "rho_w": d.rho_w,
            "c_s": d.c_s,
            "c_w": d.c_w,
            "lam": d.latent_heat,
            "u": d.velocity,
            "rho_air": d.rho_air,
            "mu_air": d.mu_air,
            "k_air": d.k_air,
            "D_v": d.D_v,
            "Sc": d.mu_air / (d.rho_air * d.D_v),
            "Pr": d.cp_air * d.mu_air / d.k_air,
            "M_w": M_W,
            "R": R_GAS,
            "F_s": self.solids_flow,
            "G": self.air_flow,
            "c_a": d.cp_air,
            "c_v": self.cp_vapour,
            "wall": self.U_wall * math.pi * self.diameter * self.particle_velocity,
            "Tamb": self.T_ambient,
        }
        ex = d.expressions()
        unknown_functions = None
        rate = ex["rate"].replace("Y_air", "Y")
        if unknown_rate is None:
            rate = rate.replace("*f/", f"*{d.closure_expression()}/")
        else:
            spec = {
                "inputs": ["X"],
                "input_ranges": {"X": [0.0, d.X0]},
                "monotone": {"X": "increasing"},
                "anchor": {"at": {"X": d.X_eq}, "value": 0.0},
                "hidden": [16, 16],
                "truth": d.closure_expression(),
            }
            spec.update(unknown_rate)
            unknown_functions = {"f": spec}
        heat = f"{ex['h']}*{ex['area']}*(Ta - T)"
        equations = [
            f"X_t + {rate}",
            f"T_t - ({heat} - lam*m_s*{rate})/(m_s*(c_s + c_w*X))",
            f"Y_t - F_s/G*{rate}",
            f"Ta_t + (F_s/m_s*{heat} + c_v*(Ta - T)*F_s*{rate} + wall*(Ta - Tamb))/(G*(c_a + c_v*Y))",
        ]
        out = self.reference().outlet()
        return CustomPDE(
            equations,
            domain={"t": (0.0, self.residence_time())},
            fields=["X", "T", "Y", "Ta"],
            parameters=params,
            unknown_functions=unknown_functions,
            initial_condition={
                "X": str(d.X0),
                "T": str(d.T0),
                "Y": str(self.Y_in),
                "Ta": str(self.T_in),
            },
            scales={
                "X": (0.5 * d.X0, 0.5 * d.X0),
                "T": (0.5 * (d.T0 + out["T_p"]), max(0.5 * abs(out["T_p"] - d.T0), 5.0)),
                "Y": (0.5 * (self.Y_in + out["Y"]), max(0.5 * (out["Y"] - self.Y_in), 1e-3)),
                "Ta": (0.5 * (self.T_in + out["T_a"]), max(0.5 * (self.T_in - out["T_a"]), 5.0)),
            },
            sensors=sensors,
            name=name,
        )
