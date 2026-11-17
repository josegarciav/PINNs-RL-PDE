"""A counter-current spray-drying tower, steady and transient (J13).

In the co-current tower of :mod:`pinnrl.process.dryer` the droplets and the
air travel together, and the steady profiles follow from one initial-value
problem along the residence time: 5 ms a solve. Here the air enters at the
bottom and rises against the falling droplets. The droplets' state is given
at the top (the atomiser) and the air's at the bottom (its inlet), so the
steady state is a two-point boundary-value problem, and the transient is a
system of transport equations whose characteristics run in opposite
directions, with ``z`` downward from the atomiser::

    X_t  + v_p X_z  = -r
    T_t  + v_p T_z  = (h A (T_a - T) - lambda m_s r) / (m_s (c_s + c_w X))
    Y_t  - v_a Y_z  = (v_a / G) S
    Ta_t - v_a Ta_z = -(v_a / G) [Q + c_v (T_a - T) S + U_w pi D (T_a - T_amb)] / (c_a + c_v Y)

with the same per-droplet drying rate ``r`` and closure ``f(X)`` as the
droplet and the co-current tower, ``S = (F_s / v_p) r`` the water the
droplets give to the air per metre of height, ``Q = (F_s / (m_s v_p)) h A
(T_a - T)`` the heat they take from it, ``v_a = G / (rho_a A_tower)`` the
air's velocity. The droplets cross the tower in seconds, the air in minutes;
the dry powder leaves at the bottom into the hottest air, which is what makes
the counter-current tower efficient and its stickiness margin the design
question.

:meth:`CounterCurrentDryer.steady` solves the boundary-value problem by
shooting on the air's state at the top; :meth:`CounterCurrentDryer.transient`
integrates the method-of-lines form (first-order upwind along each family of
characteristics, BDF in time with its sparse Jacobian) from the steady state
under time-varying inlet air; :meth:`CounterCurrentDryer.custom_pde` writes
the same equations in ``(z, t)`` for the PINN, with the inlet signals as
inputs, the steady profiles as the initial condition, and optionally the
curve's parameters or the whole closure as unknowns. Experiment J13 asks
whether this is where the PINN twin's case begins.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from scipy.sparse.linalg import spsolve

from pinnrl.benchmarks.reference import ReferenceSolution

from .droplet import M_W, R_GAS, DryingDroplet, saturation_pressure
from .dryer import _dryer_droplet, gordon_taylor

Signal = float | Callable[[float], float]
FIELDS = ("X", "T_p", "Y", "T_a")


def _as_fn(value: Signal) -> Callable[[float], float]:
    if callable(value):
        return value
    return lambda _t, v=float(value): v  # type: ignore[misc]


@dataclass
class CounterCurrentProfile:
    """The steady profiles along the tower, ``z`` downward from the atomiser.

    Attributes:
        z: Height below the atomiser, m.
        X: Droplet moisture (dry basis).
        T_p: Droplet temperature, K.
        Y: Air humidity ratio.
        T_a: Air temperature, K.
        air_top: The air's state at the top, ``(Y, T_a)``, the shooting unknowns.
    """

    z: np.ndarray
    X: np.ndarray
    T_p: np.ndarray
    Y: np.ndarray
    T_a: np.ndarray
    air_top: tuple[float, float]

    def outlet(self) -> dict[str, float]:
        """The powder at the bottom (``X``, ``T_p``) and the air at the top (``Y``, ``T_a``)."""
        return {
            "X": float(self.X[-1]),
            "T_p": float(self.T_p[-1]),
            "Y": float(self.Y[0]),
            "T_a": float(self.T_a[0]),
        }


@dataclass
class CounterCurrentTransient:
    """The method-of-lines solution on its cell centres ``z`` and output times ``t``.

    Attributes:
        z: Cell centres, m.
        t: Output times, s.
        X: ``(nz, nt)`` droplet moisture.
        T_p: Droplet temperature, K.
        Y: Air humidity ratio.
        T_a: Air temperature, K.
        stats: ``{"steps", "rhs_evaluations", "jacobians", "seconds"}`` of the integration.
    """

    z: np.ndarray
    t: np.ndarray
    X: np.ndarray
    T_p: np.ndarray
    Y: np.ndarray
    T_a: np.ndarray
    stats: dict[str, float]

    def outlet(self) -> dict[str, np.ndarray]:
        """``{"t", "X", "T_p"}`` of the powder at the bottom and ``{"Y", "T_a"}`` of the air at the top."""
        return {
            "t": self.t,
            "X": self.X[-1],
            "T_p": self.T_p[-1],
            "Y": self.Y[0],
            "T_a": self.T_a[0],
        }

    def as_reference_solutions(self) -> dict[str, ReferenceSolution]:
        """The four fields under the names of :meth:`CounterCurrentDryer.custom_pde` (``X``, ``T``, ``Y``, ``Ta``)."""
        names = {"X": "X", "T_p": "T", "Y": "Y", "T_a": "Ta"}
        return {
            names[attr]: ReferenceSolution(x=self.z, t=self.t, u=getattr(self, attr))
            for attr in FIELDS
        }


@dataclass
class CounterCurrentDryer:
    """A counter-current spray-drying tower with mono-disperse droplets (SI units).

    Defaults are the pilot tower of :class:`~pinnrl.process.dryer.SprayDryer`
    (8 m by 2.5 m, 0.36 kg/s of dry air, 0.01 kg/s of solids as 250 µm
    droplets of a 40 % concentrate falling at 1.5 m/s) with the air entering
    at the bottom at 160 °C and rising at 0.08 m/s (106 s of transit against
    5.3 s for the droplets). The powder leaves at a moisture of 0.067 and
    139 °C into the inlet air, the air at the top at 52 °C with a humidity
    ratio of 0.050.

    ``T_in`` and ``Y_in`` may be functions of time for the transient; the
    solids flow is constant.
    """

    droplet: DryingDroplet = field(default_factory=_dryer_droplet)
    height: float = 8.0  # m
    diameter: float = 2.5  # m
    particle_velocity: float = 1.5  # m/s, downward
    air_velocity: float | None = None  # m/s upward; default G / (rho_air * tower area)
    solids_flow: float = 0.01  # kg solids / s
    air_flow: float = 0.36  # kg dry air / s
    T_in: Signal = 433.15  # K, inlet air at the bottom
    Y_in: Signal = 0.01  # inlet humidity ratio
    T_ambient: float = 293.15  # K
    U_wall: float = 1.0  # W/m²K
    cp_vapour: float = 1880.0  # J/kg/K
    T_g_solids: float = 461.15  # K
    T_g_water: float = 138.15  # K
    gordon_taylor_k: float = 7.7
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- geometry
    def tower_area(self) -> float:
        """Cross-section of the tower, m²."""
        return math.pi * self.diameter**2 / 4.0

    def air_speed(self) -> float:
        """The air's velocity up the tower, m/s (``air_velocity`` or ``G / (rho_air A)``)."""
        if self.air_velocity is not None:
            return float(self.air_velocity)
        return self.air_flow / (self.droplet.rho_air * self.tower_area())

    def transit_times(self) -> dict[str, float]:
        """``{"droplets": H / v_p, "air": H / v_a}``, s."""
        return {
            "droplets": self.height / self.particle_velocity,
            "air": self.height / self.air_speed(),
        }

    # ----------------------------------------------------------------- physics
    def rates(
        self, X: np.ndarray, T_p: np.ndarray, Y: np.ndarray, T_a: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The material derivatives ``(DX, DT_p, DY, DT_a)``, per second, following each phase.

        The first two follow a droplet, the last two a parcel of air; they are
        the right-hand sides of the transport equations in the module docstring.
        """
        d = self.droplet
        X = np.asarray(X, dtype=float)
        area, h, h_m = d.transfer(np.maximum(X, 0.0))
        rho_vs = saturation_pressure(T_p) * M_W / (R_GAS * np.asarray(T_p, dtype=float))
        rate = h_m * area * (rho_vs - np.asarray(Y) * d.rho_air) * d.drying_curve(X) / d.m_s
        heat = h * area * (np.asarray(T_a) - np.asarray(T_p))  # W per droplet
        dT_p = (heat - d.latent_heat * d.m_s * rate) / (d.m_s * (d.c_s + d.c_w * X))
        per_metre = self.solids_flow / self.particle_velocity  # kg solids per m of height
        water = per_metre * rate  # kg water / s / m
        taken = per_metre / d.m_s * heat  # W / m
        wall = self.U_wall * math.pi * self.diameter * (np.asarray(T_a) - self.T_ambient)
        v_a, G = self.air_speed(), self.air_flow
        dY = v_a / G * water
        dT_a = -(v_a / G) * (taken + self.cp_vapour * (np.asarray(T_a) - T_p) * water + wall)
        dT_a = dT_a / (d.cp_air + self.cp_vapour * np.asarray(Y))
        return -rate, dT_p, dY, dT_a

    def _steady_rhs(self, _z: float, s: np.ndarray) -> np.ndarray:
        dX, dT, dY, dTa = self.rates(*s)
        v_p, v_a = self.particle_velocity, self.air_speed()
        return np.array([dX / v_p, dT / v_p, -dY / v_a, -dTa / v_a])

    def _shoot(self, air_top: Sequence[float] | np.ndarray, z: np.ndarray | None, t: float) -> Any:
        d = self.droplet
        return solve_ivp(
            self._steady_rhs,
            (0.0, self.height),
            [d.X0, d.T0, float(air_top[0]), float(air_top[1])],
            t_eval=z,
            method="LSODA",
            rtol=1e-10,
            atol=1e-13,
        )

    def steady(
        self,
        n: int = 201,
        t: float = 0.0,
        guess: Sequence[float] | None = None,
    ) -> CounterCurrentProfile:
        """The steady profiles at the inlet conditions of time ``t``, by shooting.

        The unknowns are the air's humidity and temperature at the top; each
        trial integrates the four ODEs down the tower and the two residuals are
        the air's state at the bottom against its inlet. ``guess`` starts the
        root search (default: the water balance for a powder moisture of 0.1).

        Raises:
            RuntimeError: When the shooting does not converge.
        """
        T_in, Y_in = _as_fn(self.T_in)(t), _as_fn(self.Y_in)(t)
        d = self.droplet
        if guess is None:
            dY = self.solids_flow * (d.X0 - 0.1) / self.air_flow
            guess = (Y_in + dY, T_in - d.latent_heat * dY / d.cp_air)

        def residual(u: np.ndarray) -> np.ndarray:
            sol = self._shoot(u, None, t)
            if not sol.success:
                return np.array([1e3, 1e3])
            return np.array([(sol.y[2, -1] - Y_in) / 1e-2, (sol.y[3, -1] - T_in) / 10.0])

        res = root(residual, np.asarray(guess, dtype=float), method="hybr", options={"eps": 1e-7})
        if not res.success or np.max(np.abs(res.fun)) > 1e-6:
            raise RuntimeError(f"counter-current steady state did not converge: {res.message}")
        z = np.linspace(0.0, self.height, n)
        sol = self._shoot(res.x, z, t)
        X, T_p, Y, T_a = sol.y
        return CounterCurrentProfile(
            z=z, X=X, T_p=T_p, Y=Y, T_a=T_a, air_top=(float(res.x[0]), float(res.x[1]))
        )

    def balances(self, prof: CounterCurrentProfile | None = None) -> dict[str, float]:
        """Relative closure of the water and energy balances of a steady profile.

        ``water``: the water the droplets lost against the water the air
        gained; ``energy``: the enthalpy leaving minus the enthalpy entering
        plus the wall loss (and the latent-heat inconsistency of a constant
        latent heat, as for the co-current tower), relative to the heat the air
        gave up. Both vanish for the exact solution.
        """
        prof = self.steady() if prof is None else prof
        d = self.droplet
        Fs, G, lam = self.solids_flow, self.air_flow, d.latent_heat
        lost = Fs * (prof.X[0] - prof.X[-1])
        gained = G * (prof.Y[0] - prof.Y[-1])
        T0 = 273.15

        def air(k: int) -> float:
            Y, T = prof.Y[k], prof.T_a[k]
            return float(G * ((d.cp_air + self.cp_vapour * Y) * (T - T0) + lam * Y))

        def powder(k: int) -> float:
            return float(Fs * (d.c_s + d.c_w * prof.X[k]) * (prof.T_p[k] - T0))

        wall = float(
            np.trapezoid(
                self.U_wall * math.pi * self.diameter * (prof.T_a - self.T_ambient), prof.z
            )
        )
        rate = -np.gradient(prof.X, prof.z, edge_order=2) * self.particle_velocity
        latent = (
            float(np.trapezoid(Fs * rate * (self.cp_vapour - d.c_w) * (prof.T_p - T0), prof.z))
            / self.particle_velocity
        )
        entering = air(-1) + powder(0)
        leaving = air(0) + powder(-1)
        given_up = G * d.cp_air * (prof.T_a[-1] - prof.T_a[0])
        return {
            "water": float((lost - gained) / lost),
            "energy": float((leaving - entering + wall - latent) / given_up),
            "latent_inconsistency": float(latent / given_up),
            "wall_loss_fraction": float(wall / given_up),
        }

    # -------------------------------------------------------------- stickiness
    def glass_transition(self, X: np.ndarray) -> np.ndarray:
        """Gordon–Taylor glass transition of the solids at moisture ``X``, K."""
        return gordon_taylor(X, self.T_g_solids, self.T_g_water, self.gordon_taylor_k)

    def stickiness_margin(self, prof: CounterCurrentProfile | None = None) -> np.ndarray:
        """``T_p - T_g(X)`` along the tower, K; largest at the bottom, where the dry powder meets the inlet air."""
        prof = self.steady() if prof is None else prof
        return prof.T_p - self.glass_transition(prof.X)

    # ---------------------------------------------------------- method of lines
    def cell_centres(self, nz: int) -> np.ndarray:
        """The ``nz`` cell centres of the method-of-lines grid, m."""
        dz = self.height / nz
        return (np.arange(nz) + 0.5) * dz

    def mol_rhs(self, nz: int) -> Callable[[float, np.ndarray], np.ndarray]:
        """``d/dt`` of the state ``[X, T_p, Y, T_a]`` (each ``nz`` cells), first-order upwind along each family.

        The droplets take their upstream value from the cell above (the
        atomiser's feed for the first cell), the air from the cell below (the
        inlet signals for the last).
        """
        dz = self.height / nz
        v_p, v_a = self.particle_velocity, self.air_speed()
        d = self.droplet
        T_in, Y_in = _as_fn(self.T_in), _as_fn(self.Y_in)

        def rhs(t: float, s: np.ndarray) -> np.ndarray:
            X, T, Y, Ta = s.reshape(4, nz)
            dX, dT, dY, dTa = self.rates(X, T, Y, Ta)
            out = np.empty((4, nz))
            out[0] = dX - v_p / dz * (X - np.concatenate([[d.X0], X[:-1]]))
            out[1] = dT - v_p / dz * (T - np.concatenate([[d.T0], T[:-1]]))
            out[2] = dY + v_a / dz * (np.concatenate([Y[1:], [Y_in(t)]]) - Y)
            out[3] = dTa + v_a / dz * (np.concatenate([Ta[1:], [T_in(t)]]) - Ta)
            return out.ravel()

        return rhs

    @staticmethod
    def jacobian_sparsity(nz: int) -> csr_matrix:
        """The sparsity of the method-of-lines Jacobian: four coupled states per cell, one neighbour per family."""
        S = lil_matrix((4 * nz, 4 * nz), dtype=np.int8)
        for i in range(nz):
            for k in range(4):
                for m in range(4):
                    S[k * nz + i, m * nz + i] = 1
            if i > 0:
                S[i, i - 1] = 1
                S[nz + i, nz + i - 1] = 1
            if i < nz - 1:
                S[2 * nz + i, 2 * nz + i + 1] = 1
                S[3 * nz + i, 3 * nz + i + 1] = 1
        return S.tocsr()

    def steady_cells(self, nz: int, t: float = 0.0) -> np.ndarray:
        """The steady state of the ``nz``-cell method-of-lines model at the inlet conditions of time ``t``.

        The shooting profile interpolated to the cells starts a damped Newton
        solve of ``rhs = 0`` with the sparse finite-difference Jacobian, so a
        transient started from it is at rest to rounding, with no
        discretisation transient of its own.

        Raises:
            RuntimeError: When the solve does not reach a steady state.
        """
        prof = self.steady(t=t)
        zc = self.cell_centres(nz)
        s0 = np.concatenate([np.interp(zc, prof.z, getattr(prof, a)) for a in FIELDS])
        rhs = self.mol_rhs(nz)
        sparsity = self.jacobian_sparsity(nz)
        groups = group_columns(sparsity)
        scale = np.concatenate(
            [np.full(nz, 1.0), np.full(nz, 100.0), np.full(nz, 0.01), np.full(nz, 100.0)]
        )
        weight = 1.0 / scale

        def residual(u: np.ndarray) -> np.ndarray:
            return rhs(t, u * scale) * weight

        u = s0 / scale
        f = residual(u)
        for _ in range(30):
            norm = float(np.max(np.abs(f)))
            if norm < 1e-11:
                break
            J = approx_derivative(residual, u, f0=f, sparsity=(sparsity, groups))
            step = spsolve(csc_matrix(J), -f)
            damping = 1.0
            while damping > 1e-3:  # a damped Newton step: accept only a decrease
                trial = u + damping * step
                f_trial = residual(trial)
                if np.max(np.abs(f_trial)) < norm:
                    u, f = trial, f_trial
                    break
                damping *= 0.5
            else:
                break
        s = u * scale
        rest = float(np.max(np.abs(residual(u))))
        if rest > 1e-8:
            raise RuntimeError(
                f"method-of-lines steady state did not converge (scaled residual {rest:.1e})"
            )
        return s

    def transient(
        self,
        t_max: float,
        nz: int = 101,
        nt: int = 201,
        initial: np.ndarray | str = "steady",
        method: str = "BDF",
        rtol: float = 1e-7,
        atol: float = 1e-10,
    ) -> CounterCurrentTransient:
        """The dynamic solution under the time-varying inlet air, by the method of lines.

        Args:
            t_max: Final time, s.
            nz: Cells along the height.
            nt: Output times on ``[0, t_max]``.
            initial: ``"steady"`` (the steady state at ``t = 0``, see
                :meth:`steady_cells`) or a state vector of ``4 nz`` values.
            method: A ``solve_ivp`` method; ``BDF`` (default) and ``Radau``
                receive the Jacobian's sparsity.
            rtol: Relative tolerance.
            atol: Absolute tolerance.

        Raises:
            RuntimeError: When the integration fails.
        """
        rhs = self.mol_rhs(nz)
        s0 = self.steady_cells(nz) if isinstance(initial, str) else np.asarray(initial, dtype=float)
        t = np.linspace(0.0, t_max, nt)
        start = time.perf_counter()
        kwargs: dict[str, Any] = {}
        if method in ("BDF", "Radau", "LSODA"):
            kwargs["jac_sparsity"] = self.jacobian_sparsity(nz)
        sol = solve_ivp(
            rhs, (0.0, t_max), s0, t_eval=t, method=method, rtol=rtol, atol=atol, **kwargs
        )
        if not sol.success:
            raise RuntimeError(f"counter-current transient failed: {sol.message}")
        X, T_p, Y, T_a = sol.y.reshape(4, nz, nt)
        stats = {
            "steps": float(sol.t.size),
            "rhs_evaluations": float(sol.nfev),
            "jacobians": float(sol.njev),
            "seconds": time.perf_counter() - start,
        }
        return CounterCurrentTransient(
            z=self.cell_centres(nz), t=t, X=X, T_p=T_p, Y=Y, T_a=T_a, stats=stats
        )

    # ----------------------------------------------------------------- records
    def records(
        self,
        trans: CounterCurrentTransient,
        probe_heights: Sequence[float] = (0.5, 2.0, 3.5, 5.0, 6.5),
    ) -> dict[str, np.ndarray]:
        """What a plant records over a transient, without noise.

        ``{"t", "T_air": air temperatures at the probe heights ``(n_probes,
        nt)``, "Y_out": the humidity of the air leaving at the top, "X_out":
        the moisture of the powder leaving at the bottom}``.
        """
        z = np.asarray(probe_heights, dtype=float)
        if np.any(z < 0) or np.any(z > self.height):
            raise ValueError(f"probe heights must lie in [0, {self.height}]")
        T_air = np.stack(
            [np.interp(z, trans.z, trans.T_a[:, j]) for j in range(trans.t.size)], axis=1
        )
        return {"t": trans.t, "T_air": T_air, "Y_out": trans.Y[0], "X_out": trans.X[-1]}

    def with_closure(self, closure: Callable[[np.ndarray], np.ndarray]) -> CounterCurrentDryer:
        """The same tower with the droplets' drying-rate closure replaced by ``closure(X)``."""
        return replace(self, droplet=replace(self.droplet, rate_closure=closure))

    def operating_point(self, **changes: Any) -> CounterCurrentDryer:
        """The same tower at other conditions (``T_in``, ``solids_flow``, ``air_flow``, ...)."""
        return replace(self, **changes)

    def residual_scales(self, t: float = 0.0) -> dict[str, float]:
        """The size of each equation's terms along the steady profile, to make the four residuals comparable.

        ``X``, ``Y`` and ``Ta``: the largest material derivative of the field
        along the steady profile at time ``t`` (which the transport term
        balances); ``T``: the largest heating term ``h A (T_a - T) / (m_s c)``,
        thousands of kelvin per second, because the droplet's temperature is
        a stiff balance of two large terms and its residual is only
        meaningful on their scale. :meth:`custom_pde` divides each equation by
        these, so the residual losses weigh the fields alike.
        """
        prof = self.steady(t=t)
        d = self.droplet
        dX, _dT, dY, dTa = self.rates(prof.X, prof.T_p, prof.Y, prof.T_a)
        area, h, _h_m = d.transfer(np.maximum(prof.X, 0.0))
        heating = h * area * (prof.T_a - prof.T_p) / (d.m_s * (d.c_s + d.c_w * prof.X))
        return {
            "X": max(float(np.max(np.abs(dX))), 1e-3),
            "T": max(float(np.max(np.abs(heating))), 1.0),
            "Y": max(float(np.max(np.abs(dY))), 1e-6),
            "Ta": max(float(np.max(np.abs(dTa))), 1e-2),
        }

    # -------------------------------------------------------------------- PINN
    def custom_pde(
        self,
        t_max: float,
        *,
        inputs_grid: int = 201,
        unknown_rate: Mapping[str, Any] | None = None,
        identify: Sequence[str] = (),
        initial_guesses: Mapping[str, float] | None = None,
        sensors: list | None = None,
        name: str = "Counter-current spray dryer",
    ):
        """The same model as a :class:`~pinnrl.pdes.custom.CustomPDE` in ``(z, t)``.

        Fields ``X``, ``T`` (droplets), ``Y``, ``Ta`` (air); the inlet air's
        temperature and humidity as :class:`~pinnrl.pdes.inputs.TimeSeries`
        inputs ``Tin`` and ``Yin`` on ``inputs_grid`` points (a name with an
        underscore after a field's name would read as a derivative); the feed as Dirichlet conditions at
        the top and the inlet air at the bottom; the steady profiles at ``t =
        0`` as the initial condition. Each equation is divided by its scale
        from :meth:`residual_scales`, so the four residuals are comparable in
        the loss. The closure is the droplet's curve in
        the parameters ``Xcr``, ``ncdc`` and ``Xeq`` or, with
        ``unknown_rate``, an anchored monotone
        :class:`~pinnrl.pdes.unknowns.UnknownFunction` of ``X``. ``identify``
        names parameters to estimate in inverse mode (``"Xcr"``, ``"ncdc"``,
        ``"U_w"`` for the wall-loss coefficient), with ``initial_guesses``.

        Raises:
            ValueError: For a name in ``identify`` that is not a parameter of the model.
        """
        from pinnrl.pdes.custom import CustomPDE
        from pinnrl.pdes.inputs import TimeSeries

        d = self.droplet
        ts = np.linspace(0.0, t_max, inputs_grid)
        series = {
            "Tin": TimeSeries(ts, [_as_fn(self.T_in)(x) for x in ts], name="Tin"),
            "Yin": TimeSeries(ts, [_as_fn(self.Y_in)(x) for x in ts], name="Yin"),
        }
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
            "v_p": self.particle_velocity,
            "v_a": self.air_speed(),
            "U_w": self.U_wall,
            "pi_D": math.pi * self.diameter,
            "Tamb": self.T_ambient,
            "Xcr": d.X_cr,
            "ncdc": d.n,
            "Xeq": d.X_eq,
        }
        for pname in identify:
            if pname not in ("Xcr", "ncdc", "U_w"):
                raise ValueError(f"identify accepts 'Xcr', 'ncdc' and 'U_w', not {pname!r}")
        ex = d.expressions()
        rate = ex["rate"].replace("Y_air", "Y")
        unknown_functions = None
        if unknown_rate is None:
            closure = "min(1, (max(X - Xeq, 0)/(Xcr - Xeq))^ncdc)"
            rate = rate.replace("*f/", f"*{closure}/")
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
        water = f"(F_s/v_p*{rate})"
        taken = f"(F_s/(m_s*v_p)*{heat})"
        r = self.residual_scales()
        equations = [
            f"(X_t + v_p*X_z + {rate})/{r['X']!r}",
            f"(T_t + v_p*T_z - ({heat} - lam*m_s*{rate})/(m_s*(c_s + c_w*X)))/{r['T']!r}",
            f"(Y_t - v_a*Y_z - v_a/G*{water})/{r['Y']!r}",
            f"(Ta_t - v_a*Ta_z + v_a/G*({taken} + c_v*(Ta - T)*{water} + U_w*pi_D*(Ta - Tamb))"
            f"/(c_a + c_v*Y))/{r['Ta']!r}",
        ]
        prof = self.steady(n=inputs_grid, t=0.0)

        def initial(x):
            import torch

            zz = x[:, 0].detach().cpu().numpy()
            cols = [
                torch.as_tensor(np.interp(zz, prof.z, getattr(prof, a)), dtype=x.dtype).reshape(
                    -1, 1
                )
                for a in FIELDS
            ]
            return torch.cat(cols, dim=1)

        out = prof.outlet()
        T_in0 = float(_as_fn(self.T_in)(0.0))
        Y_in0 = float(_as_fn(self.Y_in)(0.0))
        return CustomPDE(
            equations,
            domain={"z": (0.0, self.height), "t": (0.0, t_max)},
            fields=["X", "T", "Y", "Ta"],
            parameters=params,
            inputs=series,
            unknown_functions=unknown_functions,
            initial_condition=initial,
            boundary_conditions={
                "X": {"dirichlet": {"left": str(d.X0)}},
                "T": {"dirichlet": {"left": str(d.T0)}},
                "Y": {"dirichlet": {"right": "Yin"}},
                "Ta": {"dirichlet": {"right": "Tin"}},
            },
            trainable_parameters=list(identify),
            parameter_initial_guesses=dict(initial_guesses or {}),
            scales={
                "X": (0.5 * d.X0, 0.5 * d.X0),
                "T": (0.5 * (d.T0 + out["T_p"]), max(0.5 * abs(out["T_p"] - d.T0), 5.0)),
                "Y": (0.5 * (Y_in0 + out["Y"]), max(0.5 * (out["Y"] - Y_in0), 1e-3)),
                "Ta": (0.5 * (T_in0 + out["T_a"]), max(0.5 * (T_in0 - out["T_a"]), 5.0)),
            },
            sensors=sensors,
            name=name,
        )
