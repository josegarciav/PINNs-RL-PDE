"""A falling-film evaporator tube as a one-dimensional process model (J01).

The liquid film runs down a heated tube of length ``L`` at velocity ``v``.
Steam at ``T_s`` heats it through the wall; the film boils, so its
temperature is the boiling point ``T_b(X) = T_sat + k_bpe X`` of a solution
with solids mass fraction ``X`` (the boiling-point elevation), and the heat
flux goes into evaporation. With a fouling resistance ``R_f`` on the wall the
flux is ``q = (T_s - T_b(X)) / (1/U + R_f)``. Per unit length the film loses
water at the rate ``P q / lambda`` (``P`` the wetted perimeter, ``lambda``
the latent heat) and the solids are conserved, which gives two transport
equations for the liquid mass flow ``W`` and the mass fraction ``X``::

    W_t + v W_z = -P q(X) / lambda
    X_t + v X_z =  X P q(X) / (lambda W)

with the feed ``W(0, t) = W_in(t)``, ``X(0, t) = X_in(t)`` at the inlet, and
``T_s(t)`` and ``R_f(t)`` as inputs. The outlet ``X(L, t)`` and ``W(L, t)``
are what a plant measures. This is the smallest model with the structure of
the twin: measured inputs, a slowly drifting unknown (fouling), transport
delays and outlet sensors.

:class:`FallingFilmEvaporator` holds the geometry and physics,
:meth:`FallingFilmEvaporator.reference` solves it by the method of lines
(first-order upwind in ``z``, an implicit integrator in ``t``),
:meth:`FallingFilmEvaporator.steady_state` solves the steady profile as an
ODE in ``z``, and :meth:`FallingFilmEvaporator.custom_pde` writes the same
model as a :class:`~pinnrl.pdes.custom.CustomPDE` with its scales, inputs,
sensors and, optionally, ``R_f(t)`` as an unknown function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from pinnrl.benchmarks.reference import ReferenceSolution

Signal = float | Callable[[float], float]


def _as_fn(value: Signal) -> Callable[[float], float]:
    if callable(value):
        return value
    return lambda _t, v=float(value): v


@dataclass
class EvaporatorReference:
    """The reference fields of one simulation.

    Attributes:
        W: Liquid mass flow on the ``(nz, nt)`` grid, kg/s.
        X: Solids mass fraction on the grid.
        q: Wall heat flux on the grid, W/m².
        z: Axial grid, m.
        t: Time grid, s.
    """

    W: ReferenceSolution
    X: ReferenceSolution
    q: np.ndarray
    z: np.ndarray
    t: np.ndarray

    def outlet(self) -> dict[str, np.ndarray]:
        """Outlet records ``{"t", "W", "X"}`` at ``z = L``."""
        return {"t": self.t, "W": self.W.u[-1], "X": self.X.u[-1]}


@dataclass
class FallingFilmEvaporator:
    """Geometry, physics and operating point of one evaporator tube (SI units).

    Defaults are a dairy-evaporator tube: 10 m long, 50 mm bore, a clean
    overall coefficient of 2,000 W/m²K, steam 12 K above the solvent's
    saturation temperature, a feed of 0.1 kg/s at 10 % solids, and a film
    velocity of 0.5 m/s (20 s residence time); one pass evaporates about 14 %
    of the feed's water.
    """

    length: float = 10.0  # m
    velocity: float = 0.5  # m/s
    perimeter: float = float(np.pi * 0.05)  # m, wetted perimeter of a 50 mm tube
    U: float = 2000.0  # W/m²K, clean overall heat-transfer coefficient
    latent_heat: float = 2.3e6  # J/kg
    T_sat: float = 343.15  # K, solvent saturation temperature at the operating pressure
    k_bpe: float = 4.0  # K per unit mass fraction, boiling-point elevation
    W_in: Signal = 0.1  # kg/s
    X_in: Signal = 0.1
    T_s: Signal = 355.15  # K, steam temperature
    R_f: Signal = 0.0  # m²K/W, fouling resistance
    extra: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- physics
    def boiling_point(self, X: np.ndarray) -> np.ndarray:
        """``T_b(X) = T_sat + k_bpe X``."""
        return self.T_sat + self.k_bpe * np.asarray(X)

    def heat_flux(self, X: np.ndarray, t: float) -> np.ndarray:
        """``q = (T_s(t) - T_b(X)) / (1/U + R_f(t))``, W/m²."""
        return (_as_fn(self.T_s)(t) - self.boiling_point(X)) / (1.0 / self.U + _as_fn(self.R_f)(t))

    def residence_time(self) -> float:
        """``L / v``, s."""
        return self.length / self.velocity

    # ---------------------------------------------------------------- solvers
    def steady_state(self, nz: int = 201, t: float = 0.0) -> dict[str, np.ndarray]:
        """The steady profile at the inputs' values at time ``t``: an ODE in ``z``.

        Returns ``{"z", "W", "X", "q"}``.
        """
        z = np.linspace(0.0, self.length, nz)
        P, lam, v = self.perimeter, self.latent_heat, self.velocity

        def rhs(_z, s):
            W, X = s
            q = self.heat_flux(X, t)
            return [-P * q / (lam * v), X * P * q / (lam * W * v)]

        sol = solve_ivp(
            rhs,
            (0.0, self.length),
            [_as_fn(self.W_in)(t), _as_fn(self.X_in)(t)],
            t_eval=z,
            method="LSODA",
            rtol=1e-10,
            atol=1e-12,
        )
        if not sol.success:
            raise RuntimeError(f"steady-state solve failed: {sol.message}")
        return {"z": z, "W": sol.y[0], "X": sol.y[1], "q": self.heat_flux(sol.y[1], t)}

    def mol_rhs(self, nz: int = 201) -> Callable[[float, np.ndarray], np.ndarray]:
        """The method-of-lines right-hand side ``ds/dt = f(t, s)`` on ``nz`` axial points.

        The state is ``s = [W_0..W_{nz-1}, X_0..X_{nz-1}]``; first-order upwind
        in ``z``; the inlet entries are overwritten by the feed signals. The
        Jacobian of this function is what the linearized-stability analysis
        (K09) looks at.
        """
        dz = self.length / (nz - 1)
        P, lam, v = self.perimeter, self.latent_heat, self.velocity
        W_in, X_in = _as_fn(self.W_in), _as_fn(self.X_in)

        def rhs(tau, s):
            W, X = s[:nz].copy(), s[nz:].copy()
            W[0], X[0] = W_in(tau), X_in(tau)  # the inlet is a boundary value
            q = self.heat_flux(X, tau)
            dW = np.zeros(nz)
            dX = np.zeros(nz)
            dW[1:] = -v * (W[1:] - W[:-1]) / dz - P * q[1:] / lam
            dX[1:] = -v * (X[1:] - X[:-1]) / dz + X[1:] * P * q[1:] / (lam * W[1:])
            return np.concatenate([dW, dX])

        return rhs

    def reference(
        self,
        t_max: float,
        nz: int = 201,
        nt: int = 201,
        initial: Mapping[str, np.ndarray] | str = "steady",
        method: str = "BDF",
    ) -> EvaporatorReference:
        """Dynamic solution by the method of lines (first-order upwind, implicit in time).

        Args:
            t_max: Final time, s.
            nz: Axial grid points (including the inlet).
            nt: Output time points on ``[0, t_max]``.
            initial: ``"steady"`` (the steady profile at ``t = 0``) or
                ``{"W": array, "X": array}`` on the ``nz`` grid.
            method: ``solve_ivp`` method; ``BDF`` handles the stiff source.
        """
        z = np.linspace(0.0, self.length, nz)
        t = np.linspace(0.0, t_max, nt)
        W_in, X_in = _as_fn(self.W_in), _as_fn(self.X_in)
        if initial == "steady":
            ss = self.steady_state(nz, t=0.0)
            W0, X0 = ss["W"], ss["X"]
        else:
            W0, X0 = np.asarray(initial["W"], dtype=float), np.asarray(initial["X"], dtype=float)
        rhs = self.mol_rhs(nz)

        s0 = np.concatenate([W0, X0])
        sol = solve_ivp(rhs, (0.0, t_max), s0, t_eval=t, method=method, rtol=1e-8, atol=1e-11)
        if not sol.success:
            raise RuntimeError(f"evaporator reference failed: {sol.message}")
        W = sol.y[:nz]
        X = sol.y[nz:]
        for k, tau in enumerate(t):  # the inlet row carries the boundary values
            W[0, k], X[0, k] = W_in(tau), X_in(tau)
        q = np.stack([self.heat_flux(X[:, k], tau) for k, tau in enumerate(t)], axis=1)
        return EvaporatorReference(
            W=ReferenceSolution(x=z, t=t, u=W), X=ReferenceSolution(x=z, t=t, u=X), q=q, z=z, t=t
        )

    def fouling_state_space(self, nz: int = 11):
        """The interior states ``[W_1..W_{nz-1}, X_1..X_{nz-1}]`` with ``R_f`` as a parameter, for an estimator.

        Returns ``(f, h, s0)``: ``f(t, s, theta)`` the interior dynamics with
        ``theta = [R_f]`` (the other signals from this model's ``W_in``,
        ``X_in`` and ``T_s``), ``h(t, s)`` the outlet records ``[X(L), W(L)]``,
        and ``s0`` the steady interior state at ``t = 0`` with this model's
        ``R_f(0)``. The J05 Kalman-filter baseline is built on it.
        """
        holder = [float(_as_fn(self.R_f)(0.0))]
        model = replace(self, R_f=lambda t: holder[0])
        rhs = model.mol_rhs(nz)
        W_in, X_in = _as_fn(self.W_in), _as_fn(self.X_in)

        def f(t: float, s: np.ndarray, theta: np.ndarray) -> np.ndarray:
            holder[0] = float(theta[0])
            full = np.concatenate([[W_in(t)], s[: nz - 1], [X_in(t)], s[nz - 1 :]])
            d = rhs(t, full)
            return np.concatenate([d[1:nz], d[nz + 1 :]])

        def h(t: float, s: np.ndarray) -> np.ndarray:
            return np.array([s[2 * nz - 3], s[nz - 2]])

        ss = self.steady_state(nz, t=0.0)
        s0 = np.concatenate([ss["W"][1:], ss["X"][1:]])
        return f, h, s0

    # ------------------------------------------------------------- the PINN
    def custom_pde(
        self,
        t_max: float,
        *,
        inputs_grid: int = 201,
        unknown_fouling: Mapping[str, Any] | None = None,
        sensors: list | None = None,
        name: str = "Falling-film evaporator",
    ):
        """The same model as a :class:`~pinnrl.pdes.custom.CustomPDE`.

        Fields ``W`` and ``X`` of ``(z, t)``; the inputs ``W_in``, ``X_in``,
        ``T_s`` and (unless ``unknown_fouling`` is given) ``R_f`` as
        :class:`~pinnrl.pdes.inputs.TimeSeries` sampled on ``inputs_grid``
        points of ``[0, t_max]``; the inlet as one-sided Dirichlet conditions;
        the steady profile at ``t = 0`` as the initial condition; the scales
        of ``W`` and ``X`` from the feed. ``unknown_fouling`` is the spec of an
        :class:`~pinnrl.pdes.unknowns.UnknownFunction` of ``t`` for ``R_f``
        (e.g. ``{"positive": True, "initial": 1e-4, "hidden": [16, 16]}``),
        for the fouling-estimation twin (J03).
        """
        from pinnrl.pdes.custom import CustomPDE
        from pinnrl.pdes.inputs import TimeSeries

        ts = np.linspace(0.0, t_max, inputs_grid)
        series = {
            "W_in": TimeSeries(ts, [_as_fn(self.W_in)(x) for x in ts], name="W_in"),
            "X_in": TimeSeries(ts, [_as_fn(self.X_in)(x) for x in ts], name="X_in"),
            "T_s": TimeSeries(ts, [_as_fn(self.T_s)(x) for x in ts], name="T_s"),
        }
        params = {
            "v": self.velocity,
            "P": self.perimeter,
            "U": self.U,
            "lam": self.latent_heat,
            "T_sat": self.T_sat,
            "k_bpe": self.k_bpe,
        }
        unknown_functions = None
        if unknown_fouling is None:
            series["R_f"] = TimeSeries(ts, [_as_fn(self.R_f)(x) for x in ts], name="R_f")
        else:
            spec = {"inputs": ["t"], "positive": True, "initial": 1e-4, "hidden": [16, 16]}
            spec.update(unknown_fouling)
            unknown_functions = {"R_f": spec}
        flux = "(T_s - (T_sat + k_bpe*X))/(1/U + R_f)"
        ss = self.steady_state(inputs_grid, t=0.0)
        W0 = np.interp
        z_grid = ss["z"]

        def initial(x):
            import torch

            zz = x[:, 0].detach().cpu().numpy()
            W = torch.as_tensor(W0(zz, z_grid, ss["W"]), dtype=x.dtype).reshape(-1, 1)
            X = torch.as_tensor(np.interp(zz, z_grid, ss["X"]), dtype=x.dtype).reshape(-1, 1)
            return torch.cat([W, X], dim=1)

        W_scale = float(_as_fn(self.W_in)(0.0))
        X_scale = float(_as_fn(self.X_in)(0.0))
        return CustomPDE(
            [
                f"W_t + v*W_z + P*{flux}/lam",
                f"X_t + v*X_z - X*P*{flux}/(lam*W)",
            ],
            domain={"z": (0.0, self.length), "t": (0.0, t_max)},
            fields=["W", "X"],
            parameters=params,
            inputs=series,
            unknown_functions=unknown_functions,
            initial_condition=initial,
            boundary_conditions={
                "W": {"dirichlet": {"left": "W_in"}},
                "X": {"dirichlet": {"left": "X_in"}},
            },
            scales={"W": (W_scale, 0.2 * W_scale), "X": (X_scale, 0.5 * X_scale)},
            sensors=sensors,
            name=name,
        )
