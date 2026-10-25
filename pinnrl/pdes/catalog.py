"""A catalogue of equations written for :class:`~pinnrl.pdes.custom.CustomPDE`.

Each entry is the keyword dict ``CustomPDE(**spec)`` accepts: a residual as
text, the domain, parameters, and where one exists the exact solution, so a
problem is one line away::

    from pinnrl.pdes.catalog import CATALOG, from_catalog

    pde = from_catalog("fisher_kpp")
    pde = from_catalog("burgers", parameters={"nu": 0.1})   # override anything

The entries double as the test bed of the expression framework: every exact
solution in here is checked to zero its residual (``test_catalog.py``).
"""

from __future__ import annotations

import copy
from typing import Any

from .custom import CustomPDE

PI2 = "pi^2"

CATALOG: dict[str, dict[str, Any]] = {
    # ------------------------------------------------------------ parabolic
    "heat": dict(
        name="Heat equation",
        equations="u_t - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.1},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=f"exp(-alpha*{PI2}*t)*sin(pi*x)",
    ),
    "heat_2d": dict(
        name="Heat equation 2-D",
        equations="u_t - alpha*(u_xx + u_yy)",
        domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.05},
        initial_condition="sin(pi*x)*sin(pi*y)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=f"exp(-2*alpha*{PI2}*t)*sin(pi*x)*sin(pi*y)",
    ),
    "heat_source": dict(
        name="Heat equation with a source",
        equations="u_t - alpha*u_xx - q*sin(pi*x)",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.1, "q": 1.0},
        initial_condition="0",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=f"q/(alpha*{PI2})*(1 - exp(-alpha*{PI2}*t))*sin(pi*x)",
    ),
    "advection_diffusion": dict(
        name="Advection-diffusion",
        equations="u_t + v*u_x - D*u_xx",
        domain={"x": (0, 2), "t": (0, 1)},
        parameters={"v": 1.0, "D": 0.05},
        initial_condition="sin(pi*x)",
        boundary_conditions={"periodic": {}},
        exact_solution=f"exp(-D*{PI2}*t)*sin(pi*(x - v*t))",
    ),
    "reaction_diffusion_linear": dict(
        name="Linear reaction-diffusion",
        equations="u_t - D*u_xx - r*u",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"D": 0.1, "r": 0.5},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=f"exp((r - D*{PI2})*t)*sin(pi*x)",
    ),
    "fisher_kpp": dict(
        name="Fisher-KPP",
        equations="u_t - D*u_xx - r*u*(1 - u)",
        domain={"x": (-10, 10), "t": (0, 4)},
        parameters={"D": 1.0, "r": 1.0},
        initial_condition="1/(1 + exp(x))",
        boundary_conditions={"dirichlet": {"left": 1.0, "right": 0.0}},
    ),
    "allen_cahn": dict(
        name="Allen-Cahn",
        equations="u_t - eps2*u_xx - r*(u - u^3)",
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"eps2": 0.0001, "r": 5.0},
        initial_condition="x^2*cos(pi*x)",
        boundary_conditions={"dirichlet": {"value": -1.0}},
    ),
    "cahn_hilliard": dict(
        name="Cahn-Hilliard",
        # u_t = M (u^3 - u - kappa u_xx)_xx, with the outer derivative expanded
        equations="u_t - M*(6*u*u_x^2 + (3*u^2 - 1)*u_xx - kappa*u_xxxx)",
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"M": 1.0, "kappa": 0.01},
        initial_condition="tanh(x/sqrt(2*kappa))",
        boundary_conditions={"neumann": {"value": 0.0}},
    ),
    "burgers": dict(
        name="Burgers",
        equations="u_t + u*u_x - nu*u_xx",
        domain={"x": (-1, 1), "t": (0, 1)},
        parameters={"nu": 0.01 / 3.141592653589793},
        initial_condition="-sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
    ),
    "burgers_travelling_wave": dict(
        name="Burgers (travelling wave)",
        equations="u_t + u*u_x - nu*u_xx",
        domain={"x": (-5, 5), "t": (0, 2)},
        parameters={"nu": 0.1, "c": 1.0},
        initial_condition="c*(1 - tanh(c*x/(2*nu)))",
        boundary_conditions={"dirichlet": {"type": "exact"}},
        exact_solution="c*(1 - tanh(c*(x - c*t)/(2*nu)))",
    ),
    "porous_medium": dict(
        name="Porous medium (Barenblatt)",
        equations="u_t - 2*u_x^2 - 2*u*u_xx",  # u_t = (u^2)_xx
        domain={"x": (-3, 3), "t": (1, 2)},
        initial_condition="max(0, 1/4 - x^2/12)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="max(0, 1/4 - x^2/(12*t^(2/3)))/t^(1/3)",
    ),
    "black_scholes": dict(
        name="Black-Scholes (time to maturity)",
        equations="V_t - 0.5*sigma^2*S^2*V_SS - r*S*V_S + r*V",
        domain={"S": (1, 200), "t": (0, 1)},
        fields=["V"],
        parameters={"sigma": 0.2, "r": 0.05, "K": 100.0},
        initial_condition="max(S - K, 0)",
        boundary_conditions={"dirichlet": {"left": 0.0, "right": 100.0}},
    ),
    # ----------------------------------------------------------- hyperbolic
    "wave": dict(
        name="Wave equation",
        equations="u_tt - c^2*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"c": 1.0},
        initial_condition="sin(pi*x)",
        initial_velocity="0",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="cos(c*pi*t)*sin(pi*x)",
    ),
    "wave_2d": dict(
        name="Wave equation 2-D",
        equations="u_tt - c^2*(u_xx + u_yy)",
        domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
        parameters={"c": 1.0},
        initial_condition="sin(pi*x)*sin(pi*y)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="cos(sqrt(2)*c*pi*t)*sin(pi*x)*sin(pi*y)",
    ),
    "damped_wave": dict(
        name="Telegraph (damped wave)",
        equations="u_tt + 2*gamma*u_t - c^2*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"c": 1.0, "gamma": 0.5},
        initial_condition="sin(pi*x)",
        initial_velocity="-gamma*sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="exp(-gamma*t)*cos(sqrt(c^2*pi^2 - gamma^2)*t)*sin(pi*x)",
    ),
    "convection": dict(
        name="Linear convection",
        equations="u_t + v*u_x",
        domain={"x": (0, 2), "t": (0, 1)},
        parameters={"v": 1.0},
        initial_condition="sin(pi*x)",
        boundary_conditions={"periodic": {}},
        exact_solution="sin(pi*(x - v*t))",
    ),
    "klein_gordon": dict(
        name="Klein-Gordon (linear)",
        equations="u_tt - u_xx + m^2*u",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"m": 1.0},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="cos(sqrt(pi^2 + m^2)*t)*sin(pi*x)",
    ),
    "sine_gordon": dict(
        name="Sine-Gordon (kink)",
        equations="u_tt - u_xx + sin(u)",
        domain={"x": (-10, 10), "t": (0, 2)},
        parameters={"v": 0.5},
        initial_condition="4*atan(exp(x/sqrt(1 - v^2)))",
        initial_velocity="-4*v/sqrt(1 - v^2)*exp(x/sqrt(1 - v^2))/(1 + exp(2*x/sqrt(1 - v^2)))",
        boundary_conditions={"dirichlet": {"type": "exact"}},
        exact_solution="4*atan(exp((x - v*t)/sqrt(1 - v^2)))",
    ),
    # ----------------------------------------------------------- dispersive
    "kdv_soliton": dict(
        name="KdV (soliton)",
        equations="u_t + 6*u*u_x + u_xxx",
        domain={"x": (-15, 15), "t": (0, 2)},
        parameters={"c": 1.0},
        initial_condition="c/2/cosh(sqrt(c)/2*x)^2",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="c/2/cosh(sqrt(c)/2*(x - c*t))^2",
    ),
    "kdv_two_parameter": dict(
        name="KdV (two parameters)",
        equations="u_t + alpha*u*u_x + beta*u_xxx",
        domain={"x": (-15, 15), "t": (0, 2)},
        parameters={"alpha": 6.0, "beta": 1.0},
        initial_condition="3/alpha/cosh(sqrt(1/beta)/2*x)^2",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="3/alpha/cosh(sqrt(1/beta)/2*(x - t))^2",
    ),
    "beam": dict(
        name="Euler-Bernoulli beam",
        equations="w_tt + EI*w_xxxx",
        domain={"x": (0, 1), "t": (0, 1)},
        fields=["w"],
        parameters={"EI": 0.01},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="cos(sqrt(EI)*pi^2*t)*sin(pi*x)",
    ),
    "schrodinger": dict(
        name="Schrodinger (free particle, real and imaginary parts)",
        equations=["p_t + 0.5*q_xx - V*q", "q_t - 0.5*p_xx + V*p"],
        domain={"x": (0, 1), "t": (0, 1)},
        fields=["p", "q"],
        parameters={"V": 0.0},
        initial_condition={"p": "sin(pi*x)", "q": "0"},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution={"p": "cos(0.5*pi^2*t)*sin(pi*x)", "q": "-sin(0.5*pi^2*t)*sin(pi*x)"},
    ),
    # ------------------------------------------------------ higher dimension
    "heat_3d": dict(
        name="Heat equation 3-D",
        equations="u_t - alpha*(u_xx + u_yy + u_zz)",
        domain={"x": (0, 1), "y": (0, 1), "z": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.05},
        initial_condition="sin(pi*x)*sin(pi*y)*sin(pi*z)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=f"exp(-3*alpha*{PI2}*t)*sin(pi*x)*sin(pi*y)*sin(pi*z)",
    ),
    "poisson_4d": dict(
        name="Poisson 4-D",
        equations=(
            "u_x1x1 + u_x2x2 + u_x3x3 + u_x4x4 + 4*pi^2*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4)"
        ),
        domain={"x1": (0, 1), "x2": (0, 1), "x3": (0, 1), "x4": (0, 1)},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4)",
    ),
    "heat_5d": dict(
        name="Heat equation 5-D (6 network inputs)",
        equations="u_t - alpha*(u_x1x1 + u_x2x2 + u_x3x3 + u_x4x4 + u_x5x5)",
        domain={f"x{i}": (0, 1) for i in range(1, 6)} | {"t": (0, 1)},
        parameters={"alpha": 0.02},
        initial_condition="sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4)*sin(pi*x5)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=(
            f"exp(-5*alpha*{PI2}*t)*sin(pi*x1)*sin(pi*x2)*sin(pi*x3)*sin(pi*x4)*sin(pi*x5)"
        ),
    ),
    "hjb_linear_quadratic": dict(
        name="Linear-quadratic HJB (3 states; t is time to go)",
        # V_t = |grad V|^2/4 - |x|^2 with V(x, 0) = 0 has the closed form V = -tanh(t) |x|^2
        equations="V_t - 0.25*(V_x1^2 + V_x2^2 + V_x3^2) + (x1^2 + x2^2 + x3^2)",
        domain={"x1": (-1, 1), "x2": (-1, 1), "x3": (-1, 1), "t": (0, 1)},
        fields=["V"],
        initial_condition="0",
        boundary_conditions={"dirichlet": {"type": "exact"}},
        exact_solution="-tanh(t)*(x1^2 + x2^2 + x3^2)",
    ),
    # ---------------------------------------------------------------- steady
    "poisson": dict(
        name="Poisson",
        equations="u_xx + u_yy + 2*pi^2*sin(pi*x)*sin(pi*y)",
        domain={"x": (0, 1), "y": (0, 1)},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="sin(pi*x)*sin(pi*y)",
    ),
    "helmholtz": dict(
        name="Helmholtz",
        equations="u_xx + u_yy + k^2*u - (k^2 - 2*pi^2)*sin(pi*x)*sin(pi*y)",
        domain={"x": (0, 1), "y": (0, 1)},
        parameters={"k": 4.0},
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="sin(pi*x)*sin(pi*y)",
    ),
    "steady_convection_diffusion": dict(
        name="Steady convection-diffusion (boundary layer)",
        equations="Pe*u_x - u_xx",
        domain={"x": (0, 1)},
        parameters={"Pe": 10.0},
        boundary_conditions={"dirichlet": {"left": 0.0, "right": 1.0}},
        exact_solution="(exp(Pe*x) - 1)/(exp(Pe) - 1)",
    ),
    "bratu": dict(
        name="Bratu (nonlinear elliptic)",
        equations="u_xx + lam*exp(u)",
        domain={"x": (0, 1)},
        parameters={"lam": 1.0},
        boundary_conditions={"dirichlet": {"value": 0.0}},
    ),
    # ------------------------------------------------------------------ ODEs
    "logistic": dict(
        name="Logistic growth",
        equations="N_t - r*N*(1 - N/K)",
        domain={"t": (0, 10)},
        fields=["N"],
        parameters={"r": 1.0, "K": 10.0, "N0": 1.0},
        initial_condition="N0",
        exact_solution="K/(1 + (K/N0 - 1)*exp(-r*t))",
    ),
    "harmonic_oscillator": dict(
        name="Damped harmonic oscillator",
        equations="x_tt + 2*zeta*w0*x_t + w0^2*x",
        domain={"t": (0, 10)},
        fields=["x"],
        parameters={"zeta": 0.1, "w0": 2.0},
        initial_condition="1",
        initial_velocity="0",
        exact_solution=(
            "exp(-zeta*w0*t)*(cos(w0*sqrt(1 - zeta^2)*t)"
            " + zeta/sqrt(1 - zeta^2)*sin(w0*sqrt(1 - zeta^2)*t))"
        ),
    ),
    "pendulum": dict(
        name="Damped pendulum",
        equations="theta_tt + b*theta_t + (g/L)*sin(theta)",
        domain={"t": (0, 10)},
        fields=["theta"],
        parameters={"g": 9.81, "L": 1.0, "b": 0.1},
        initial_condition="0.5",
        initial_velocity="0",
    ),
    "lotka_volterra": dict(
        name="Lotka-Volterra",
        equations=["x_t - a*x + b*x*y", "y_t + c*y - d*x*y"],
        domain={"t": (0, 10)},
        fields=["x", "y"],
        parameters={"a": 1.1, "b": 0.4, "c": 0.4, "d": 0.1},
        initial_condition={"x": "10", "y": "10"},
    ),
    "van_der_pol": dict(
        name="Van der Pol oscillator",
        equations="x_tt - mu*(1 - x^2)*x_t + x",
        domain={"t": (0, 10)},
        fields=["x"],
        parameters={"mu": 1.0},
        initial_condition="2",
        initial_velocity="0",
    ),
    "lorenz": dict(
        name="Lorenz system",
        equations=["x_t - sigma*(y - x)", "y_t - x*(rho - z) + y", "z_t - x*y + beta*z"],
        domain={"t": (0, 2)},
        fields=["x", "y", "z"],
        parameters={"sigma": 10.0, "rho": 28.0, "beta": 8.0 / 3.0},
        initial_condition={"x": "1", "y": "1", "z": "1"},
    ),
    "sir": dict(
        name="SIR epidemic",
        equations=["S_t + beta*S*I", "I_t - beta*S*I + gamma*I", "R_t - gamma*I"],
        domain={"t": (0, 50)},
        fields=["S", "I", "R"],
        parameters={"beta": 0.3, "gamma": 0.1},
        initial_condition={"S": "0.99", "I": "0.01", "R": "0"},
    ),
    "cstr": dict(
        name="CSTR with a first-order exothermic reaction",
        # max(T, 0.1) keeps the Arrhenius factor finite while the network's T is still untrained
        equations=[
            "C_t - (Cin - C)/tau + k0*exp(-Ea/(R*max(T, 0.1)))*C",
            "T_t - (Tin - T)/tau - dH*k0*exp(-Ea/(R*max(T, 0.1)))*C + UA*(T - Tc)",
        ],
        domain={"t": (0, 5)},
        fields=["C", "T"],
        parameters={
            "Cin": 1.0,
            "Tin": 1.0,
            "tau": 1.0,
            "k0": 5.0,
            "Ea": 2.0,
            "R": 1.0,
            "dH": 0.5,
            "UA": 0.5,
            "Tc": 1.0,
        },
        initial_condition={"C": "1", "T": "1"},
    ),
    # --------------------------------------------------------------- systems
    "gray_scott": dict(
        name="Gray-Scott",
        equations=[
            "u_t - Du*(u_xx + u_yy) + u*v^2 - F*(1 - u)",
            "v_t - Dv*(v_xx + v_yy) - u*v^2 + (F + k)*v",
        ],
        domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
        fields=["u", "v"],
        parameters={"Du": 0.16, "Dv": 0.08, "F": 0.035, "k": 0.065},
        initial_condition={
            "u": "1 - 0.5*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.01)",
            "v": "0.25*exp(-((x - 0.5)^2 + (y - 0.5)^2)/0.01)",
        },
        boundary_conditions={"periodic": {}},
    ),
    "brusselator": dict(
        name="Brusselator",
        equations=["u_t - Du*u_xx - A + (B + 1)*u - u^2*v", "v_t - Dv*v_xx - B*u + u^2*v"],
        domain={"x": (0, 1), "t": (0, 5)},
        fields=["u", "v"],
        parameters={"A": 1.0, "B": 3.0, "Du": 0.01, "Dv": 0.1},
        initial_condition={"u": "A + 0.1*sin(2*pi*x)", "v": "B/A"},
        boundary_conditions={"periodic": {}},
    ),
    "shallow_water_linear": dict(
        name="Linearized shallow water",
        equations=["h_t + H*u_x", "u_t + g*h_x"],
        domain={"x": (0, 1), "t": (0, 1)},
        fields=["h", "u"],
        parameters={"H": 1.0, "g": 1.0},
        initial_condition={"h": "cos(2*pi*x)", "u": "0"},
        boundary_conditions={"periodic": {}},
        exact_solution={
            "h": "cos(2*pi*x)*cos(2*pi*sqrt(g*H)*t)",
            "u": "sqrt(g/H)*sin(2*pi*x)*sin(2*pi*sqrt(g*H)*t)",
        },
    ),
    "stokes_2d": dict(
        name="Stokes flow (Taylor-Green, unsteady)",
        equations=[
            "u_t + p_x - nu*(u_xx + u_yy)",
            "v_t + p_y - nu*(v_xx + v_yy)",
            "u_x + v_y",
        ],
        domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
        fields=["u", "v", "p"],
        parameters={"nu": 0.1},
        initial_condition={
            "u": "sin(2*pi*x)*cos(2*pi*y)",
            "v": "-cos(2*pi*x)*sin(2*pi*y)",
            "p": "0",
        },
        boundary_conditions={"periodic": {}},
        exact_solution={
            "u": "sin(2*pi*x)*cos(2*pi*y)*exp(-8*pi^2*nu*t)",
            "v": "-cos(2*pi*x)*sin(2*pi*y)*exp(-8*pi^2*nu*t)",
            "p": "0",
        },
    ),
}


def from_catalog(key: str, **overrides: Any) -> CustomPDE:
    """Build the catalogue entry ``key``; keyword ``overrides`` replace its fields.

    Raises:
        KeyError: If ``key`` is not in :data:`CATALOG`.
    """
    if key not in CATALOG:
        raise KeyError(f"Unknown catalogue entry {key!r}. Available: {', '.join(CATALOG)}")
    spec = copy.deepcopy(CATALOG[key])
    spec.update(overrides)
    return CustomPDE.from_dict(spec)
