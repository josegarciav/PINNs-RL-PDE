"""Physics consistency: the theory behind every PDE, checked numerically.

Each test pins one mathematical claim the PDE layer relies on:

* every configured exact solution satisfies its PDE, its initial condition and
  its boundary conditions;
* every residual implements its equation (checked on manufactured solutions,
  independently of any exact solution);
* every configured parameter actually enters the residual;
* derivatives, boundary-condition semantics, the initial-velocity term and
  ``validate()`` behave as documented;
* closed forms have their known properties (soliton speed, decay rates,
  put-call parity, energy conservation, small-angle limit).

If a change breaks one of these, it changed the physics, not just the code.
"""

import math

import numpy as np
import pytest
import torch
import yaml

from pinnrl.benchmarks.reference import burgers_cole_hopf
from pinnrl.config import DEFAULT_CONFIG_PATH, TrainingConfig
from pinnrl.pdes import PDEConfig, get_pde_spec

torch.manual_seed(0)
CPU = torch.device("cpu")

with open(DEFAULT_CONFIG_PATH) as _f:
    _YAML = yaml.safe_load(_f)
PDE_CONFIGS = _YAML["pde_configs"]
TRAINING = _YAML["training"]

# Exact solutions written as torch expressions (autograd can differentiate them).
AUTOGRAD_EXACT = [
    "heat",
    "heat_2d",
    "wave",
    "convection",
    "kdv",
    "allen_cahn",
    "cahn_hilliard",
    "black_scholes",
]
# Exact solutions evaluated numerically (quadrature / ODE solve).
NUMERIC_EXACT = ["burgers", "pendulum"]


def build(key, **overrides):
    """The PDE ``key`` exactly as config.yaml ships it (plus overrides)."""
    cfg = {**PDE_CONFIGS[key], **overrides}
    return get_pde_spec(key).cls(
        PDEConfig(
            name=cfg.get("name", key),
            domain=cfg["domain"],
            time_domain=cfg["time_domain"],
            parameters=dict(cfg.get("parameters", {})),
            boundary_conditions=cfg.get("boundary_conditions", {}),
            initial_condition=dict(cfg.get("initial_condition", {})),
            exact_solution=dict(cfg.get("exact_solution", {})),
            dimension=cfg.get("dimension", 1),
            device=CPU,
            training=TrainingConfig.from_dict(TRAINING),
        )
    )


class ExactModel(torch.nn.Module):
    """A 'network' that returns the PDE's exact solution.

    The zero-weight term keeps an autograd path to the inputs, so boundary
    terms that differentiate (neumann, periodic) work for numeric solutions
    too (their spatial derivative is then zero, which is only valid where the
    solution does not depend on x — the pendulum).
    """

    def __init__(self, pde):
        super().__init__()
        self.pde = pde
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def forward(self, inp):
        d = self.pde.dimension
        return self.pde.exact_solution(inp[:, :d], inp[:, d:]) + 0.0 * inp.sum(1, keepdim=True)


class Formula(torch.nn.Module):
    """Wrap ``f(x, t)`` (x: (N, d), t: (N, 1)) as a model."""

    def __init__(self, f, d=1):
        super().__init__()
        self.f, self.d = f, d

    def forward(self, inp):
        return self.f(inp[:, : self.d], inp[:, self.d :])


def interior(pde, n=256, margin=0.05):
    """Random points strictly inside the space-time domain."""
    lo = torch.tensor([float(b[0]) for b in pde.domain])
    hi = torch.tensor([float(b[1]) for b in pde.domain])
    x = lo + (hi - lo) * (margin + (1 - 2 * margin) * torch.rand(n, pde.dimension))
    t0, t1 = float(pde.time_domain[0]), float(pde.time_domain[1])
    t = t0 + (t1 - t0) * (margin + (1 - 2 * margin) * torch.rand(n, 1))
    return x, t


def rel(err, ref):
    return (err.abs().mean() / (ref.abs().mean() + 1e-12)).item()


# ===========================================================================
# 1. Every shipped exact solution solves its own problem
# ===========================================================================


def test_every_config_is_covered():
    assert set(AUTOGRAD_EXACT + NUMERIC_EXACT) == set(PDE_CONFIGS)


@pytest.mark.parametrize("key", AUTOGRAD_EXACT)
def test_exact_solution_satisfies_pde(key):
    pde = build(key)
    model = ExactModel(pde)
    x, t = interior(pde)
    residual = pde.compute_residual(model, x, t).detach()
    u = model(torch.cat([x, t], 1)).detach()
    assert rel(residual, u) < 1e-4


def test_burgers_cole_hopf_satisfies_pde():
    """Central differences of the Cole-Hopf solution satisfy u_t + u u_x = nu u_xx."""
    nu = float(PDE_CONFIGS["burgers"]["parameters"]["nu"])
    x, t, h = np.linspace(-0.95, 0.95, 39), 0.4, 1e-3
    f = lambda a, b: burgers_cole_hopf(a, b, nu)  # noqa: E731
    u = f(x, t)
    # Fourth-order central differences (truncation ~h^4).
    u_t = (-f(x, t + 2 * h) + 8 * f(x, t + h) - 8 * f(x, t - h) + f(x, t - 2 * h)) / (12 * h)
    u_x = (-f(x + 2 * h, t) + 8 * f(x + h, t) - 8 * f(x - h, t) + f(x - 2 * h, t)) / (12 * h)
    u_xx = (-f(x + 2 * h, t) + 16 * f(x + h, t) - 30 * u + 16 * f(x - h, t) - f(x - 2 * h, t)) / (
        12 * h**2
    )
    # Relative to the size of the terms (|u u_x| ~ 1e2 near the forming shock).
    scale = np.abs(u * u_x).max()
    assert np.abs(u_t + u * u_x - nu * u_xx).max() < 1e-6 * scale


def test_pendulum_trajectory_satisfies_ode():
    """θ'' + b θ' + (g/L) sin θ = 0 along the reference trajectory."""
    pde = build("pendulum")
    t = torch.linspace(0.5, 9.5, 91).reshape(-1, 1).double()
    h = 1e-3
    theta = lambda s: pde.exact_solution(torch.zeros_like(s), s)  # noqa: E731
    th, thp, thm = theta(t), theta(t + h), theta(t - h)
    th_t = (thp - thm) / (2 * h)
    th_tt = (thp - 2 * th + thm) / h**2
    residual = th_tt + pde.damping * th_t + (pde.g / pde.L) * torch.sin(th)
    assert residual.abs().max().item() < 1e-5


@pytest.mark.parametrize("key", AUTOGRAD_EXACT + NUMERIC_EXACT)
def test_exact_solution_matches_initial_condition(key):
    pde = build(key)
    x, _ = interior(pde, margin=0.0)
    t0 = torch.full((x.shape[0], 1), float(pde.time_domain[0]))
    u0 = pde.initial_condition(x, t0)
    assert rel(pde.exact_solution(x, t0) - u0, u0) < 1e-6


@pytest.mark.parametrize("key", AUTOGRAD_EXACT + NUMERIC_EXACT)
def test_exact_solution_satisfies_boundary_conditions(key):
    pde = build(key)
    model = ExactModel(pde)
    scale = model(torch.cat(interior(pde), 1)).detach().abs().mean().item()
    assert pde._boundary_loss(model).item() < 1e-6 * max(scale, 1.0) ** 2


@pytest.mark.parametrize("key", [k for k in AUTOGRAD_EXACT if k != "black_scholes"])
def test_exact_solution_has_zero_initial_loss(key):
    """IC (and initial velocity for second-order PDEs) are met by the exact solution."""
    pde = build(key)
    assert pde._initial_loss(ExactModel(pde)).item() < 1e-10


def test_pendulum_starts_at_rest():
    pde = build("pendulum")
    h = 1e-4
    theta = lambda s: pde.exact_solution(torch.zeros(1, 1), torch.full((1, 1), s))  # noqa: E731
    velocity = (theta(h) - theta(0.0)) / h
    assert abs(velocity.item() - pde.config.initial_condition["velocity"]) < 1e-2


# ===========================================================================
# 2. Residuals implement their equations (manufactured solutions)
# ===========================================================================

# key -> (u(x, t), expected residual(x, t, pde)) on plain polynomials.
MANUFACTURED = {
    "heat": (lambda x, t: x**2 * t + t**2, lambda x, t, p: x**2 + 2 * t - p.alpha * 2 * t),
    "wave": (lambda x, t: x**2 * t**2, lambda x, t, p: 2 * x**2 - p.c**2 * 2 * t**2),
    "burgers": (lambda x, t: x * t, lambda x, t, p: x + x * t**2),
    "convection": (lambda x, t: x * t, lambda x, t, p: x + p.velocity[0] * t),
    "kdv": (
        lambda x, t: x**3 * t,
        lambda x, t, p: x**3 + p.alpha * 3 * x**5 * t**2 + p.beta * 6 * t,
    ),
    "allen_cahn": (
        lambda x, t: x * t,
        lambda x, t, p: x - p.reaction_rate * (x * t - (x * t) ** 3),
    ),
    "cahn_hilliard": (
        lambda x, t: x**2 * t,
        lambda x, t, p: x**2 - p.mobility * (30 * x**4 * t**3 - 2 * t),
    ),
    "black_scholes": (
        lambda x, t: x**2 * t,
        lambda x, t, p: x**2 - p.sigma**2 * x**2 * t - p.r * x**2 * t,
    ),
    "pendulum": (
        lambda x, t: t**2 + 0 * x,
        lambda x, t, p: 2 + p.damping * 2 * t + (p.g / p.L) * torch.sin(t**2),
    ),
}


@pytest.mark.parametrize("key", sorted(MANUFACTURED))
def test_residual_on_manufactured_solution(key):
    pde = build(key)
    u, expected = MANUFACTURED[key]
    x, t = 0.5 * torch.rand(64, 1), 0.5 * torch.rand(64, 1)
    residual = pde.compute_residual(Formula(u), x, t).detach()
    assert torch.allclose(residual, expected(x, t, pde), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("key", sorted(PDE_CONFIGS))
def test_every_configured_parameter_enters_the_residual(key):
    """Perturbing any configured parameter changes the residual of a generic field."""
    field = Formula(
        lambda x, t: torch.sin(1.3 * x.sum(1, keepdim=True) + 0.7 * t) * (1 + t),
        d=PDE_CONFIGS[key].get("dimension", 1),
    )
    base = build(key)
    x, t = interior(base, n=64)
    r0 = base.compute_residual(field, x, t).detach()
    for name, value in PDE_CONFIGS[key].get("parameters", {}).items():
        params = dict(PDE_CONFIGS[key]["parameters"])
        params[name] = [v * 1.1 for v in value] if isinstance(value, list) else value * 1.1
        r1 = build(key, parameters=params).compute_residual(field, x, t).detach()
        assert not torch.allclose(r0, r1), f"{key}: parameter {name!r} has no effect"


# ===========================================================================
# 3. Derivative operators
# ===========================================================================


def test_derivatives_1d_match_closed_form():
    pde = build("heat")
    u = Formula(lambda x, t: torch.sin(2 * x) * torch.exp(3 * t))
    x, t = torch.rand(32, 1), torch.rand(32, 1)
    d = pde.compute_derivatives(
        u, x, t, temporal_derivatives=[1, 2], spatial_derivatives=[1, 2, 3, 4]
    )
    s, c, e = torch.sin(2 * x), torch.cos(2 * x), torch.exp(3 * t)
    expected = {
        "u": s * e,
        "dt": 3 * s * e,
        "dt2": 9 * s * e,
        "dx": 2 * c * e,
        "dx2": -4 * s * e,
        "dx3": -8 * c * e,
        "dx4": 16 * s * e,
        "laplacian": -4 * s * e,
    }
    for key, value in expected.items():
        assert torch.allclose(d[key], value, atol=1e-4), key


def test_derivatives_2d_match_closed_form():
    pde = build("heat_2d")
    u = Formula(lambda x, t: torch.sin(x[:, :1]) * torch.cos(2 * x[:, 1:]) * t, d=2)
    x, t = torch.rand(32, 2), torch.rand(32, 1)
    d = pde.compute_derivatives(u, x, t, temporal_derivatives=[1], spatial_derivatives=[1, 2])
    s1, c1 = torch.sin(x[:, :1]), torch.cos(x[:, :1])
    s2, c2 = torch.sin(2 * x[:, 1:]), torch.cos(2 * x[:, 1:])
    assert torch.allclose(d["dt"], s1 * c2, atol=1e-5)
    assert torch.allclose(d["dx1"], c1 * c2 * t, atol=1e-5)
    assert torch.allclose(d["dx2"], -2 * s1 * s2 * t, atol=1e-5)
    assert torch.allclose(d["dx1x1"], -s1 * c2 * t, atol=1e-5)
    assert torch.allclose(d["dx2x2"], -4 * s1 * c2 * t, atol=1e-5)
    assert torch.allclose(d["laplacian"], -5 * s1 * c2 * t, atol=1e-5)


# ===========================================================================
# 4. Boundary-condition, initial-condition and validate() semantics
# ===========================================================================


def _bc_pde(boundary_conditions, domain=((0.0, 1.0),), exact_solution=None):
    return build(
        "heat",
        domain=[list(b) for b in domain],
        boundary_conditions=boundary_conditions,
        exact_solution=exact_solution or {},
    )


def test_dirichlet_value():
    pde = _bc_pde({"dirichlet": {"value": 2.0}})
    assert pde._boundary_loss(Formula(lambda x, t: 2.0 + 0 * x)).item() < 1e-12
    assert pde._boundary_loss(Formula(lambda x, t: 3.0 + 0 * x)).item() == pytest.approx(1.0)


def test_dirichlet_left_right():
    pde = _bc_pde({"dirichlet": {"left": -1.0, "right": 1.0}}, domain=((-1.0, 1.0),))
    assert pde._boundary_loss(Formula(lambda x, t: x)).item() < 1e-12
    assert pde._boundary_loss(Formula(lambda x, t: -x)).item() == pytest.approx(4.0)


def test_neumann_constrains_the_derivative():
    assert (
        _bc_pde({"neumann": {"value": 3.0}})._boundary_loss(Formula(lambda x, t: 3 * x)).item()
        < 1e-10
    )
    zero = _bc_pde({"neumann": {"type": "zero"}})
    assert zero._boundary_loss(Formula(lambda x, t: 3 * x)).item() == pytest.approx(9.0)
    assert zero._boundary_loss(Formula(lambda x, t: 5.0 + 0 * x)).item() < 1e-12


def test_periodic_matches_value_and_slope():
    pde = _bc_pde({"periodic": {}})
    assert pde._boundary_loss(Formula(lambda x, t: torch.sin(2 * torch.pi * x) * t)).item() < 1e-10
    # u = x: values differ by 1 across the domain, slopes agree.
    assert pde._boundary_loss(Formula(lambda x, t: x)).item() == pytest.approx(0.5)


def test_boundary_points_cover_every_face_in_2d():
    pde = build("heat_2d")
    faces = pde._seeded_points("boundary", 10)
    assert [axis for axis, *_ in faces] == [0, 1]
    for axis, x_lo, x_hi, _ in faces:
        assert torch.all(x_lo[:, axis] == 0.0) and torch.all(x_hi[:, axis] == 1.0)


@pytest.mark.parametrize(
    "bcs",
    [
        {"robin": {}},
        {"dirichlet": {"type": "periodic"}},
        {"dirichlet": {"type": "custom"}},
        {"neumann": {"type": "x"}},
    ],
)
def test_unknown_boundary_conditions_raise(bcs):
    with pytest.raises(ValueError):
        _bc_pde(bcs)


def test_exact_dirichlet_needs_an_exact_solution():
    with pytest.raises(ValueError, match="exact_solution"):
        _bc_pde({"dirichlet": {"type": "exact"}})


def test_boundary_and_initial_point_counts_follow_the_config():
    pde = build("wave")
    pde.config.training.num_boundary_points = 7
    pde.config.training.num_initial_points = 5
    model = ExactModel(pde)
    pde._boundary_loss(model)
    pde._initial_loss(model)
    assert ("boundary", 7) in pde._fixed_points and ("initial", 5) in pde._fixed_points
    assert pde._fixed_points[("initial", 5)][0].shape == (5, 1)


def test_initial_velocity_is_enforced_for_second_order_pdes():
    """A traveling wave matches u(x, 0) of the standing wave but not u_t(x, 0) = 0."""
    pde = build("wave")
    standing = ExactModel(pde)
    k = pde.config.initial_condition["frequency"]
    moving = Formula(lambda x, t: torch.sin(k * torch.pi * (x - t)))
    assert pde._initial_loss(standing).item() < 1e-12
    assert pde._initial_loss(moving).item() > 1.0


def test_initial_condition_is_not_applied_on_the_boundary():
    """A model that is exact at t0 but violates the IC elsewhere in time only pays the BC."""
    pde = build("allen_cahn")
    model = ExactModel(pde)
    assert pde._initial_loss(model).item() < 1e-12
    assert pde._boundary_loss(model).item() < 1e-12


def test_loss_is_deterministic():
    pde = build("heat")
    model = Formula(lambda x, t: torch.sin(x) * t)
    x, t = interior(pde, n=32)
    a, b = pde.compute_loss(model, x, t), pde.compute_loss(model, x, t)
    for key in ("boundary", "initial", "total"):
        assert a[key].item() == b[key].item()


def test_validate_reports_relative_l2():
    pde = build("heat")
    exact = ExactModel(pde)
    metrics = pde.validate(exact, num_points=400)
    assert set(metrics) == {"l2_error", "mse", "max_error", "mean_error"}
    assert metrics["l2_error"] < 1e-6
    doubled = Formula(lambda x, t: 2 * pde.exact_solution(x, t))
    assert pde.validate(doubled, num_points=400)["l2_error"] == pytest.approx(1.0, rel=1e-5)


def test_no_exact_solution_means_none():
    pde = build(
        "allen_cahn",
        exact_solution={},
        initial_condition={"type": "x2_cos"},
        boundary_conditions={"dirichlet": {"value": -1.0}},
    )
    x, t = interior(pde, n=4)
    assert pde.exact_solution(x, t) is None
    with pytest.raises(ValueError, match="no exact solution"):
        pde.validate(Formula(lambda x, t: x), num_points=16)


# ===========================================================================
# 5. Properties of the closed forms
# ===========================================================================


def test_heat_decay_rate():
    for key in ("heat", "heat_2d"):
        pde = build(key)
        x, t = interior(pde, n=16)
        ratio = pde.exact_solution(x, t) / pde.exact_solution(x, torch.zeros_like(t))
        if key == "heat":
            k = pde.config.initial_condition["frequency"]
            rate = pde.alpha * sum((2 * math.pi * k / (hi - lo)) ** 2 for lo, hi in pde.domain)
        else:
            ic = pde.config.initial_condition
            rate = pde.alpha * math.pi**2 * (ic["frequency_x"] ** 2 + ic["frequency_y"] ** 2)
        assert torch.allclose(ratio, torch.exp(-rate * t), rtol=1e-5)


def test_kdv_soliton_travels_at_its_speed():
    pde = build("kdv")
    x, t = interior(pde, n=64)
    c, dt = pde.speed, 0.7
    assert torch.allclose(
        pde.exact_solution(x + c * dt, t + dt), pde.exact_solution(x, t), atol=1e-6
    )


def test_convection_transports_the_initial_profile():
    pde = build("convection")
    x, t = interior(pde, n=64)
    shifted = x - pde.velocity[0] * t
    assert torch.allclose(pde.exact_solution(x, t), pde.initial_condition(shifted, t), atol=1e-6)


def test_black_scholes_put_call_parity():
    call = build("black_scholes")
    put_ic = {**PDE_CONFIGS["black_scholes"]["initial_condition"], "option_type": "put"}
    put = build("black_scholes", initial_condition=put_ic)
    x, t = interior(call, n=64)
    strike = call.config.initial_condition["strike"]
    parity = x - strike * torch.exp(-call.r * t)
    assert torch.allclose(call.exact_solution(x, t) - put.exact_solution(x, t), parity, atol=1e-3)


def test_black_scholes_value_at_zero_price():
    pde = build("black_scholes")
    t = torch.rand(8, 1) + 0.1
    assert torch.allclose(pde.exact_solution(torch.zeros_like(t), t), torch.zeros_like(t))


def test_pendulum_small_angle_limit():
    theta0 = 1e-3
    ic = {**PDE_CONFIGS["pendulum"]["initial_condition"], "initial_angle": theta0}
    pde = build("pendulum", initial_condition=ic, parameters={"g": 9.81, "L": 1.0, "damping": 0.0})
    t = torch.linspace(0, 10, 50).reshape(-1, 1).double()
    approx = theta0 * torch.cos(math.sqrt(9.81) * t)
    assert (pde.exact_solution(torch.zeros_like(t), t) - approx).abs().max().item() < 1e-8


def test_pendulum_conserves_energy_without_damping():
    pde = build("pendulum", parameters={"g": 9.81, "L": 1.0, "damping": 0.0})
    t = torch.linspace(0.1, 9.9, 50).reshape(-1, 1).double()
    h = 1e-5
    theta = lambda s: pde.exact_solution(torch.zeros_like(s), s)  # noqa: E731
    th, th_t = theta(t), (theta(t + h) - theta(t - h)) / (2 * h)
    energy = 0.5 * pde.L**2 * th_t**2 + pde.g * pde.L * (1 - torch.cos(th))
    assert (energy.max() - energy.min()).item() < 1e-6 * energy.mean().item()


def test_allen_cahn_and_cahn_hilliard_kinks_are_stationary():
    for key in ("allen_cahn", "cahn_hilliard"):
        pde = build(key)
        x, t = interior(pde, n=32)
        assert torch.allclose(pde.exact_solution(x, t), pde.exact_solution(x, torch.zeros_like(t)))


def test_cahn_hilliard_random_ic_is_reproducible():
    ic = {"type": "random", "amplitude": 0.1, "seed": 3}
    a = build("cahn_hilliard", initial_condition=ic).initial_condition
    b = build("cahn_hilliard", initial_condition=ic).initial_condition
    x = torch.rand(32, 1)
    assert torch.equal(a(x, None), b(x, None))
    assert a(x, None).abs().max().item() <= 0.1 + 1e-12


# ===========================================================================
# 6. Domain handling and collocation sampling
# ===========================================================================


@pytest.mark.parametrize("domain", [(-1.0, 1.0), [-1.0, 1.0], [(-1.0, 1.0)], [[-1.0, 1.0]]])
def test_domain_spellings_are_equivalent(domain):
    assert build("heat", domain=domain).domain == [(-1.0, 1.0)]


@pytest.mark.parametrize("key", ["heat", "heat_2d", "black_scholes"])
@pytest.mark.parametrize("strategy", ["uniform", "stratified"])
@pytest.mark.parametrize("n", [1, 37, 500])
def test_collocation_points_count_and_bounds(key, strategy, n):
    pde = build(key)
    x, t = pde.generate_collocation_points(n, strategy=strategy)
    assert x.shape == (n, pde.dimension) and t.shape == (n, 1)
    for d, (lo, hi) in enumerate(pde.domain):
        assert x[:, d].min() >= lo and x[:, d].max() <= hi
    assert t.min() >= pde.time_domain[0] and t.max() <= pde.time_domain[1]


def test_uniform_points_fill_a_wide_domain():
    """Jitter scales with the axis length (Black-Scholes prices span [0, 200])."""
    x, _ = build("black_scholes").generate_collocation_points(400)
    assert x.min() < 20 and x.max() > 180
