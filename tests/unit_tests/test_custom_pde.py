"""User-defined equations: expression parsing, derivatives, parity with the shipped PDEs, systems."""

import math

import pytest
import torch
import yaml

from pinnrl import CustomPDE, Fields
from pinnrl.pdes import BurgersEquation, HeatEquation, PDEConfig, PendulumEquation
from pinnrl.pdes.symbolic import compile_expression, split_derivative
from pinnrl.training.train import build_config_dict, create_pde

torch.manual_seed(0)
NET1 = torch.nn.Sequential(torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 1))
NET2 = torch.nn.Sequential(torch.nn.Linear(3, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2))
X = torch.rand(40, 1)
X2 = torch.rand(40, 2)
T = torch.rand(40, 1)


# ------------------------------------------------------------------ expressions
def test_split_derivative_reads_subscripts_longest_coordinate_first():
    assert split_derivative("u_xxt", ["u"], ["x", "t"]) == ("u", ("x", "x", "t"))
    assert split_derivative("u_theta", ["u"], ["theta", "t"]) == ("u", ("theta",))
    assert split_derivative("u_ttheta", ["u"], ["theta", "t"]) == ("u", ("t", "theta"))
    assert split_derivative("nu", ["u"], ["x", "t"]) is None
    assert split_derivative("u_", ["u"], ["x", "t"]) is None


def test_compile_expression_classifies_names_and_evaluates():
    c = compile_expression("u_t + u*u_x - nu*u_xx + sin(pi*x) + abs(u) + 2^2", ["u"], ["x", "t"])
    assert c.fields == ("u",)
    assert set(c.derivatives) == {("u", ("t",)), ("u", ("x",)), ("u", ("x", "x"))}
    assert c.coordinates == ("x",)
    assert c.parameters == ("nu",)
    assert c.time_order == 1
    ones = torch.ones(3, 1)
    out = c(
        {"u": ones, "u_t": 0 * ones, "u_x": 2 * ones, "u_xx": 3 * ones, "x": 0.5 * ones, "nu": 0.1},
        like=ones,
    )
    assert torch.allclose(out, torch.full((3, 1), 2 - 0.3 + 1 + 1 + 4))


def test_compile_expression_constants_broadcast_and_e_is_a_free_name():
    like = torch.zeros(5, 1)
    assert compile_expression("0", ["u"], ["x", "t"])({}, like).shape == (5, 1)
    assert compile_expression("u_x - u_x", ["u"], ["x", "t"])({"u_x": like}, like).shape == (5, 1)
    assert compile_expression("E*beta", ["u"], ["x", "t"]).parameters == ("E", "beta")


def test_compile_expression_errors():
    with pytest.raises(ValueError, match="Cannot parse"):
        compile_expression("u_t +* u", ["u"], ["x", "t"])
    with pytest.raises(ValueError, match="Cannot parse|Unknown function"):
        compile_expression("foo(u)", ["u"], ["x", "t"])


# ------------------------------------------------------------------- Fields
def test_fields_derivatives_match_autograd_of_a_known_function():
    class Known(torch.nn.Module):
        def forward(self, inp):
            x, t = inp[:, 0:1], inp[:, 1:2]
            return torch.sin(x) * torch.exp(-t)

    f = Fields(Known(), X, T, ["u"], ["x", "t"], torch.device("cpu"))
    x, t = f.x, f.t
    assert torch.allclose(f.u_x, torch.cos(x) * torch.exp(-t), atol=1e-6)
    assert torch.allclose(f.u_xx, -torch.sin(x) * torch.exp(-t), atol=1e-6)
    assert torch.allclose(f.u_t, -torch.sin(x) * torch.exp(-t), atol=1e-6)
    assert torch.allclose(f.u_xt, -torch.cos(x) * torch.exp(-t), atol=1e-6)
    assert torch.allclose(f.d("u", "t", "x"), f.u_xt)  # order of differentiation is free
    assert torch.allclose(f.laplacian("u"), f.u_xx)
    with pytest.raises(AttributeError, match="not a field"):
        _ = f.w_x


# ---------------------------------------------------------- parity with shipped
def test_custom_heat_matches_heat_equation_residual():
    heat = HeatEquation(PDEConfig("heat", [[0, 2]], (0, 1), {"alpha": 0.01}, {}, {}, {}))
    custom = CustomPDE(
        "u_t - alpha*u_xx", domain={"x": (0, 2), "t": (0, 1)}, parameters={"alpha": 0.01}
    )
    assert torch.allclose(
        heat.compute_residual(NET1, X, T), custom.compute_residual(NET1, X, T), atol=1e-6
    )


def test_custom_burgers_matches_burgers_residual():
    ref = BurgersEquation(PDEConfig("b", [[-1, 1]], (0, 1), {"nu": 0.1}, {}, {}, {}))
    custom = CustomPDE(
        "u_t + u*u_x - nu*u_xx", domain={"x": (-1, 1), "t": (0, 1)}, parameters={"nu": 0.1}
    )
    assert torch.allclose(
        ref.compute_residual(NET1, X, T), custom.compute_residual(NET1, X, T), atol=1e-6
    )


def test_custom_ode_matches_pendulum_and_is_second_order_in_time():
    ref = PendulumEquation(
        PDEConfig(
            "p",
            [[0, 1]],
            (0, 10),
            {"g": 9.81, "L": 1.0, "damping": 0.1},
            {},
            {"type": "fixed", "value": 0.5},
            {},
        )
    )
    ode = CustomPDE(
        "theta_tt + b*theta_t + (g/L)*sin(theta)",
        domain={"t": (0, 10)},
        fields=["theta"],
        parameters={"g": 9.81, "L": 1.0, "b": 0.1},
        initial_condition="0.5",
    )
    assert ode.is_ode and ode.time_order == 2 and ode.config.input_dim == 2
    assert torch.allclose(
        ref.compute_residual(NET1, X, T), ode.compute_residual(NET1, X, T), atol=1e-5
    )
    assert torch.allclose(ref._initial_loss(NET1), ode._initial_loss(NET1), atol=1e-6)


# ------------------------------------------------------------------- systems
def _gray_scott(residual):
    return CustomPDE(
        residual,
        domain={"x": (0, 1), "y": (0, 1), "t": (0, 1)},
        fields=["u", "v"],
        parameters={"Du": 0.16, "Dv": 0.08, "F": 0.035, "k": 0.065},
        initial_condition={
            "u": "1 - 0.5*exp(-((x-0.5)^2 + (y-0.5)^2)/0.01)",
            "v": "0.25*exp(-((x-0.5)^2 + (y-0.5)^2)/0.01)",
        },
        boundary_conditions={"periodic": {}},
    )


def test_system_text_and_callable_agree_and_train_pieces_work():
    text = _gray_scott(
        ["u_t - Du*(u_xx + u_yy) + u*v^2 - F*(1 - u)", "v_t - Dv*(v_xx + v_yy) - u*v^2 + (F + k)*v"]
    )

    def residual(f, p):
        return [
            f.u_t - p["Du"] * f.laplacian("u") + f.u * f.v**2 - p["F"] * (1 - f.u),
            f.v_t - p["Dv"] * f.laplacian("v") - f.u * f.v**2 + (p["F"] + p["k"]) * f.v,
        ]

    r_text = text.compute_residual(NET2, X2, T)
    assert r_text.shape == (40, 2)
    assert torch.allclose(r_text, _gray_scott(residual).compute_residual(NET2, X2, T), atol=1e-6)
    losses = text.compute_loss(NET2, X2, T)
    assert all(torch.isfinite(v) for v in losses.values())
    x, t = text.generate_collocation_points(64, strategy="residual_based", model=NET2)
    assert x.shape == (64, 2) and t.shape == (64, 1)
    again = CustomPDE.from_dict(text.to_dict())
    assert torch.allclose(again.compute_residual(NET2, X2, T), r_text)


def test_steady_problem_has_no_time_and_validates_all_components():
    poisson = CustomPDE(
        "u_xx + u_yy + 2*pi^2*sin(pi*x)*sin(pi*y)",
        domain={"x": (0, 1), "y": (0, 1)},
        exact_solution="sin(pi*x)*sin(pi*y)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
    )
    assert (
        not poisson.time_dependent and poisson.time_order == 0 and poisson.initial_condition is None
    )
    assert poisson.compute_residual(NET2, X2, T).shape == (40, 1)
    metrics = poisson.validate(NET2)
    assert 0 < metrics["l2_error"] < 10


def test_identifiable_parameter_is_in_the_graph():
    inv = CustomPDE(
        "u_t - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.05},
        trainable_parameters=["alpha"],
        parameter_initial_guesses={"alpha": 0.5},
    )
    inv.compute_residual(NET1, X, T).pow(2).mean().backward()
    assert inv._trainable_params["alpha"].grad is not None
    assert inv.get_trainable_parameter_values() == {"alpha": 0.5}


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(equations="u_t - alpha*u_xx", domain={"x": (0, 1), "t": (0, 1)}), "alpha"),
        (dict(equations="u_t - u_zz", domain={"x": (0, 1), "t": (0, 1)}), "u_zz"),
        (
            dict(equations=["u_t"], domain={"x": (0, 1), "t": (0, 1)}, fields=["u", "v"]),
            "one equation per field",
        ),
        (
            dict(equations="u_t", domain={"x": (0, 1), "t": (0, 1)}, initial_condition="u + 1"),
            "must not depend",
        ),
    ],
)
def test_bad_specs_raise(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CustomPDE(**kwargs)


# --------------------------------------------------------------------- training
def test_build_trainer_solves_heat_to_a_few_percent():
    torch.manual_seed(0)
    pde = CustomPDE(
        "u_t - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        parameters={"alpha": 0.1},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",
    )
    trainer = pde.build_trainer(
        "fourier",
        num_epochs=150,
        num_collocation_points=512,
        batch_size=512,
        num_boundary_points=128,
        num_initial_points=128,
        learning_rate=2e-3,
        early_stopping={"enabled": False},
        loss_weights={"residual": 1, "boundary": 10, "initial": 10},
    )
    trainer.train(progress=False)
    assert pde.validate(trainer.model)["l2_error"] < 0.1


def test_cli_config_round_trips_a_yaml_spec(tmp_path):
    spec = {
        "name": "Fisher-KPP",
        "equations": "u_t - D*u_xx - r*u*(1 - u)",
        "domain": {"x": [-10, 10], "t": [0, 4]},
        "parameters": {"D": 1.0, "r": 1.0},
        "initial_condition": "1/(1 + exp(x))",
        "boundary_conditions": {"dirichlet": {"left": 1.0, "right": 0.0}},
    }
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    from pinnrl.config import DEFAULT_CONFIG_PATH

    with open(DEFAULT_CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    cfg = build_config_dict(
        base, None, "fourier", custom=yaml.safe_load((tmp_path / "spec.yaml").read_text())
    )
    assert (
        cfg["pde"]["name"] == "Fisher-KPP"
        and cfg["pde"]["custom"]["equations"] == spec["equations"]
    )
    assert cfg["model"]["input_dim"] == 2 and cfg["model"]["output_dim"] == 1
    cfg["pde"]["trainable_parameters"] = ["r"]
    pde = create_pde(cfg, torch.device("cpu"))
    assert isinstance(pde, CustomPDE) and "r" in pde._trainable_params
    assert math.isclose(pde.time_domain[1], 4.0)


def test_build_trainer_model_overrides_the_architecture_block():
    pde = CustomPDE("u_t - u_xx", domain={"x": (0, 1), "t": (0, 1)}, initial_condition="sin(pi*x)")
    trainer = pde.build_trainer(
        "feedforward", model={"hidden_dims": [16, 16], "layer_norm": False}, num_epochs=1
    )
    assert trainer.config.model.hidden_dims == [16, 16]
    assert trainer.config.model.num_layers == 2
    assert trainer.config.model.layer_norm is False
    assert sum(p.numel() for p in trainer.model.parameters()) < 1000
