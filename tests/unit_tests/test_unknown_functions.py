"""Field-valued unknowns: constraints, derivatives, truth, trainer wiring, run snapshot."""

import math

import numpy as np
import pytest
import torch
import yaml

from pinnrl import CustomPDE, UnknownFunction
from pinnrl.benchmarks import heat_variable_reference
from pinnrl.training.train import build_config_dict, create_pde, load_run, run_training

torch.manual_seed(0)
X = torch.rand(40, 1)
T = torch.rand(40, 1)


def _heat_alpha_t(**kwargs):
    return CustomPDE(
        "u_t - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="exp(-pi^2*(0.05*t + 0.05*t^2))*sin(pi*x)",
        unknown_functions={
            "alpha": {"inputs": ["t"], "positive": True, "initial": 0.2, "truth": "0.05 + 0.1*t"}
        },
        **kwargs,
    )


# ------------------------------------------------------------- the function
def test_unknown_function_starts_at_its_initial_constant():
    for kwargs in ({}, {"positive": True}, {"bounds": (0.01, 1.0)}, {"monotone": "increasing"}):
        fn = UnknownFunction("a", ["x"], initial=0.3, **kwargs)
        out = fn({"x": torch.linspace(-1, 1, 20).reshape(-1, 1)})
        assert out.shape == (20, 1)
        assert torch.allclose(out, torch.full_like(out, 0.3), atol=1e-3)


def test_positive_and_bounded_outputs_stay_in_their_sets_after_random_weights():
    torch.manual_seed(1)
    pos = UnknownFunction("a", ["x"], positive=True, initial=0.1)
    box = UnknownFunction("b", ["x"], bounds=(0.2, 0.5), initial=0.3)
    with torch.no_grad():
        for fn in (pos, box):
            for p in fn.parameters():
                p.add_(3 * torch.randn_like(p))
    x = {"x": torch.linspace(-1, 1, 200).reshape(-1, 1)}
    assert (pos(x) > 0).all()
    assert (box(x) >= 0.2).all() and (box(x) <= 0.5).all()
    assert box(x).max() - box(x).min() > 0.05  # the weights moved it off the start


@pytest.mark.parametrize("direction", ["increasing", "decreasing"])
def test_monotone_network_is_monotone_with_random_weights(direction):
    fn = UnknownFunction("f", ["u"], monotone=direction, hidden=(16, 16), initial=0.0)
    with torch.no_grad():
        for p in fn.parameters():
            p.add_(2 * torch.randn_like(p))
    v = fn({"u": torch.linspace(-1, 1, 300).reshape(-1, 1)}).ravel()
    diffs = v[1:] - v[:-1]
    assert (diffs >= -1e-6).all() if direction == "increasing" else (diffs <= 1e-6).all()


def test_monotone_in_one_input_leaves_the_other_free():
    fn = UnknownFunction("f", ["u", "x"], monotone={"u": "increasing"}, hidden=(16,), initial=0.0)
    with torch.no_grad():
        for p in fn.parameters():
            p.add_(2 * torch.randn_like(p))
    u = torch.linspace(-1, 1, 100).reshape(-1, 1)
    for x0 in (-0.7, 0.2, 0.9):
        v = fn({"u": u, "x": torch.full_like(u, x0)}).ravel()
        assert (v[1:] - v[:-1] >= -1e-6).all()


def test_truth_expression_and_relative_error():
    fn = UnknownFunction("a", ["t"], input_ranges={"t": (0, 1)}, initial=0.1, truth="0.05 + 0.1*t")
    t = torch.linspace(0, 1, 5).reshape(-1, 1)
    assert torch.allclose(fn.truth({"t": t}), 0.05 + 0.1 * t)
    # constant 0.1 against 0.05 + 0.1 t over [0, 1]
    expected = math.sqrt(np.mean((0.1 - (0.05 + 0.1 * np.linspace(0, 1, 256))) ** 2))
    expected /= math.sqrt(np.mean((0.05 + 0.1 * np.linspace(0, 1, 256)) ** 2))
    assert fn.relative_error() == pytest.approx(expected, rel=1e-3)
    assert fn.has_truth
    assert not UnknownFunction("b", ["t"]).has_truth


def test_spec_round_trip_and_validation_errors():
    fn = UnknownFunction(
        "a", ["x", "u"], positive=True, monotone={"u": "decreasing"}, initial=0.5, truth="x + u"
    )
    spec = fn.to_spec()
    rebuilt = UnknownFunction.from_spec("a", yaml.safe_load(yaml.safe_dump(spec)))
    assert rebuilt.to_spec() == spec
    with pytest.raises(ValueError, match="at least one input"):
        UnknownFunction("a", [])
    with pytest.raises(ValueError, match="activation"):
        UnknownFunction("a", ["x"], activation="cosine")
    with pytest.raises(ValueError, match="non-decreasing"):
        UnknownFunction("a", ["x"], monotone="increasing", activation="gelu")
    with pytest.raises(ValueError, match="lo < hi"):
        UnknownFunction("a", ["x"], bounds=(1, 0))
    with pytest.raises(ValueError, match="outside bounds"):
        UnknownFunction("a", ["x"], bounds=(0, 1), initial=2.0)
    with pytest.raises(ValueError, match="not an input"):
        UnknownFunction("a", ["x"], monotone={"y": "increasing"})
    with pytest.raises(ValueError, match="not among its inputs"):
        UnknownFunction("a", ["x"], truth="x + beta")


# ------------------------------------------------------------ in a CustomPDE
def test_unknown_appears_in_the_residual_like_a_parameter():
    pde = _heat_alpha_t()
    assert "alpha" not in pde.parameters
    assert pde.unknown_functions["alpha"].input_ranges["t"] == (0.0, 1.0)  # domain filled in
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    res = pde.compute_residual(net, X, T)
    assert res.shape == (40, 1)
    # At the start alpha ≡ 0.2, so the residual equals u_t - 0.2 u_xx.
    fixed = CustomPDE(
        "u_t - 0.2*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
    )
    assert torch.allclose(res, fixed.compute_residual(net, X, T), atol=1e-5)


def test_derivative_of_an_unknown_by_autograd():
    pde = CustomPDE(
        "u_t - alpha_x*u_x - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        unknown_functions={"alpha": {"inputs": ["x"], "truth": "0.1 + 0.05*x"}},
    )
    fn = pde.unknown_functions["alpha"]
    # Make alpha a known non-constant function: alpha = 0.3 + 0.2 * tanh(...)-ish via random weights
    with torch.no_grad():
        for p in fn.parameters():
            p.add_(torch.randn_like(p))
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    from pinnrl import Fields

    f = Fields(net, X, T, ("u",), ("x", "t"), torch.device("cpu"), pde.unknown_functions)
    a_x = f.alpha_x
    # finite difference of the function itself
    h = 1e-3
    with torch.no_grad():
        fd = (fn({"x": X + h}) - fn({"x": X - h})) / (2 * h)
    assert torch.allclose(a_x, fd, atol=1e-3, rtol=1e-2)
    assert torch.allclose(f.alpha_t, torch.zeros_like(a_x))  # alpha does not depend on t
    assert f.unknowns == ("alpha",)
    with pytest.raises(AttributeError, match="unknown function"):
        _ = f.beta


def test_closure_of_the_field_differentiates_through_the_network():
    pde = CustomPDE(
        "u_t - 0.1*u_xx - f",
        domain={"x": (0, 1), "t": (0, 1)},
        unknown_functions={"f": {"inputs": ["u"], "monotone": "increasing", "truth": "u"}},
    )
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    from pinnrl import Fields

    f = Fields(net, X, T, ("u",), ("x", "t"), torch.device("cpu"), pde.unknown_functions)
    # d f(u)/dx = f'(u) u_x, with f ≡ constant at start: zero
    assert torch.allclose(f.f_x, torch.zeros_like(f.u), atol=1e-3)
    assert pde.compute_residual(net, X, T).shape == (40, 1)
    err = pde.unknown_function_error("f", num_points=200, model=net)
    assert err > 0


def test_synthetic_observations_and_exact_solution_use_the_truth():
    pde = CustomPDE(
        "u_t - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",  # mentions the unknown: its truth is used
        unknown_functions={"alpha": {"inputs": ["t"], "initial": 0.5, "truth": "0.1"}},
    )
    u = pde.exact_solution(X, T)
    assert torch.allclose(u, torch.exp(-0.1 * math.pi**2 * T) * torch.sin(math.pi * X))
    obs = pde.generate_synthetic_observations(30, 0.0, seed=1)
    assert torch.allclose(
        obs["u"], torch.exp(-0.1 * math.pi**2 * obs["t"]) * torch.sin(math.pi * obs["x"])
    )
    with pytest.raises(ValueError, match="needs a truth"):
        CustomPDE(
            "u_t - alpha*u_xx",
            domain={"x": (0, 1), "t": (0, 1)},
            exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)",
            unknown_functions={"alpha": {"inputs": ["t"]}},
        )
    with pytest.raises(ValueError, match="clashes"):
        CustomPDE(
            "u_t - alpha*u_xx",
            domain={"x": (0, 1), "t": (0, 1)},
            parameters={"alpha": 0.1},
            unknown_functions={"alpha": {"inputs": ["t"]}},
        )
    with pytest.raises(ValueError, match="neither a coordinate"):
        CustomPDE(
            "u_t - alpha*u_xx",
            domain={"x": (0, 1), "t": (0, 1)},
            unknown_functions={"alpha": {"inputs": ["y"]}},
        )


def test_build_trainer_goes_inverse_and_optimizes_the_function_weights():
    pde = _heat_alpha_t()
    pde.generate_synthetic_observations(50, 0.0, seed=0)
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [16, 16]},
        num_collocation_points=64,
        batch_size=64,
        num_boundary_points=16,
        num_initial_points=16,
        scheduler_type="constant",
        early_stopping={"enabled": False},
    )
    assert trainer.config.training.mode == "inverse"
    fn = pde.unknown_functions["alpha"]
    before = [p.detach().clone() for p in fn.parameters()]
    n_fn = sum(p.numel() for p in fn.parameters())
    n_opt = sum(p.numel() for g in trainer.optimizer.param_groups for p in g["params"])
    assert n_opt == sum(p.numel() for p in trainer.model.parameters()) + n_fn
    trainer.train_steps(num_steps=3)
    assert any(not torch.equal(a, b) for a, b in zip(before, fn.parameters(), strict=True))
    assert pde.get_trainable_parameter_values() == {}  # no scalar unknowns
    assert 0 < pde.unknown_function_error("alpha") < 2


def test_evaluate_unknown_and_the_variable_coefficient_reference():
    ref = heat_variable_reference(lambda x, t: 0.05 + 0.1 * t, nx=101, nt=101, refine=2)
    xx, tt = np.meshgrid(ref.x, ref.t, indexing="ij")
    exact = np.exp(-(np.pi**2) * (0.05 * tt + 0.05 * tt**2)) * np.sin(np.pi * xx)
    assert ref.relative_l2(exact) < 1e-5
    pde = CustomPDE(
        "u_t - alpha_x*u_x - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        initial_condition="sin(pi*x)",
        boundary_conditions={"dirichlet": {"value": 0.0}},
        exact_solution=ref.as_exact_solution(),
        unknown_functions={"alpha": {"inputs": ["t"], "initial": 0.1, "truth": "0.05 + 0.1*t"}},
    )
    u = pde.exact_solution(X, T)
    assert torch.allclose(
        u, torch.exp(-(math.pi**2) * (0.05 * T + 0.05 * T**2)) * torch.sin(math.pi * X), atol=1e-3
    )
    assert torch.allclose(pde.evaluate_unknown("alpha", X, T, truth=True), 0.05 + 0.1 * T)
    assert torch.allclose(pde.evaluate_unknown("alpha", X, T), torch.full_like(T, 0.1), atol=1e-5)


def test_pde_file_run_saves_and_reloads_the_unknown_weights(tmp_path):
    from pinnrl.config import DEFAULT_CONFIG_PATH

    spec = _heat_alpha_t().to_dict()
    assert spec["unknown_functions"]["alpha"]["truth"] == "0.05 + 0.1*t"
    with open(DEFAULT_CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    config = build_config_dict(base, None, "feedforward", custom=spec, epochs=2)
    config["training"].update(
        {"mode": "inverse", "num_collocation_points": 64, "batch_size": 64, "device": "cpu"}
    )
    config["model"]["hidden_dims"] = [8, 8]
    config["inverse"] = {"obs_points": 20, "obs_noise": 0.0, "obs_seed": 0}
    config["paths"]["results_dir"] = str(tmp_path)
    pde = create_pde(config, torch.device("cpu"))
    assert pde.observation_data is not None  # synthetic data generated for the unknown
    run_training(config, torch.device("cpu"))
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    assert (run_dir / "unknown_functions.pt").exists()
    pde2, model, _ = load_run(run_dir)
    trained = torch.load(run_dir / "unknown_functions.pt")
    for k, v in pde2.unknown_state_dict().items():
        assert torch.equal(v, trained[k])
    assert not torch.allclose(
        pde2.evaluate_unknown("alpha", X, T), torch.full_like(T, 0.2)
    )  # the weights moved and were restored
