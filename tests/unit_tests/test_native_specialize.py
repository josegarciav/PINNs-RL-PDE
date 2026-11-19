"""The per-equation compiled step (pinnrl.native.specialize) against autograd."""

import os

import pytest
import torch

from pinnrl import CustomPDE, native
from pinnrl.native import specialize
from pinnrl.pdes.catalog import from_catalog

if os.environ.get("PINNRL_REQUIRE_NATIVE") and not native.available():  # CI: never skip silently
    pytest.fail(f"the native kernel must build here: {native.unavailable_reason()}")
pytestmark = pytest.mark.skipif(not native.available(), reason=str(native.unavailable_reason()))

SETTINGS = dict(
    num_collocation_points=256,
    batch_size=256,
    num_boundary_points=32,
    num_initial_points=32,
    weight_decay=0.0,
    gradient_clipping=0.0,
    scheduler_type="constant",
    early_stopping={"enabled": False},
)
DOMAIN = {"x": (0.0, 1.0), "t": (0.0, 1.0)}
ZERO = {"dirichlet": {"value": 0.0}}

PROBLEMS = {
    "heat": lambda: from_catalog("heat"),
    "kdv": lambda: from_catalog("kdv_soliton"),
    "mixed": lambda: CustomPDE(
        "u_t + u_x + u*u_x - u_xxt",
        domain=DOMAIN,
        initial_condition="sin(pi*x)",
        boundary_conditions=ZERO,
    ),
    "system": lambda: CustomPDE(
        ["u_t - 0.1*u_xx + u*v^2 - a*sin(x)", "v_t - 0.05*v_xx - u*v^2 + b*v"],
        fields=["u", "v"],
        domain=DOMAIN,
        parameters={"a": 0.3, "b": 0.7},
        initial_condition={"u": "sin(pi*x)", "v": "x"},
        boundary_conditions=ZERO,
        field_weights={"u": 1.0, "v": 4.0},
    ),
    "two_dimensional": lambda: CustomPDE(
        "u_t - 0.1*(u_xx + u_yy) + u^3",
        domain={"x": (0.0, 1.0), "y": (0.0, 2.0), "t": (0.0, 1.0)},
        initial_condition="sin(pi*x)*y",
        boundary_conditions=ZERO,
    ),
    "ode": lambda: CustomPDE("u_t + 2*u - cos(t)", domain={"t": (0.0, 2.0)}, initial_condition="1"),
}


def _build(make, mode, layer_norm=False, **kwargs):
    torch.manual_seed(0)
    pde = make()
    trainer = pde.build_trainer(
        "feedforward",
        model={"hidden_dims": [12, 12], "layer_norm": layer_norm},
        derivatives=mode,
        **SETTINGS,
        **kwargs,
    )
    return pde, trainer


def _loss_and_gradient(pde, model):
    torch.manual_seed(1)
    x, t = pde.generate_collocation_points(256)
    losses = pde.compute_loss(model, x, t)
    grads = torch.autograd.grad(losses["total"], list(model.parameters()))
    return losses["residual"], torch.cat([g.reshape(-1) for g in grads])


def _specialised(loss):
    return type(loss.grad_fn).__name__ == "_ResidualStepBackward"


@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("name", list(PROBLEMS))
def test_loss_and_gradient_agree_with_autograd(name, layer_norm, monkeypatch):
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "1")
    pde_a, trainer_a = _build(PROBLEMS[name], "autograd", layer_norm)
    pde_s, trainer_s = _build(PROBLEMS[name], "auto", layer_norm)
    loss_a, grad_a = _loss_and_gradient(pde_a, trainer_a.model)
    loss_s, grad_s = _loss_and_gradient(pde_s, trainer_s.model)
    assert _specialised(loss_s) and not _specialised(loss_a)
    assert float(loss_s) == pytest.approx(float(loss_a), rel=1e-4)
    assert (grad_s - grad_a).norm() < 1e-4 * grad_a.norm()


def test_the_general_kernel_serves_the_first_calls(monkeypatch):
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "4")
    assert specialize.specialize_after() == 4
    pde, trainer = _build(PROBLEMS["heat"], "auto")
    kinds = [_specialised(_loss_and_gradient(pde, trainer.model)[0]) for _ in range(6)]
    assert kinds == [False, False, False, True, True, True]
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "off")
    assert specialize.specialize_after() is None
    pde, trainer = _build(PROBLEMS["heat"], "auto")
    assert not any(_specialised(_loss_and_gradient(pde, trainer.model)[0]) for _ in range(3))
    assert pde.native_active(trainer.model)  # the general kernel is still in charge


def test_a_gradient_of_the_step_can_be_differentiated_again(monkeypatch):
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "1")
    products = []
    for mode in ("autograd", "auto"):
        pde, trainer = _build(PROBLEMS["heat"], mode)
        parameters = list(trainer.model.parameters())
        torch.manual_seed(1)
        x, t = pde.generate_collocation_points(256)
        loss = pde.compute_loss(trainer.model, x, t)["residual"]
        assert _specialised(loss) == (mode == "auto")
        # the residual does not see the output bias, so that gradient is absent
        parameters = parameters[:-1]
        grads = torch.autograd.grad(loss, parameters, create_graph=True)
        flat = torch.cat([g.reshape(-1) for g in grads])
        torch.manual_seed(2)
        v = torch.randn_like(flat)
        hv = torch.autograd.grad(flat @ v, parameters)
        products.append(torch.cat([h.reshape(-1) for h in hv]))
    assert (products[0] - products[1]).norm() < 1e-3 * products[0].norm()


INVERSE = {
    "a scaled network": (
        lambda: CustomPDE(
            "u_t - 0.1*u_xx + sin(x)*u",
            domain={"x": (0.0, 2.0), "t": (0.0, 3.0)},
            initial_condition="300 + 20*sin(pi*x)",
            boundary_conditions=ZERO,
            scales={"u": (300.0, 20.0)},
        ),
        dict(normalize=True),
    ),
    "trainable parameters": (
        lambda: CustomPDE(
            "u_t - alpha*u_xx + beta*u^2",
            domain=DOMAIN,
            parameters={"alpha": 0.1, "beta": 0.5},
            trainable_parameters=["alpha", "beta"],
            initial_condition="sin(pi*x)",
            boundary_conditions=ZERO,
        ),
        {},
    ),
    "an unknown function of time": (
        lambda: CustomPDE(
            "u_t - k*u_xx",
            domain=DOMAIN,
            unknown_functions={"k": {"inputs": ["t"], "positive": True, "initial": 0.1}},
            initial_condition="sin(pi*x)",
            boundary_conditions=ZERO,
        ),
        {},
    ),
    "an unknown function, a parameter and scaling": (
        lambda: CustomPDE(
            "u_t - k*u_xx + a*u",
            domain=DOMAIN,
            parameters={"a": 0.3},
            trainable_parameters=["a"],
            unknown_functions={"k": {"inputs": ["x", "t"], "positive": True, "initial": 0.1}},
            initial_condition="sin(pi*x)",
            boundary_conditions=ZERO,
        ),
        dict(normalize=True),
    ),
}


@pytest.mark.parametrize("layer_norm", [False, True])
@pytest.mark.parametrize("name", list(INVERSE))
def test_inverse_problems_agree_with_autograd(name, layer_norm, monkeypatch):
    """Every unknown gets its gradient from the step: weights, parameters, unknown functions."""
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "1")
    make, kwargs = INVERSE[name]
    results = []
    for mode in ("autograd", "auto"):
        pde, trainer = _build(make, mode, layer_norm, **kwargs)
        torch.manual_seed(1)
        x, t = pde.generate_collocation_points(256)
        loss = pde.compute_loss(trainer.model, x, t)["residual"]
        assert _specialised(loss) == (mode == "auto")
        network = list(trainer.model.parameters())
        others = [
            p
            for group in trainer.optimizer.param_groups
            for p in group["params"]
            if all(p is not q for q in network)
        ]
        grads = torch.autograd.grad(loss, network + others, allow_unused=True)
        flat = [
            (g if g is not None else torch.zeros_like(p)).reshape(-1)
            for g, p in zip(grads, network + others, strict=True)
        ]
        results.append((float(loss), torch.cat(flat[: len(network)]), flat[len(network) :]))
    (loss_a, net_a, other_a), (loss_s, net_s, other_s) = results
    assert loss_s == pytest.approx(loss_a, rel=1e-4)
    assert (net_s - net_a).norm() < 1e-4 * net_a.norm()
    assert len(other_a) == len(other_s) and (other_a or name == "a scaled network")
    for a, b in zip(other_a, other_s, strict=True):
        assert (a - b).norm() <= 1e-3 * a.norm() + 1e-7


def test_falls_back_where_the_step_does_not_apply(monkeypatch):
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "1")
    cases = {
        "another loss": (PROBLEMS["heat"], dict(loss_function="mae")),
        "a function without a C derivative": (
            lambda: CustomPDE(
                "u_t - 0.1*u_xx - heaviside(u)",
                domain=DOMAIN,
                initial_condition="sin(pi*x)",
                boundary_conditions=ZERO,
            ),
            {},
        ),
        "an unknown function of the field": (
            lambda: CustomPDE(
                "u_t - k*u_xx",
                domain=DOMAIN,
                unknown_functions={"k": {"inputs": ["u"], "positive": True, "initial": 0.1}},
                initial_condition="sin(pi*x)",
                boundary_conditions=ZERO,
            ),
            {},
        ),
        "a callable residual": (
            lambda: CustomPDE(lambda f, p: f.u_t - 0.1 * f.u_xx, domain=DOMAIN),
            {},
        ),
    }
    for label, (make, kwargs) in cases.items():
        pde, trainer = _build(make, "auto", **kwargs)
        torch.manual_seed(1)
        x, t = pde.generate_collocation_points(256)
        loss = pde.compute_loss(trainer.model, x, t)["residual"]
        assert not _specialised(loss), label
        assert torch.isfinite(loss), label


def test_a_parameter_made_trainable_later_is_not_served_by_a_stale_step(monkeypatch):
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "1")
    pde, trainer = _build(
        lambda: CustomPDE(
            "u_t - alpha*u_xx",
            domain=DOMAIN,
            parameters={"alpha": 0.1},
            initial_condition="sin(pi*x)",
            boundary_conditions=ZERO,
        ),
        "auto",
    )
    torch.manual_seed(1)
    x, t = pde.generate_collocation_points(256)
    assert _specialised(pde.compute_loss(trainer.model, x, t)["residual"])
    theta = torch.nn.Parameter(torch.tensor(0.1))
    monkeypatch.setattr(pde, "_parameter_values", lambda: {"alpha": theta})
    loss = pde.compute_loss(trainer.model, x, t)["residual"]
    assert not _specialised(loss)  # the graph to alpha is kept
    assert torch.autograd.grad(loss, theta)[0].abs() > 0


def test_training_with_the_step_lowers_the_loss(monkeypatch):
    monkeypatch.setenv("PINNRL_NATIVE_SPECIALIZE", "2")
    pde, trainer = _build(PROBLEMS["heat"], "auto", learning_rate=2e-3)
    torch.manual_seed(1)
    x, t = pde.generate_collocation_points(256)
    before = float(pde.compute_loss(trainer.model, x, t)["total"])
    trainer.train_steps(num_steps=60, resample_every=1)
    after = pde.compute_loss(trainer.model, x, t)
    assert _specialised(after["residual"])
    assert float(after["total"]) < 0.5 * before


def test_tanh_program_covers_every_stream():
    cols, alphas = native.taylor_streams([(0, 0, 1), (1, 1)])
    forward, reverse = specialize.tanh_program(alphas)
    assert cols == [0, 1]
    for s in range(len(alphas)):
        assert any(line.startswith(f"const float h{s} =") for line in forward)
        assert any(line.startswith(f"const float r{s} =") for line in reverse)
    assert specialize.block_size() >= 1
    source = specialize.norm_program(alphas)
    assert "gen_norm_fwd" in source and "gen_norm_bwd" in source
