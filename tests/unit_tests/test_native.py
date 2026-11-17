"""The compiled jet kernel (pinnrl.native) against autograd, and its wiring into CustomPDE."""

import os

import pytest
import torch

from pinnrl import CustomPDE, native
from pinnrl.neural_networks import ExactLayerNorm
from pinnrl.pdes.catalog import from_catalog

if os.environ.get("PINNRL_REQUIRE_NATIVE") and not native.available():  # CI: never skip silently
    pytest.fail(f"the native kernel must build here: {native.unavailable_reason()}")
pytestmark = pytest.mark.skipif(not native.available(), reason=str(native.unavailable_reason()))

SETTINGS = dict(
    num_collocation_points=256,
    batch_size=256,
    num_boundary_points=64,
    num_initial_points=64,
    weight_decay=0.0,
    gradient_clipping=0.0,
    scheduler_type="constant",
    early_stopping={"enabled": False},
)


def _mlp(sizes, seed=0):
    torch.manual_seed(seed)
    return [torch.nn.Linear(a, b) for a, b in zip(sizes[:-1], sizes[1:], strict=True)]


def _forward(linears, x, norms=()):
    h = x
    for k, lin in enumerate(linears):
        h = lin(h)
        if k < len(linears) - 1:
            h = torch.tanh(norms[k](h) if norms else h)
    return h


def _autograd_jets(linears, x, keys, norms=()):
    """The value and the derivatives ``keys`` (tuples of input columns) by repeated autograd."""
    x = x.clone().requires_grad_(True)
    u = _forward(linears, x, norms)
    du = {}
    for key in keys:
        cols = []
        for j in range(u.shape[1]):
            g = u[:, j]
            for c in key:
                g = torch.autograd.grad(g.sum(), x, create_graph=True)[0][:, c]
            cols.append(g)
        du[key] = torch.stack(cols, 1)
    return u, du


def _check_weight_gradients(modules, u, du, ur, dur, rtol, atol):
    """A functional of every derivative has the same weight gradient from both."""
    torch.manual_seed(7)
    w0 = torch.randn_like(u)
    ws = {k: torch.randn_like(v) for k, v in du.items()}

    def functional(u, du):
        total = (w0 * u).sum()
        for k, v in du.items():
            total = total + (ws[k] * v * (1 + 0.1 * u)).sum()
        return total

    functional(u, du).backward()
    got = [p.grad.clone() for m in modules for p in m.parameters()]
    for m in modules:
        m.zero_grad()
    functional(ur, dur).backward()
    for g, p in zip(got, [p for m in modules for p in m.parameters()], strict=True):
        assert torch.allclose(g, p.grad, rtol=rtol, atol=atol)


def test_jets_and_weight_gradients_match_autograd():
    linears = _mlp([3, 16, 12, 2])
    x = torch.rand(40, 3)
    u, du = native.mlp_jets(linears, x, [(0,), (2, 2), (1, 1)])
    assert set(du) == {(0,), (1,), (2,), (1, 1), (2, 2)}  # a second derivative brings its first
    ur, dur = _autograd_jets(linears, x, list(du))
    assert torch.allclose(u, ur, atol=1e-6)
    assert all(torch.allclose(du[c], dur[c], atol=1e-5) for c in du if len(c) == 1)
    assert all(torch.allclose(du[c], dur[c], atol=1e-4) for c in du if len(c) == 2)
    _check_weight_gradients(linears, u, du, ur, dur, rtol=2e-3, atol=2e-4)


def test_higher_and_mixed_derivatives_match_autograd():
    """Third, fourth and mixed derivatives (KdV, the beam, u_xt), and their weight gradients (I13)."""
    linears = _mlp([2, 16, 12, 1], seed=3)
    x = torch.rand(40, 2)
    u, du = native.mlp_jets(linears, x, [(0, 0, 0, 0), (0, 1), (1, 1), (0, 0, 1)])
    assert set(du) == {(0,), (1,), (0, 0), (0, 1), (1, 1), (0, 0, 0), (0, 0, 1), (0, 0, 0, 0)}
    ur, dur = _autograd_jets(linears, x, list(du))
    for key, value in du.items():
        scale = dur[key].abs().max().item()
        assert torch.allclose(value, dur[key], rtol=1e-4, atol=1e-5 * scale), key
    _check_weight_gradients(linears, u, du, ur, dur, rtol=2e-3, atol=2e-4)
    cols, alphas = native.taylor_streams([(0, 0, 1)])
    assert cols == [0, 1] and alphas == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    with pytest.raises(ValueError, match="columns must be in"):
        native.mlp_jets(linears, x, [(0, 5)])


def test_layer_normalization_jets_and_gradients_match_autograd():
    linears = _mlp([3, 16, 12, 2])
    norms = [ExactLayerNorm(16), ExactLayerNorm(12)]
    with torch.no_grad():
        for m in norms:
            m.weight.add_(0.3 * torch.randn_like(m.weight))
            m.bias.add_(0.3 * torch.randn_like(m.bias))
    x = torch.rand(40, 3)
    # second, mixed and third derivatives through the normalization
    u, du = native.mlp_jets(linears, x, [(0,), (2, 2), (1, 1), (0, 1), (2, 2, 2)], norms=norms)
    ur, dur = _autograd_jets(linears, x, list(du), norms)
    assert torch.allclose(u, ur, atol=1e-5)
    for key, value in du.items():
        scale = dur[key].abs().max().item()
        assert torch.allclose(value, dur[key], rtol=1e-4, atol=1e-5 * scale), key
    _check_weight_gradients([*linears, *norms], u, du, ur, dur, rtol=5e-3, atol=5e-4)
    with pytest.raises(ValueError, match="one LayerNorm per hidden layer"):
        native.mlp_jets(linears, x, norms=norms[:1])


def test_value_only_and_validation():
    linears = _mlp([2, 8, 1])
    x = torch.rand(10, 2)
    u, du = native.mlp_jets(linears, x)
    assert du == {} and torch.allclose(u, _forward(linears, x), atol=1e-6)
    with pytest.raises(RuntimeError, match="float32"):
        native.mlp_jets(linears, x.double())
    with pytest.raises(RuntimeError, match="does not require grad|differentiate twice"):
        u, du = native.mlp_jets(linears, x, [(0,)])
        g = torch.autograd.grad(du[(0,)].sum(), linears[0].weight, create_graph=True)[0]
        g.sum().backward()
    tanh, lin = torch.nn.Tanh, torch.nn.Linear
    assert native.supported_layers([lin(2, 4), tanh(), lin(4, 1)]) is not None
    assert (
        native.supported_layers([lin(2, 4), torch.nn.Dropout(0.0), tanh(), lin(4, 1)]) is not None
    )
    stack = native.supported_stack([lin(2, 4), torch.nn.LayerNorm(4), tanh(), lin(4, 1)])
    assert stack is not None and len(stack[1]) == 1
    assert native.supported_stack([lin(2, 4), torch.nn.LayerNorm(4), lin(4, 1)]) is None
    plain = torch.nn.LayerNorm(4, elementwise_affine=False)
    assert native.supported_stack([lin(2, 4), plain, tanh(), lin(4, 1)]) is None
    mixed = [lin(2, 4), torch.nn.LayerNorm(4), tanh(), lin(4, 4), tanh(), lin(4, 1)]
    assert native.supported_stack(mixed) is None  # a norm on some hidden layers only
    assert native.supported_layers([lin(2, 4), torch.nn.ReLU(), lin(4, 1)]) is None
    assert native.supported_layers([lin(2, 4), tanh()]) is None
    assert native.supported_layers([lin(2, 4), torch.nn.Dropout(0.1), tanh(), lin(4, 1)]) is None


def _pair(pde_factory, layer_norm=False, **kwargs):
    """The same problem and network with autograd and with the kernel."""
    out = []
    for mode in ("autograd", "native"):
        torch.manual_seed(0)
        pde = pde_factory()
        trainer = pde.build_trainer(
            "feedforward",
            model={"hidden_dims": [16, 16], "layer_norm": layer_norm},
            derivatives=mode,
            **SETTINGS,
            **kwargs,
        )
        out.append((pde, trainer))
    return out


@pytest.mark.parametrize("layer_norm", [False, True])
def test_heat_residual_loss_and_gradients_agree_with_autograd(layer_norm):
    (pde_a, tr_a), (pde_n, tr_n) = _pair(lambda: from_catalog("heat"), layer_norm=layer_norm)
    assert pde_n.native_active(tr_n.model) and not pde_a.native_active(tr_a.model)
    x, t = torch.rand(100, 1), torch.rand(100, 1)
    r_a, r_n = pde_a.compute_residual(tr_a.model, x, t), pde_n.compute_residual(tr_n.model, x, t)
    assert torch.allclose(r_a, r_n, atol=1e-4)
    torch.manual_seed(1)
    loss_a = pde_a.compute_loss(tr_a.model, x, t)["total"]
    torch.manual_seed(1)
    loss_n = pde_n.compute_loss(tr_n.model, x, t)["total"]
    assert loss_n.item() == pytest.approx(loss_a.item(), rel=1e-4)
    loss_a.backward()
    loss_n.backward()
    for a, n in zip(tr_a.model.parameters(), tr_n.model.parameters(), strict=True):
        assert torch.allclose(a.grad, n.grad, rtol=5e-3, atol=1e-4)
    # values at points nothing is differentiated at go through the kernel too
    xt = torch.rand(20, 2)
    assert torch.allclose(tr_n.model(xt), tr_a.model(xt), atol=1e-6)
    before = loss_n.item()
    tr_n.train_steps(num_steps=150, resample_every=1)
    torch.manual_seed(1)
    assert pde_n.compute_loss(tr_n.model, x, t)["total"].item() < 0.5 * before


def test_normalized_system_and_an_ode_agree_with_autograd():
    def system():
        return CustomPDE(
            ["u_t - 0.02*u_xx + 3*u - v", "v_t - 0.01*v_xx - u_x"],
            domain={"x": (0.0, 2.0), "t": (0.0, 5.0)},
            fields=["u", "v"],
            scales={"u": (1.0, 3.0), "v": (0.0, 0.5)},
            initial_condition={"u": "sin(pi*x)", "v": "0"},
            boundary_conditions={"dirichlet": {"value": 0.0}},
        )

    (pde_a, tr_a), (pde_n, tr_n) = _pair(system, normalize=True)
    assert pde_n.native_active(tr_n.model)
    x, t = 2 * torch.rand(60, 1), 5 * torch.rand(60, 1)
    r_a, r_n = pde_a.compute_residual(tr_a.model, x, t), pde_n.compute_residual(tr_n.model, x, t)
    assert r_n.shape == (60, 2) and torch.allclose(r_a, r_n, rtol=1e-3, atol=1e-4)

    def oscillator():
        return CustomPDE(
            "u_tt + 4*u", domain={"t": (0.0, 3.0)}, initial_condition="1", exact_solution="cos(2*t)"
        )

    (pde_a, tr_a), (pde_n, tr_n) = _pair(oscillator)
    assert pde_n.native_active(tr_n.model)
    x, t = torch.rand(30, 1), 3 * torch.rand(30, 1)
    assert torch.allclose(
        pde_a.compute_residual(tr_a.model, x, t),
        pde_n.compute_residual(tr_n.model, x, t),
        atol=1e-4,
    )


@pytest.mark.parametrize(
    "key, layer_norm",
    [("kdv_soliton", False), ("kdv_soliton", True), ("beam", False), ("cahn_hilliard", True)],
)
def test_third_and_fourth_order_residuals_agree_with_autograd(key, layer_norm):
    """KdV (u_xxx), the Euler-Bernoulli beam (w_tt, w_xxxx) and Cahn-Hilliard (u_xxxx) (I13)."""
    (pde_a, tr_a), (pde_n, tr_n) = _pair(lambda: from_catalog(key), layer_norm=layer_norm)
    assert pde_n.native_active(tr_n.model) and not pde_a.native_active(tr_a.model)
    lo, hi = pde_a.domain[0]
    x, t = lo + (hi - lo) * torch.rand(80, 1), torch.rand(80, 1)
    r_a, r_n = pde_a.compute_residual(tr_a.model, x, t), pde_n.compute_residual(tr_n.model, x, t)
    assert torch.allclose(r_a, r_n, rtol=1e-3, atol=1e-4 * r_a.abs().max().item())
    torch.manual_seed(1)
    loss_a = pde_a.compute_loss(tr_a.model, x, t)["total"]
    torch.manual_seed(1)
    loss_n = pde_n.compute_loss(tr_n.model, x, t)["total"]
    assert loss_n.item() == pytest.approx(loss_a.item(), rel=1e-3)
    loss_a.backward()
    loss_n.backward()
    for a, n in zip(tr_a.model.parameters(), tr_n.model.parameters(), strict=True):
        assert torch.allclose(
            a.grad, n.grad, rtol=5e-3, atol=1e-3 * a.grad.abs().max().item() + 1e-7
        )


def test_mixed_derivative_residual_agrees_with_autograd():
    """A Benjamin-Bona-Mahony term u_xxt and a mixed u_xt through the kernel (I13)."""

    def bbm():
        return CustomPDE(
            "u_t + u_x + u*u_x - u_xxt + 0.5*u_xt",
            domain={"x": (-5.0, 5.0), "t": (0.0, 1.0)},
            initial_condition="exp(-x^2)",
            boundary_conditions={"dirichlet": {"value": 0.0}},
        )

    (pde_a, tr_a), (pde_n, tr_n) = _pair(bbm, layer_norm=True)
    assert pde_n.native_active(tr_n.model)
    x, t = 10 * torch.rand(60, 1) - 5, torch.rand(60, 1)
    r_a, r_n = pde_a.compute_residual(tr_a.model, x, t), pde_n.compute_residual(tr_n.model, x, t)
    assert torch.allclose(r_a, r_n, rtol=1e-3, atol=1e-4 * r_a.abs().max().item())


def test_falls_back_to_autograd_where_the_kernel_does_not_apply():
    def build(pde, **model):
        torch.manual_seed(0)
        return pde.build_trainer(
            "feedforward", model={"hidden_dims": [8, 8], **model}, derivatives="native", **SETTINGS
        )

    x, t = torch.rand(20, 1), torch.rand(20, 1)
    heat = from_catalog("heat")
    trainer = build(heat, layer_norm=False, activation="gelu")
    assert not heat.native_active(trainer.model)
    assert torch.isfinite(heat.compute_residual(trainer.model, x, t)).all()
    callable_residual = CustomPDE(
        lambda f, p: f.u_t - f.u_xx, domain={"x": (0, 1), "t": (0, 1)}
    )  # the kernel needs the derivatives named in advance, which text gives and a callable does not
    trainer = build(callable_residual, layer_norm=False)
    assert not callable_residual.native_active(trainer.model)
    assert torch.isfinite(callable_residual.compute_residual(trainer.model, x, t)).all()
    unknown = CustomPDE(
        "u_t - alpha*u_xx",
        domain={"x": (0, 1), "t": (0, 1)},
        unknown_functions={"alpha": {"inputs": ["t"], "positive": True, "initial": 0.1}},
    )
    trainer = build(unknown, layer_norm=False)
    assert not unknown.native_active(trainer.model)
    assert torch.isfinite(unknown.compute_residual(trainer.model, x, t)).all()
    with pytest.raises(ValueError, match="'autograd' or 'native'"):
        heat.build_trainer("feedforward", derivatives="fast")
