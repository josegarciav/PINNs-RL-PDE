"""The compiled jet kernel (pinnrl.native) against autograd, and its wiring into CustomPDE."""

import pytest
import torch

from pinnrl import CustomPDE, native
from pinnrl.neural_networks import ExactLayerNorm
from pinnrl.pdes.catalog import from_catalog

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


def _autograd_jets(linears, x, first, second, norms=()):
    x = x.clone().requires_grad_(True)
    u = _forward(linears, x, norms)
    grads = [torch.autograd.grad(u[:, j].sum(), x, create_graph=True)[0] for j in range(u.shape[1])]
    du = {c: torch.stack([g[:, c] for g in grads], 1) for c in first}
    d2u = {
        c: torch.stack(
            [torch.autograd.grad(g[:, c].sum(), x, create_graph=True)[0][:, c] for g in grads], 1
        )
        for c in second
    }
    return u, du, d2u


def test_jets_and_weight_gradients_match_autograd():
    linears = _mlp([3, 16, 12, 2])
    x = torch.rand(40, 3)
    u, du, d2u = native.mlp_jets(linears, x, first=[0, 2], second=[2, 1])
    assert set(du) == {0, 2, 1} and set(d2u) == {2, 1}  # a second derivative brings its first
    ur, dur, d2ur = _autograd_jets(linears, x, [0, 1, 2], [2, 1])
    assert torch.allclose(u, ur, atol=1e-6)
    assert all(torch.allclose(du[c], dur[c], atol=1e-5) for c in du)
    assert all(torch.allclose(d2u[c], d2ur[c], atol=1e-4) for c in d2u)

    weights = [torch.randn(40, 2) for _ in range(5)]

    def functional(u, du, d2u):
        return (
            (weights[0] * u).sum()
            + (weights[1] * du[0] * du[2]).sum()
            + (weights[2] * d2u[2]).sum()
            + (weights[3] * d2u[1] * u).sum()
            + (weights[4] * du[1]).sum()
        )

    functional(u, du, d2u).backward()
    got = [p.grad.clone() for lin in linears for p in (lin.weight, lin.bias)]
    for lin in linears:
        lin.zero_grad()
    functional(ur, dur, d2ur).backward()
    for g, lin_p in zip(got, [p for lin in linears for p in (lin.weight, lin.bias)], strict=True):
        assert torch.allclose(g, lin_p.grad, rtol=2e-3, atol=2e-4)


def test_layer_normalization_jets_and_gradients_match_autograd():
    linears = _mlp([3, 16, 12, 2])
    norms = [ExactLayerNorm(16), ExactLayerNorm(12)]
    with torch.no_grad():
        for m in norms:
            m.weight.add_(0.3 * torch.randn_like(m.weight))
            m.bias.add_(0.3 * torch.randn_like(m.bias))
    x = torch.rand(40, 3)
    u, du, d2u = native.mlp_jets(linears, x, first=[0, 2], second=[2, 1], norms=norms)
    ur, dur, d2ur = _autograd_jets(linears, x, [0, 1, 2], [2, 1], norms)
    assert torch.allclose(u, ur, atol=1e-5)
    assert all(torch.allclose(du[c], dur[c], atol=1e-4) for c in du)
    assert all(torch.allclose(d2u[c], d2ur[c], atol=1e-3) for c in d2u)
    weights = [torch.randn(40, 2) for _ in range(4)]

    def functional(u, du, d2u):
        return (
            (weights[0] * u).sum()
            + (weights[1] * du[0] * du[2]).sum()
            + (weights[2] * d2u[2]).sum()
            + (weights[3] * d2u[1] * u).sum()
        )

    modules = [*linears, *norms]
    functional(u, du, d2u).backward()
    got = [p.grad.clone() for m in modules for p in m.parameters()]
    for m in modules:
        m.zero_grad()
    functional(ur, dur, d2ur).backward()
    for g, p in zip(got, [p for m in modules for p in m.parameters()], strict=True):
        assert torch.allclose(g, p.grad, rtol=5e-3, atol=5e-4)
    with pytest.raises(ValueError, match="one LayerNorm per hidden layer"):
        native.mlp_jets(linears, x, norms=norms[:1])


def test_value_only_and_validation():
    linears = _mlp([2, 8, 1])
    x = torch.rand(10, 2)
    u, du, d2u = native.mlp_jets(linears, x)
    assert du == {} and d2u == {} and torch.allclose(u, _forward(linears, x), atol=1e-6)
    with pytest.raises(RuntimeError, match="float32"):
        native.mlp_jets(linears, x.double())
    with pytest.raises(RuntimeError, match="does not require grad|differentiate twice"):
        u, du, _ = native.mlp_jets(linears, x, first=[0])
        g = torch.autograd.grad(du[0].sum(), linears[0].weight, create_graph=True)[0]
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
    kdv = CustomPDE("u_t + 6*u*u_x + u_xxx", domain={"x": (0, 1), "t": (0, 1)})  # third order
    trainer = build(kdv, layer_norm=False)
    assert not kdv.native_active(trainer.model)
    assert torch.isfinite(kdv.compute_residual(trainer.model, x, t)).all()
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
