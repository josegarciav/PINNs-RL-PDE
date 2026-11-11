"""ExactLayerNorm: the values of nn.LayerNorm, and weight gradients that survive second derivatives."""

import pytest
import torch

from pinnrl.neural_networks import ExactLayerNorm
from pinnrl.pdes.catalog import from_catalog


def _net(norm_cls, seed=0, width=24):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(2, width), norm_cls(width), torch.nn.Tanh(),
        torch.nn.Linear(width, width), norm_cls(width), torch.nn.Tanh(),
        torch.nn.Linear(width, 1),
    ).double()  # fmt: skip


def _heat_loss(net, xt):
    xt = xt.clone().requires_grad_(True)
    g = torch.autograd.grad(net(xt).sum(), xt, create_graph=True)[0]
    u_xx = torch.autograd.grad(g[:, 0].sum(), xt, create_graph=True)[0][:, 0]
    return ((g[:, 1] - 0.1 * u_xx) ** 2).mean()


def _directional_error(net, xt, seed, h=1e-6):
    """Relative error of the autograd directional derivative against central differences."""
    params = list(net.parameters())
    grads = torch.autograd.grad(_heat_loss(net, xt), params, allow_unused=True)
    grads = [torch.zeros_like(p) if g is None else g for g, p in zip(grads, params, strict=True)]
    torch.manual_seed(seed)
    direction = [torch.randn_like(p) for p in params]
    analytic = sum(float((g * d).sum()) for g, d in zip(grads, direction, strict=True))
    values = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            for p, d in zip(params, direction, strict=True):
                p.add_(sign * h * d)
        values.append(float(_heat_loss(net, xt)))
        with torch.no_grad():
            for p, d in zip(params, direction, strict=True):
                p.sub_(sign * h * d)
    numeric = (values[0] - values[1]) / (2 * h)
    return abs(analytic - numeric) / abs(numeric)


def test_same_values_and_parameters_as_torch_layer_norm():
    torch.manual_seed(0)
    ours, theirs = ExactLayerNorm(8), torch.nn.LayerNorm(8)
    with torch.no_grad():
        ours.weight.copy_(torch.randn(8))
        ours.bias.copy_(torch.randn(8))
    theirs.load_state_dict(ours.state_dict())  # the same parameter names
    x = torch.randn(5, 3, 8)
    assert torch.allclose(ours(x), theirs(x), atol=1e-6)
    plain = ExactLayerNorm(8, elementwise_affine=False)
    assert torch.allclose(plain(x), torch.nn.functional.layer_norm(x, (8,)), atol=1e-6)
    assert isinstance(ours, torch.nn.LayerNorm)


def test_weight_gradient_of_a_second_order_residual_matches_finite_differences():
    torch.manual_seed(1)
    xt = torch.rand(64, 2, dtype=torch.float64)
    errors = [_directional_error(_net(ExactLayerNorm), xt, seed) for seed in range(4)]
    assert max(errors) < 1e-6
    # the reason the class exists: the built-in layer is off by percent on the same loss.
    # If this starts failing, PyTorch has fixed its third-order derivative.
    builtin = [_directional_error(_net(torch.nn.LayerNorm), xt, seed) for seed in range(4)]
    if max(builtin) < 1e-6:
        pytest.skip("torch.nn.LayerNorm now differentiates correctly to third order")
    assert max(builtin) > 1e-3


def test_the_packaged_networks_use_it():
    pde = from_catalog("heat")
    trainer = pde.build_trainer("feedforward", num_epochs=1)
    norms = [m for m in trainer.model.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert norms and all(isinstance(m, ExactLayerNorm) for m in norms)
