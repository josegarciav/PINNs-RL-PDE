"""Loss-Hessian and tangent-kernel spectra (pinnrl.analysis.spectrum)."""

import pytest
import torch

from pinnrl.analysis import loss_hessian_spectrum, tangent_kernel_spectrum
from pinnrl.pdes.catalog import from_catalog


def _tiny_net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(2, 6), torch.nn.Tanh(), torch.nn.Linear(6, 1))


def test_lanczos_matches_the_dense_hessian():
    net = _tiny_net()
    x = torch.rand(32, 2)
    target = torch.sin(3 * x[:, :1])
    params = list(net.parameters())

    def loss_fn():
        return ((net(x) - target) ** 2).mean()

    spectrum = loss_hessian_spectrum(loss_fn, params, iterations=sum(p.numel() for p in params))
    # Dense Hessian over the flattened parameters for comparison.
    flat = torch.cat([p.detach().reshape(-1) for p in params])
    shapes = [p.shape for p in params]

    def loss_of(v):
        out, i = [], 0
        for s in shapes:
            n = int(torch.tensor(s).prod())
            out.append(v[i : i + n].reshape(s))
            i += n
        pred = torch.func.functional_call(
            net, dict(zip([n for n, _ in net.named_parameters()], out, strict=True)), (x,)
        )
        return ((pred - target) ** 2).mean()

    dense = torch.autograd.functional.hessian(loss_of, flat)
    exact = torch.linalg.eigvalsh(dense.double()).flip(0)
    assert spectrum.largest == pytest.approx(float(exact[0]), rel=1e-4)
    assert spectrum.smallest == pytest.approx(float(exact[-1]), abs=1e-4)


def test_fewer_iterations_bracket_the_extremes():
    net = _tiny_net()
    x = torch.rand(32, 2)
    params = list(net.parameters())

    def loss_fn():
        return ((net(x) - torch.sin(3 * x[:, :1])) ** 2).mean()

    full = loss_hessian_spectrum(loss_fn, params, iterations=25)
    few = loss_hessian_spectrum(loss_fn, params, iterations=8)
    assert few.iterations == 8
    assert few.largest <= full.largest * 1.0001
    assert few.smallest >= full.smallest - 1e-6
    assert few.condition_number() >= 1


def test_rng_untouched():
    net = _tiny_net()
    params = list(net.parameters())
    torch.manual_seed(7)
    before = torch.rand(2)
    torch.manual_seed(7)
    loss_hessian_spectrum(lambda: (net(torch.ones(1, 2)) ** 2).sum(), params, iterations=3)
    assert torch.equal(torch.rand(2), before)


def test_tangent_kernel_of_the_output_and_of_a_residual():
    pde = from_catalog("heat")
    trainer = pde.build_trainer(
        "feedforward", model={"hidden_dims": [16, 16], "layer_norm": False}, num_epochs=1
    )
    model = trainer.model
    x, t = torch.rand(20, 1), torch.rand(20, 1)
    inputs = torch.cat([x, t], 1)
    out = tangent_kernel_spectrum(model, lambda m: m(inputs))
    assert out.eigenvalues.shape == (20,)
    assert out.trace > 0 and out.condition_number() >= 1

    def residual_values(m):
        xt = inputs.clone().requires_grad_(True)
        u = m(xt)
        g = torch.autograd.grad(u.sum(), xt, create_graph=True)[0]
        u_t = g[:, 1:2]
        u_xx = torch.autograd.grad(g[:, 0].sum(), xt, create_graph=True)[0][:, 0:1]
        return u_t - 0.1 * u_xx

    res = tangent_kernel_spectrum(model, residual_values)
    assert res.eigenvalues.shape == (20,)
    assert res.trace > 0
