"""Every catalogue entry builds, has a finite residual, and its exact solution zeroes it."""

import pytest
import torch

from pinnrl.pdes.catalog import CATALOG, from_catalog

# Solutions that hold only in a weak sense (a kink at a free boundary), so the
# point-wise residual check is skipped; validate() still works for them.
_WEAK = {"porous_medium"}


class _Exact(torch.nn.Module):
    """A 'network' that returns the exact solution, so the residual must vanish."""

    def __init__(self, pde):
        super().__init__()
        self.pde = pde

    def forward(self, inp):
        d = self.pde.dimension
        return self.pde.exact_solution(inp[:, :d], inp[:, d:])


def _points(pde, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    lo = torch.tensor([b[0] for b in pde.domain])
    hi = torch.tensor([b[1] for b in pde.domain])
    # stay off the faces: closed forms with sqrt/log can be singular there
    x = lo + (hi - lo) * (0.05 + 0.9 * torch.rand(n, pde.dimension, generator=g))
    t0, t1 = pde.time_domain
    t = t0 + (t1 - t0) * (0.05 + 0.9 * torch.rand(n, 1, generator=g))
    return x, t


@pytest.mark.parametrize("key", sorted(CATALOG))
def test_entry_builds_and_has_a_finite_residual(key):
    pde = from_catalog(key)
    net = torch.nn.Sequential(
        torch.nn.Linear(pde.config.input_dim, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, pde.config.output_dim),
    )
    x, t = _points(pde, n=64)
    residual = pde.compute_residual(net, x, t)
    assert residual.shape == (64, len(pde.fields))
    assert torch.isfinite(residual).all()
    losses = pde.compute_loss(net, x, t)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()  # the graph reaches the network


@pytest.mark.parametrize("key", sorted(k for k, v in CATALOG.items() if v.get("exact_solution")))
def test_exact_solution_satisfies_the_equation_and_initial_condition(key):
    pde = from_catalog(key)
    x, t = _points(pde)
    exact = _Exact(pde)
    u = pde.exact_solution(x, t)
    assert u.shape == (200, len(pde.fields)) and torch.isfinite(u).all()
    if key not in _WEAK:
        residual = pde.compute_residual(exact, x, t)
        scale = max(float(u.abs().max()), 1.0)
        assert float(residual.abs().max()) < 1e-3 * scale, residual.abs().max()
    if pde.initial_condition is not None:
        t0 = torch.full_like(t, pde.time_domain[0])
        assert torch.allclose(pde.exact_solution(x, t0), pde.initial_condition(x, t0), atol=1e-5)
        assert float(pde._initial_loss(exact)) < 1e-8


def test_overrides_and_unknown_key():
    pde = from_catalog("burgers", parameters={"nu": 0.5})
    assert pde.parameters["nu"] == 0.5
    with pytest.raises(KeyError, match="Unknown catalogue entry"):
        from_catalog("nope")


def test_catalogue_covers_the_problem_kinds():
    kinds = {
        "ode": [k for k, v in CATALOG.items() if set(v["domain"]) == {"t"}],
        "steady": [k for k, v in CATALOG.items() if "t" not in v["domain"]],
        "system": [k for k, v in CATALOG.items() if len(v.get("fields", ["u"])) > 1],
        "two_d": [k for k, v in CATALOG.items() if len(set(v["domain"]) - {"t"}) == 2],
    }
    assert len(CATALOG) >= 30
    assert all(len(v) >= 3 for v in kinds.values()), {k: len(v) for k, v in kinds.items()}
