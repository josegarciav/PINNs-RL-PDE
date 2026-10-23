"""Tests for :class:`pinnrl.rl.bandit_sampler.ResidualBanditSampler`."""

import pytest
import torch

from pinnrl.rl.bandit_sampler import ResidualBanditSampler
from pinnrl.rl.rl_agent import RLAgent


class _HotSpotPDE:
    """Stand-in PDE whose |residual| is large only for x > 0.5."""

    def compute_residual(self, model, x, t):
        return torch.where(x > 0.5, torch.full_like(x, 10.0), torch.full_like(x, 0.1))


def _sampler(**kwargs):
    defaults = dict(n_x_cells=4, n_t_cells=2, probe_cells=8, probe_points=4)
    defaults.update(kwargs)
    return ResidualBanditSampler((0.0, 1.0), (0.0, 1.0), **defaults)


def test_sample_shapes_bounds_and_mix():
    """Batches have the requested size, stay in the domain, and include uniform points."""
    sampler = _sampler()
    x, t = sampler.sample(_HotSpotPDE(), model=None, num_points=101)
    assert x.shape == (101, 1) and t.shape == (101, 1)
    assert torch.all((x >= 0) & (x <= 1)) and torch.all((t >= 0) & (t <= 1))
    assert not x.requires_grad


def test_probe_fills_replay_buffer_and_decays_epsilon():
    """Each probe stores one transition per probed cell."""
    sampler = _sampler()
    eps0 = sampler.agent.epsilon
    sampler.probe(_HotSpotPDE(), model=None, progress=0.3)
    assert len(sampler.agent.memory) == 8
    assert sampler.agent.epsilon < eps0


def test_probe_every_reuses_state():
    """With probe_every=3 only every third call probes the residual."""
    sampler = _sampler(probe_every=3)
    for _ in range(4):
        sampler.sample(_HotSpotPDE(), model=None, num_points=10)
    assert len(sampler.agent.memory) == 2 * 8


def test_learned_policy_prefers_hot_cells():
    """After training, cells with x > 0.5 get most of the policy mass."""
    torch.manual_seed(0)
    sampler = _sampler()
    for _ in range(150):
        state = sampler.probe(_HotSpotPDE(), model=None, progress=0.5)
    sampler.agent.epsilon = 0.0
    sampler.agent.policy_net.eval()
    probs = sampler.distribution(state)
    hot = probs.reshape(4, 2)[2:].sum()  # x-cells 2 and 3 cover x in (0.5, 1]
    assert hot > 0.7


def test_rejects_mismatched_agent():
    """A pre-built agent must have one action per cell."""
    agent = RLAgent(state_dim=4, action_dim=3, hidden_dim=8)
    with pytest.raises(ValueError, match="does not match"):
        _sampler(agent=agent)
