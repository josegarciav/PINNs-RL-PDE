"""Tests for the DQN building blocks in :mod:`pinnrl.rl.rl_agent`."""

import torch

from pinnrl.rl.rl_agent import DQNNetwork, ReplayBuffer, RLAgent


def test_dqn_forward_shape_and_init():
    """The Q-network maps a batch of states to one value per action, zero biases."""
    net = DQNNetwork(state_dim=4, action_dim=7, hidden_dim=16)
    out = net(torch.randn(5, 4))
    assert out.shape == (5, 7)
    for m in net.modules():
        if isinstance(m, torch.nn.Linear):
            assert torch.all(m.bias == 0)


def test_replay_buffer_capacity_and_sample():
    """The buffer is FIFO-bounded and samples without replacement."""
    buf = ReplayBuffer(capacity=3)
    for i in range(5):
        buf.push(torch.zeros(2), i, float(i), torch.zeros(2), False)
    assert len(buf) == 3
    actions = sorted(a for _, a, _, _, _ in buf.sample(3))
    assert actions == [2, 3, 4]


def test_update_decays_epsilon_and_syncs_target():
    """Every update decays ε to its floor and syncs the target net on schedule."""
    agent = RLAgent(
        state_dim=2,
        action_dim=3,
        hidden_dim=8,
        epsilon_start=1.0,
        epsilon_end=0.5,
        epsilon_decay=0.5,
        batch_size=100,
        target_update=2,
    )
    agent.update(torch.zeros(2), 0, 1.0, torch.zeros(2), False)
    assert agent.epsilon == 0.5
    agent.update(torch.zeros(2), 1, 1.0, torch.zeros(2), False)
    assert agent.epsilon == 0.5  # clamped at epsilon_end
    assert agent.steps == 2
    for p, q in zip(agent.policy_net.parameters(), agent.target_net.parameters(), strict=False):
        assert torch.equal(p, q)


def test_bandit_regression_learns_rewards():
    """With gamma=0 the Q-values regress onto per-action rewards."""
    torch.manual_seed(0)
    agent = RLAgent(
        state_dim=1, action_dim=2, hidden_dim=16, learning_rate=1e-2, gamma=0.0, batch_size=16
    )
    agent.policy_net.eval()  # disable dropout so the check is deterministic
    state = torch.zeros(1)
    for i in range(400):
        action = i % 2
        agent.update(state, action, 2.0 if action == 1 else 0.0, state, False)
    with torch.no_grad():
        q = agent.policy_net(state.unsqueeze(0)).squeeze(0)
    assert q[1] > q[0] + 1.0
