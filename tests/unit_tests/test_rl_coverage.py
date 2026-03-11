"""Tests for uncovered paths in src/rl/rl_agent.py."""

import pytest
import torch
import torch.nn as nn

from src.rl.rl_agent import CollocationRLAgent, DQNNetwork, RLAgent


DEVICE = torch.device("cpu")
STATE_DIM = 2
ACTION_DIM = 4
HIDDEN_DIM = 16


# ── DQNNetwork ──────────────────────────────────────────────────────────


def test_dqn_weight_init():
    """Verify that _init_weights applies xavier init and zero biases."""
    net = DQNNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM, num_layers=3, dropout=0.0)

    for m in net.modules():
        if isinstance(m, nn.Linear):
            # Xavier normal produces values with std ~ gain * sqrt(2/(fan_in+fan_out)).
            # Just verify biases are exactly zero (deterministic) and weights are not
            # the default PyTorch kaiming init by checking they were touched.
            assert m.bias is not None
            assert torch.all(m.bias == 0.0), "Biases should be initialized to zero"
            # Weights should not be all zeros (xavier_normal_ won't produce all zeros)
            assert not torch.all(m.weight == 0.0)


def test_dqn_forward():
    """Verify DQN forward pass returns correct shape (covers line 88)."""
    net = DQNNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    x = torch.randn(5, STATE_DIM)
    out = net(x)
    assert out.shape == (5, ACTION_DIM)


# ── RLAgent.get_points_from_action ──────────────────────────────────────


def test_get_points_from_action():
    """Call get_points_from_action and verify output shapes."""
    agent = RLAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )
    x, t = agent.get_points_from_action(action=0)
    assert x.shape == (1, 1)
    assert t.shape == (1, 1)


# ── RLAgent train cycle (update -> _train) ──────────────────────────────


def test_agent_train_cycle():
    """Push enough experiences so update() triggers _train() internally."""
    batch_size = 8
    agent = RLAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        batch_size=batch_size,
        memory_size=200,
        target_update=5,
        device=DEVICE,
    )

    initial_epsilon = agent.epsilon

    # Push batch_size + 1 experiences so _train fires on the last update()
    for i in range(batch_size + 1):
        state = torch.randn(STATE_DIM, device=DEVICE)
        next_state = torch.randn(STATE_DIM, device=DEVICE)
        action = i % ACTION_DIM  # valid discrete action index
        reward = float(-0.1 * (i + 1))
        done = i == batch_size  # last one ends episode
        agent.update(state, action, reward, next_state, done)

    # Epsilon should have decayed (batch_size + 1) times
    assert agent.epsilon < initial_epsilon
    # Steps counter should match
    assert agent.steps == batch_size + 1
    # Episode reward list should have one entry (done=True on last step)
    assert len(agent.episode_rewards) == 1

    # Also verify target net sync fires: push enough to reach target_update
    for i in range(agent.target_update):
        state = torch.randn(STATE_DIM, device=DEVICE)
        next_state = torch.randn(STATE_DIM, device=DEVICE)
        agent.update(state, 0, -0.1, next_state, False)

    # After these extra updates the policy_net should still be functional
    out = agent.policy_net(torch.randn(1, STATE_DIM, device=DEVICE))
    assert out.shape == (1, ACTION_DIM)


# ── RLAgent.get_statistics ──────────────────────────────────────────────


def test_get_statistics():
    """Verify get_statistics returns expected keys and types."""
    agent = RLAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )
    # Add one completed episode for mean/std to be meaningful
    agent.episode_rewards.append(-0.5)
    stats = agent.get_statistics()

    expected_keys = {
        "epsilon",
        "steps",
        "episode_reward",
        "mean_episode_reward",
        "std_episode_reward",
    }
    assert set(stats.keys()) == expected_keys
    assert stats["epsilon"] == agent.epsilon
    assert stats["steps"] == agent.steps
    assert isinstance(stats["mean_episode_reward"], float)


def test_get_statistics_empty_episodes():
    """get_statistics with no completed episodes should return 0 for mean/std."""
    agent = RLAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )
    stats = agent.get_statistics()
    assert stats["mean_episode_reward"] == 0.0
    assert stats["std_episode_reward"] == 0.0


# ── RLAgent.get_sampling_density ────────────────────────────────────────


def test_get_sampling_density():
    """Verify get_sampling_density returns correct dict keys and shapes."""
    agent = RLAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )
    # Force exploitation path so select_action goes through policy_net
    agent.epsilon = 0.0

    result = agent.get_sampling_density()
    assert "x" in result
    assert "t" in result
    assert "density" in result
    assert result["x"].shape == (100,)
    assert result["t"].shape == (100,)
    assert result["density"].shape == (100, 100)


# ── CollocationRLAgent exploit path ─────────────────────────────────────


def test_collocation_agent_exploit():
    """With epsilon=0 the agent should use the network (exploitation)."""
    agent = CollocationRLAgent(
        state_dim=STATE_DIM,
        action_dim=1,
        hidden_dim=HIDDEN_DIM,
        device=DEVICE,
    )
    agent.epsilon = 0.0  # Force exploitation

    state = torch.randn(10, STATE_DIM, device=DEVICE)
    action = agent.get_action(state)

    assert action.shape == (10, 1)
    # Exploitation output should be deterministic
    action2 = agent.get_action(state)
    assert torch.allclose(action, action2)


# ── CollocationRLAgent.update ───────────────────────────────────────────


def test_collocation_agent_update():
    """Call CollocationRLAgent.update and verify loss step runs."""
    agent = CollocationRLAgent(
        state_dim=STATE_DIM,
        action_dim=1,
        hidden_dim=HIDDEN_DIM,
        learning_rate=0.01,
        device=DEVICE,
    )

    # Snapshot weights before update
    params_before = [p.clone() for p in agent.parameters()]

    state = torch.randn(5, STATE_DIM, device=DEVICE)
    action = torch.randn(5, 1, device=DEVICE)
    reward = torch.ones(5, 1, device=DEVICE)
    next_state = torch.randn(5, STATE_DIM, device=DEVICE)

    agent.update(state, action, reward, next_state)

    # At least some parameters should have changed
    any_changed = any(
        not torch.equal(p_before, p_after)
        for p_before, p_after in zip(params_before, agent.parameters())
    )
    assert any_changed, "Network parameters should change after update"
