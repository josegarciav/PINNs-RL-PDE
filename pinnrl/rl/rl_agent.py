"""DQN building blocks for the residual-bandit collocation sampler.

:class:`RLAgent` is a small DQN (policy and target network, replay buffer,
ε-greedy schedule). :class:`pinnrl.rl.bandit_sampler.ResidualBanditSampler`
uses it with ``gamma = 0``, i.e. as a contextual bandit over domain cells.
"""

from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class DQNNetwork(nn.Module):
    """MLP mapping a state to one Q-value per action (Linear-LayerNorm-ReLU-Dropout blocks)."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        """Initialize DQN network.

        Args:
            state_dim: State dimension.
            action_dim: Action dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of linear layers, input and output included.
            dropout: Dropout rate after each hidden block.
        """
        super().__init__()

        # Build network
        layers = []

        # Input layer
        layers.append(
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        )

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )

        # Output layer
        layers.append(nn.Linear(hidden_dim, action_dim))

        self.layers = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor.

        Returns:
            Q-values.
        """
        return self.layers(x)


class ReplayBuffer:
    """Experience replay buffer for training stability."""

    def __init__(self, capacity: int):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of experiences to store.
        """
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Add experience to buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> list[tuple]:
        """Sample a batch of experiences without replacement.

        Indices come from torch's generator, so ``torch.manual_seed`` makes
        the draw (and a bandit run) reproducible.

        Args:
            batch_size: Size of batch to sample.

        Returns:
            List of experiences.

        Raises:
            ValueError: If ``batch_size`` exceeds the stored experiences.
        """
        if batch_size > len(self.buffer):
            raise ValueError(
                f"batch_size {batch_size} exceeds the {len(self.buffer)} stored experiences"
            )
        return [self.buffer[i] for i in torch.randperm(len(self.buffer))[:batch_size].tolist()]

    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)


class RLAgent:
    """DQN agent with experience replay and an ε schedule that decays per update."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        learning_rate: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 100,
        device: torch.device | None = None,
    ):
        """Initialize RL agent.

        Args:
            state_dim: State dimension.
            action_dim: Number of discrete actions.
            hidden_dim: Hidden layer dimension.
            learning_rate: Learning rate.
            gamma: Discount factor (``0`` turns the DQN into a contextual bandit).
            epsilon_start: Initial exploration rate.
            epsilon_end: Final exploration rate.
            epsilon_decay: Multiplicative ε decay applied on every :meth:`update`.
            memory_size: Size of replay buffer.
            batch_size: Batch size for training.
            target_update: Updates between target-network syncs.
            device: Device to place the networks on.
        """
        self.device = device or torch.device("cpu")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(memory_size)
        self.steps = 0

    def update(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ):
        """Store a transition, train on a replayed batch, and decay ε.

        Training starts once the buffer holds ``batch_size`` transitions; the
        target network is synced every ``target_update`` calls.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether the episode ended.
        """
        self.memory.push(state, action, reward, next_state, done)
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        if len(self.memory) >= self.batch_size:
            self._train()
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _train(self):
        """One Huber-loss step of the policy net toward ``r + γ (1 - done) max_a Q_target``."""
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=False)

        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Compute current Q-values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute next Q-values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss
        loss = F.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
