import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque
import random

class DQNNetwork(nn.Module):
    """Deep Q-Network for point selection."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        dropout: float = 0.1
    ):
        """
        Initialize DQN network.
        
        :param state_dim: State dimension
        :param action_dim: Action dimension
        :param hidden_dim: Hidden layer dimension
        :param num_layers: Number of layers
        :param dropout: Dropout rate
        """
        super().__init__()
        
        # Build network
        layers = []
        
        # Input layer
        layers.append(nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        ))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            layers.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
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
        """
        Forward pass through the network.
        
        :param x: Input tensor
        :return: Q-values
        """
        return self.layers(x)

class ReplayBuffer:
    """Experience replay buffer for training stability."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        :param capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: torch.Tensor, action: int, reward: float,
             next_state: torch.Tensor, done: bool):
        """
        Add experience to buffer.
        
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        Sample batch of experiences.
        
        :param batch_size: Size of batch to sample
        :return: List of experiences
        """
        return random.sample(self.buffer, batch_size)
    
    def __len__(self) -> int:
        """Get current size of buffer."""
        return len(self.buffer)

class RLAgent:
    """Reinforcement Learning agent for point selection."""
    
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
        reward_weights: Optional[Dict[str, float]] = None,
        device: Optional[torch.device] = None
    ):
        """
        Initialize RL agent.
        
        :param state_dim: State dimension
        :param action_dim: Action dimension
        :param hidden_dim: Hidden layer dimension
        :param learning_rate: Learning rate
        :param gamma: Discount factor
        :param epsilon_start: Initial exploration rate
        :param epsilon_end: Final exploration rate
        :param epsilon_decay: Exploration decay rate
        :param memory_size: Size of replay buffer
        :param batch_size: Batch size for training
        :param target_update: Frequency of target network updates
        :param reward_weights: Weights for different reward components
        :param device: Device to place the model on
        """
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Store parameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.target_update = target_update
        self.reward_weights = reward_weights or {
            'residual': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'exploration': 0.1
        }
        
        # Initialize networks
        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size)
        
        # Training statistics
        self.steps = 0
        self.episode_rewards = []
        self.episode_reward = 0
    
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Select an action using epsilon-greedy policy.
        
        :param state: Current state tensor
        :return: Selected action tensor
        """
        if torch.rand(1).item() > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).view(1, -1)
        else:
            return torch.rand(1, 1, device=self.device)  # Random action between 0 and 1
    
    def get_points_from_action(self, action: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert action to collocation points.
        
        :param action: Selected action
        :return: Tuple of (x, t) coordinates
        """
        # Convert action to point coordinates
        x = torch.rand(self.action_dim, 1, device=self.device)
        t = torch.rand(self.action_dim, 1, device=self.device)
        
        # Sort points based on action
        x = x[action:action+1]
        t = t[action:action+1]
        
        return x, t
    
    def compute_reward(
        self,
        residual_loss: float,
        boundary_loss: float,
        initial_loss: float,
        exploration_bonus: float = 0.0
    ) -> float:
        """
        Compute reward from losses.
        
        :param residual_loss: PDE residual loss
        :param boundary_loss: Boundary condition loss
        :param initial_loss: Initial condition loss
        :param exploration_bonus: Bonus for exploring new regions
        :return: Computed reward
        """
        reward = (
            -self.reward_weights['residual'] * residual_loss
            -self.reward_weights['boundary'] * boundary_loss
            -self.reward_weights['initial'] * initial_loss
            +self.reward_weights['exploration'] * exploration_bonus
        )
        return reward
    
    def update(self, state: torch.Tensor, action: int, reward: float,
               next_state: torch.Tensor, done: bool):
        """
        Update agent with new experience.
        
        :param state: Current state
        :param action: Action taken
        :param reward: Reward received
        :param next_state: Next state
        :param done: Whether episode is done
        """
        # Store experience
        self.memory.push(state, action, reward, next_state, done)
        
        # Update episode reward
        self.episode_reward += reward
        
        # If episode is done, store episode reward
        if done:
            self.episode_rewards.append(self.episode_reward)
            self.episode_reward = 0
        
        # Update target network if needed
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Train if enough samples
        if len(self.memory) >= self.batch_size:
            self._train()
        
        # Decay exploration rate
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _train(self):
        """Train the policy network."""
        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.stack(states).to(self.device)
        actions = torch.tensor(actions, device=self.device)
        rewards = torch.tensor(rewards, device=self.device)
        next_states = torch.stack(next_states).to(self.device)
        dones = torch.tensor(dones, device=self.device)
        
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
    
    def save_state(self, path: str):
        """
        Save agent state.
        
        :param path: Path to save the agent
        """
        state = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_rewards': self.episode_rewards,
            'episode_reward': self.episode_reward
        }
        torch.save(state, path)
    
    def load_state(self, path: str):
        """
        Load agent state.
        
        :param path: Path to load the agent from
        """
        state = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(state['policy_net_state_dict'])
        self.target_net.load_state_dict(state['target_net_state_dict'])
        self.optimizer.load_state_dict(state['optimizer_state_dict'])
        self.epsilon = state['epsilon']
        self.steps = state['steps']
        self.episode_rewards = state['episode_rewards']
        self.episode_reward = state['episode_reward']
    
    def get_statistics(self) -> Dict[str, float]:
        """
        Get training statistics.
        
        :return: Dictionary of statistics
        """
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'episode_reward': self.episode_reward,
            'mean_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'std_episode_reward': np.std(self.episode_rewards) if self.episode_rewards else 0.0
        }

class AdaptiveSamplingMixin:
    """Mixin class that provides adaptive collocation point sampling using RL."""
    
    def generate_adaptive_collocation_points(
        self,
        num_points: int,
        rl_agent: Optional['CollocationRLAgent'] = None,
        batch_size: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate collocation points using RL agent for adaptive sampling.
        
        Args:
            num_points: Total number of points to generate
            rl_agent: Optional RL agent for adaptive sampling
            batch_size: Size of batches for RL agent processing
            
        Returns:
            Tuple of (x, t) tensors for collocation points
        """
        if rl_agent is None:
            # Fallback to default sampling if no RL agent
            return self.generate_collocation_points(num_points)
        
        # Generate initial uniform grid for RL agent to sample from
        x = torch.linspace(self.config.domain[0], self.config.domain[1], 100, device=self.device)
        t = torch.linspace(self.config.time_domain[0], self.config.time_domain[1], 100, device=self.device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        
        # Flatten grid points
        grid_points = torch.stack([X.flatten(), T.flatten()], dim=1)
        
        # Process in batches to avoid memory issues
        selected_points = []
        for i in range(0, len(grid_points), batch_size):
            batch = grid_points[i:i + batch_size]
            
            # Get sampling probabilities from RL agent
            with torch.no_grad():
                probs = rl_agent.get_action(batch)
            
            # Normalize probabilities
            probs = torch.softmax(probs, dim=0)
            
            # Sample points based on probabilities
            indices = torch.multinomial(probs, min(batch_size, num_points - len(selected_points)))
            selected_points.append(batch[indices])
        
        # Concatenate selected points
        selected_points = torch.cat(selected_points, dim=0)
        
        # Split into x and t coordinates
        x = selected_points[:, 0]
        t = selected_points[:, 1]
        
        # Add some random noise to avoid exact grid points
        noise_scale = min(
            (self.config.domain[1] - self.config.domain[0]) / 100,
            (self.config.time_domain[1] - self.config.time_domain[0]) / 100
        )
        x = x + torch.randn_like(x) * noise_scale * 0.1
        t = t + torch.randn_like(t) * noise_scale * 0.1
        
        # Clip to domain bounds
        x = torch.clamp(x, self.config.domain[0], self.config.domain[1])
        t = torch.clamp(t, self.config.time_domain[0], self.config.time_domain[1])
        
        return x.reshape(-1, 1), t.reshape(-1, 1)

class CollocationRLAgent(nn.Module):
    """RL agent for adaptive collocation point sampling."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        device: Optional[torch.device] = None
    ):
        """
        Initialize the RL agent.
        
        Args:
            state_dim: Dimension of state space (x, t)
            action_dim: Dimension of action space (sampling probability)
            hidden_dim: Number of hidden units
            num_layers: Number of layers in the network
            learning_rate: Learning rate for optimization
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Decay rate for exploration
            device: Computing device
        """
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Network architecture
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.network = nn.Sequential(*layers).to(self.device)
        
        # Training parameters
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.gamma = gamma
        
        # Exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Get action (sampling probability) for given state.
        
        Args:
            state: State tensor (x, t coordinates)
            
        Returns:
            Action tensor (sampling probability)
        """
        if np.random.random() < self.epsilon:
            # Random action (exploration)
            return torch.randn(state.shape[0], 1, device=self.device)
        else:
            # Network action (exploitation)
            return self.network(state)
    
    def update_epsilon(self, epoch: int):
        """Update exploration rate."""
        self.epsilon = max(
            self.epsilon_end,
            self.epsilon * self.epsilon_decay
        )
    
    def update(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, next_state: torch.Tensor):
        """
        Update the network using Q-learning.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        # Compute current Q-value
        current_q = self.network(state)
        
        # Compute next Q-value
        with torch.no_grad():
            next_q = self.network(next_state)
        
        # Compute target Q-value
        target_q = reward + self.gamma * next_q
        
        # Compute loss and update
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
