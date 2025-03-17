
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CollocationRLAgent(nn.Module):
    """
    Reinforcement Learning agent for adaptive collocation point selection in PINNs.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001, gamma=0.99, device=None):
        super().__init__()
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.gamma = gamma
        self.action_dim = action_dim

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

    def select_action(self, state):
        """
        Select an action based on the current policy.

        :param state: Current state (numpy array or tensor).
        :return: Selected action (integer index).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs = self.policy_net(state_tensor).cpu().detach().numpy().flatten()
        return np.random.choice(self.action_dim, p=action_probs)

    def select_points(self, num_points, domain):
        """
        Select collocation points adaptively based on the RL policy.

        :param num_points: Number of points to generate.
        :param domain: Spatial domain tuple.
        :return: Collocation points as tensor of shape [num_points, 2].
        """
        # Properly dimensioned state (example: [mean domain position, mean time position])
        state = torch.tensor([
            (domain[0] + domain[1]) / 2,  # mean spatial domain as example state
            0.5  # example mean time
        ]).to(self.device).cpu().numpy()

        action = self.select_action(state)

        # Simple logic based on selected action
        if action == 0:
            # Uniform sampling
            x = torch.rand(num_points, 1, device=self.device) * (domain[1] - domain[0]) + domain[0]
        else:
            # Boundary-focused sampling
            x_boundary_left = torch.ones(num_points // 2, 1, device=self.device) * domain[0]
            x_boundary_right = torch.ones(num_points - num_points // 2, 1, device=self.device) * domain[1]
            x = torch.cat([x_boundary_left, x_boundary_right], dim=0)

        t = torch.rand(num_points, 1, device=self.device)  # Random time points uniformly
        collocation_points = torch.cat([x, t], dim=1)

        return collocation_points

    def update_policy(self, experiences):
        """
        Update policy network using collected experiences.

        :param experiences: List of (state, action, reward) tuples.
        """
        G = 0
        policy_loss = []

        # Compute discounted rewards and accumulate loss
        for state, action, reward in reversed(experiences):
            G = reward + self.gamma * G
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(state_tensor)
            log_prob = torch.log(action_probs[0, action])
            policy_loss.append(-log_prob * G)

        # Optimize policy
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()
        self.optimizer.step()
