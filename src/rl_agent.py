
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class CollocationRLAgent(nn.Module):
    """
    Reinforcement Learning agent for adaptive collocation point selection in PINNs.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001, gamma=0.99, device=None):
        super(CollocationRLAgent, self).__init__()
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.gamma = gamma

        # Policy network definition
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

        :param state: Current state representation (numpy array or tensor).
        :return: Action (int).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs = self.policy_net(state_tensor).cpu().numpy()[0]
        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, experiences):
        """
        Update the policy network based on experiences.

        :param experiences: List of (state, action, reward) tuples.
        """
        G = 0
        policy_loss = []

        # Compute discounted rewards and losses
        for state, action, reward in reversed(experiences):
            G = reward + self.gamma * G
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.policy_net(state_tensor)
            log_prob = torch.log(action_probs.squeeze(0)[action])
            policy_loss.append(-log_prob * G)

        # Backpropagation step
        self.optimizer.zero_grad()
        policy_loss = torch.stack(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
