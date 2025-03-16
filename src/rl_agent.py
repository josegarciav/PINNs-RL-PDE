
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class CollocationRLAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.001):
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_probs = self.model(state_tensor).detach().numpy()[0]
        return np.random.choice(len(action_probs), p=action_probs)

    def update(self, experiences, gamma=0.99):
        loss = 0
        for state, action, reward in reversed(experiences):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action_probs = self.model(state_tensor)
            log_prob = torch.log(action_probs[0, action])
            loss -= log_prob * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
