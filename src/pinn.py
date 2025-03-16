
import torch
import torch.nn as nn

class PINNModel(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model.
    """

    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, num_layers=4, activation=nn.Tanh, device=None):
        super(PINNModel, self).__init__()
        self.device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        layers = [nn.Linear(input_dim, hidden_dim), activation()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), activation()])
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        """
        Forward pass through the PINN model.

        :param x: Input tensor of shape (N, input_dim).
        :return: Predicted PDE solution tensor.
        """
        x = x.to(self.device)
        return self.net(x)
