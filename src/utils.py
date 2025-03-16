
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_collocation_points(num_points, domain=(0, 1)):
    """
    Generate collocation points uniformly within the domain.
    """
    points = torch.linspace(domain[0], domain[1], num_points).view(-1, 1)
    return points

def save_model(model, filename="pinn_model.pth"):
    """
    Save the trained model.
    """
    torch.save(model.state_dict(), filename)

def load_model(model, filename="pinn_model.pth"):
    """
    Load a pre-trained model.
    """
    model.load_state_dict(torch.load(filename))
    model.eval()

    import torch

def plot_pinn_solution(model, pde, domain=(0, 1), resolution=100):
    """
    Visualize the PINN-predicted solution vs. expected behavior.
    
    :param model: Trained PINN model.
    :param pde: PDE class instance.
    :param domain: Tuple defining the spatial range.
    :param resolution: Number of points for plotting.
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.linspace(domain[0], domain[1], resolution).view(-1, 1).to(device)
    t = torch.zeros_like(x)  # Assume t=0 for initial testing

    with torch.no_grad():
        u_pred = model(torch.cat((x, t), dim=1))

    plt.figure(figsize=(8, 5))
    plt.plot(x.cpu().numpy(), u_pred.cpu().numpy(), label="PINN Prediction", linestyle="dashed")
    plt.xlabel("x")
    plt.ylabel("u(x, t=0)")
    plt.title(f"Solution of {pde.__class__.__name__} using PINNs")
    plt.legend()
    plt.show()
