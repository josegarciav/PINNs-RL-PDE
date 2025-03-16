
import torch
import numpy as np
import matplotlib.pyplot as plt

def generate_collocation_points(num_points, domain=(0, 1), device=None):
    """
    Generate collocation points uniformly within the domain.
    
    :param num_points: Number of points to generate.
    :param domain: Domain as a tuple (min, max).
    :param device: Torch device (CPU/MPS/GPU).
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    points = torch.rand(num_points, 1, device=device) * (domain[1] - domain[0]) + domain[0]
    return points

def save_model(model, path="pinn_model.pth"):
    """
    Save the trained PINN model.
    
    :param model: Trained PINN model.
    :param path: Path to save the model file.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path="pinn_model.pth", device=None):
    """
    Load a pre-trained PINN model.
    
    :param model: Uninitialized PINN model.
    :param path: Path to load the model file from.
    :param device: Device to map the loaded model parameters.
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

def plot_pinn_solution(model, pde, domain=(0, 1), resolution=100, device=None):
    """
    Visualize the PINN-predicted solution.
    
    :param model: Trained PINN model.
    :param pde: PDE class instance (for naming and boundary conditions).
    :param domain: Spatial domain tuple.
    :param resolution: Resolution for plotting.
    :param device: Torch device (CPU/MPS/GPU).
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    x = torch.linspace(domain[0], domain[1], resolution).view(-1, 1).to(device)
    t = torch.zeros_like(x).to(device)  # assuming initial condition at t=0

    with torch.no_grad():
        inputs = torch.cat((x, t), dim=1)
        u_pred = model(inputs)

    plt.figure(figsize=(8, 5))
    plt.plot(x.cpu().numpy(), u_pred.cpu().numpy(), label="PINN Prediction")
    plt.xlabel("x")
    plt.ylabel("u(x,t=0)")
    plt.title(f"{pde.__class__.__name__} Solution using PINNs")
    plt.legend()
    plt.grid()
    plt.show()
