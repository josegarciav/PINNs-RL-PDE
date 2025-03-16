
import torch
import numpy as np
import matplotlib.pyplot as plt


def generate_collocation_points(num_points, domain=(0, 1)):
    """
    Generate collocation points uniformly within the domain.
    """
    points = torch.linspace(domain[0], domain[1], num_points).view(-1, 1)
    return points

def plot_solution(model, pde, domain=(0, 1), resolution=100):
    """
    Visualize the learned solution of the PINN.
    """
    x = torch.linspace(domain[0], domain[1], resolution).view(-1, 1)
    t = torch.zeros_like(x)  # Assuming t=0 for visualization
    with torch.no_grad():
        u_pred = model(torch.cat((x, t), dim=1))

    plt.figure(figsize=(8, 5))
    plt.plot(x.numpy(), u_pred.numpy(), label="PINN Solution")
    plt.xlabel("x")
    plt.ylabel("u(x, t=0)")
    plt.title(f"Solution of {pde.__class__.__name__}")
    plt.legend()
    plt.show()

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
