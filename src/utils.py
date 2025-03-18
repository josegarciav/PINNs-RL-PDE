import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional, Dict, List
import os
import json
import logging
from datetime import datetime

def setup_logging(log_dir: str = 'logs') -> logging.Logger:
    """
    Setup logging configuration.
    
    :param log_dir: Directory to store log files
    :return: Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f'training_{timestamp}.log')),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def generate_collocation_points(
    num_points: int,
    domain: Tuple[float, float],
    device: Optional[torch.device] = None,
    distribution: str = 'uniform',
    **kwargs
) -> torch.Tensor:
    """
    Generate collocation points with different sampling strategies.
    
    :param num_points: Number of points to generate
    :param domain: Domain range (min, max)
    :param device: Device to place the tensor
    :param distribution: Sampling distribution ('uniform', 'latin_hypercube', 'sobol')
    :param kwargs: Additional parameters for specific distributions
    :return: Tensor of collocation points
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    if distribution == 'uniform':
        points = torch.rand(num_points, 1, device=device) * (domain[1] - domain[0]) + domain[0]
    
    elif distribution == 'latin_hypercube':
        # Latin Hypercube Sampling for better space-filling
        n_bins = int(np.sqrt(num_points))
        points = torch.zeros(num_points, 1, device=device)
        for i in range(num_points):
            bin_idx = i % n_bins
            bin_size = (domain[1] - domain[0]) / n_bins
            points[i] = domain[0] + bin_idx * bin_size + torch.rand(1, device=device) * bin_size
    
    elif distribution == 'sobol':
        # Sobol sequence for quasi-random sampling
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=1)
        points = torch.tensor(
            qmc.scale(sampler.random_base2(m=int(np.log2(num_points))), domain[0], domain[1]),
            device=device
        )
    
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    
    return points

def save_model(model: torch.nn.Module, path: str, config: Optional[Dict] = None):
    """
    Save the trained model and configuration.
    
    :param model: Trained model
    :param path: Path to save the model
    :param config: Optional configuration dictionary
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state
    torch.save(model.state_dict(), path)
    
    # Save configuration if provided
    if config is not None:
        config_path = path.replace('.pth', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)

def load_model(
    model: torch.nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    load_config: bool = True
) -> Tuple[torch.nn.Module, Optional[Dict]]:
    """
    Load a trained model and its configuration.
    
    :param model: Model instance to load weights into
    :param path: Path to the saved model
    :param device: Device to load the model to
    :param load_config: Whether to load the configuration
    :return: Tuple of (loaded model, configuration dictionary)
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model state
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()
    
    # Load configuration if available
    config = None
    if load_config:
        config_path = path.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
    
    return model, config

def plot_pinn_solution(
    model: torch.nn.Module,
    pde,
    domain: Tuple[float, float] = (0, 1),
    resolution: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
):
    """
    Enhanced visualization of the PINN solution with error analysis.
    
    :param model: Trained PINN model
    :param pde: PDE instance
    :param domain: Spatial domain
    :param resolution: Plot resolution
    :param device: Device to use
    :param save_path: Optional path to save the plot
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Generate points
    x = torch.linspace(domain[0], domain[1], resolution).view(-1, 1).to(device)
    t = torch.zeros_like(x).to(device)
    
    # Get predictions
    with torch.no_grad():
        inputs = torch.cat((x, t), dim=1)
        u_pred = model(inputs)
    
    # Get exact solution if available
    u_exact = None
    if hasattr(pde, 'exact_solution'):
        u_exact = pde.exact_solution(x, t)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot predictions
    ax1.plot(x.cpu().numpy(), u_pred.cpu().numpy(), 'b-', label='PINN Prediction')
    if u_exact is not None:
        ax1.plot(x.cpu().numpy(), u_exact.cpu().numpy(), 'r--', label='Exact Solution')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u(x,t=0)')
    ax1.set_title(f'{pde.__class__.__name__} Solution')
    ax1.legend()
    ax1.grid(True)
    
    # Plot error if exact solution is available
    if u_exact is not None:
        error = torch.abs(u_pred - u_exact)
        ax2.plot(x.cpu().numpy(), error.cpu().numpy(), 'g-', label='Absolute Error')
        ax2.set_xlabel('x')
        ax2.set_ylabel('|u_pred - u_exact|')
        ax2.set_title('Error Analysis')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_pinn_3d_solution(
    model: torch.nn.Module,
    pde,
    domain: Tuple[float, float] = (0, 1),
    t_domain: Tuple[float, float] = (0, 1),
    resolution: int = 100,
    device: Optional[torch.device] = None,
    save_path: Optional[str] = None
):
    """
    Enhanced 3D visualization of the PINN solution with error analysis.
    
    :param model: Trained PINN model
    :param pde: PDE instance
    :param domain: Spatial domain
    :param t_domain: Temporal domain
    :param resolution: Plot resolution
    :param device: Device to use
    :param save_path: Optional path to save the plot
    """
    device = device or torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Generate grid
    x = torch.linspace(domain[0], domain[1], resolution).to(device)
    t = torch.linspace(t_domain[0], t_domain[1], resolution).to(device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    # Get predictions
    with torch.no_grad():
        inputs = torch.stack((X.flatten(), T.flatten()), dim=-1)
        u_pred = model(inputs).reshape(resolution, resolution)
    
    # Get exact solution if available
    u_exact = None
    if hasattr(pde, 'exact_solution'):
        u_exact = pde.exact_solution(X, T)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 5))
    
    # Plot PINN prediction
    ax1 = fig.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(
        X.cpu().numpy(),
        T.cpu().numpy(),
        u_pred.cpu().numpy(),
        cmap='viridis'
    )
    ax1.set_title('PINN Prediction')
    ax1.set_xlabel('x')
    ax1.set_ylabel('t')
    ax1.set_zlabel('u(x,t)')
    fig.colorbar(surf1, ax=ax1, shrink=0.5)
    
    # Plot exact solution if available
    if u_exact is not None:
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(
            X.cpu().numpy(),
            T.cpu().numpy(),
            u_exact.cpu().numpy(),
            cmap='plasma'
        )
        ax2.set_title('Exact Solution')
        ax2.set_xlabel('x')
        ax2.set_ylabel('t')
        ax2.set_zlabel('u(x,t)')
        fig.colorbar(surf2, ax=ax2, shrink=0.5)
        
        # Plot error
        ax3 = fig.add_subplot(133, projection='3d')
        error = torch.abs(u_pred - u_exact)
        surf3 = ax3.plot_surface(
            X.cpu().numpy(),
            T.cpu().numpy(),
            error.cpu().numpy(),
            cmap='hot'
        )
        ax3.set_title('Absolute Error')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('|u_pred - u_exact|')
        fig.colorbar(surf3, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history with multiple metrics.
    
    :param history: Dictionary containing training metrics
    :param save_path: Optional path to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    for metric, values in history.items():
        plt.plot(values, label=metric)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
