import torch
import numpy as np
from typing import Tuple, Optional, Dict, List, Union
import os
import json
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def save_model(model: Union[torch.nn.Module, 'RLAgent'], path: str, config: Optional[Dict] = None):
    """
    Save the trained model and configuration.
    
    :param model: Trained model (PyTorch model or RLAgent)
    :param path: Path to save the model
    :param config: Optional configuration dictionary
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    elif hasattr(model, 'save_state'):
        model.save_state(path)
    else:
        raise ValueError("Model must be a PyTorch model or RLAgent")
    
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

def plot_solution(model, pde, num_points=1000, save_path=None, use_rl=False, rl_agent=None):
    """Plot the solution of a PDE using the trained model with interactive 3D visualization.
    
    Args:
        model: The trained PINN model
        pde: The PDE instance
        num_points: Number of points to use for plotting
        save_path: Optional path to save the plot
        use_rl: Whether to use RL agent for adaptive sampling (ignored in this version)
        rl_agent: The RL agent for adaptive sampling (ignored in this version)
    """
    # Generate grid points
    x = torch.linspace(pde.domain[0], pde.domain[1], int(np.sqrt(num_points)), device=model.device)
    t = torch.linspace(pde.config.time_domain[0], pde.config.time_domain[1], int(np.sqrt(num_points)), device=model.device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    # Flatten and stack coordinates
    xt = torch.stack([X.flatten(), T.flatten()], dim=1).to(model.device)
    
    # Get model predictions
    with torch.no_grad():
        u_pred = model(xt)
    
    # Move tensors to CPU for plotting
    X = X.cpu()
    T = T.cpu()
    U_pred = u_pred.reshape(X.shape).cpu().numpy()
    
    # Get exact solution
    x_exact = X.flatten().to(model.device).reshape(-1, 1)
    t_exact = T.flatten().to(model.device).reshape(-1, 1)
    U_exact = pde.exact_solution(x_exact, t_exact).reshape(X.shape).cpu().numpy()
    
    # Create subplots vertically with just two plots
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Exact Solution', 'PINN Prediction'),
        specs=[[{'type': 'scene'}], [{'type': 'scene'}]],
        vertical_spacing=0.05,
    )
    
    # Plot exact solution
    fig.add_trace(
        go.Surface(
            x=X.numpy(),
            y=T.numpy(),
            z=U_exact,
            colorscale='viridis',
            name='Exact',
            showscale=False,
            hoverinfo='x+y+z'
        ),
        row=1, col=1
    )
    
    # Plot PINN prediction
    fig.add_trace(
        go.Surface(
            x=X.numpy(),
            y=T.numpy(),
            z=U_pred,
            colorscale='viridis',
            name='PINN',
            showscale=False,
            hoverinfo='x+y+z'
        ),
        row=2, col=1
    )
    
    # Common camera settings
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5)
    )
    
    # Common axis settings
    axis_settings = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
        showspikes=False,
        showbackground=True,
        backgroundcolor='rgba(240, 240, 240, 0.5)'
    )
    
    # Scene settings for each subplot
    scene1 = dict(
        xaxis=dict(title='x', **axis_settings),
        yaxis=dict(title='t', **axis_settings),
        zaxis=dict(title='u(x,t)', **axis_settings),
        camera=camera,
        dragmode='turntable',
        aspectmode='cube'
    )
    
    scene2 = dict(
        xaxis=dict(title='x', **axis_settings),
        yaxis=dict(title='t', **axis_settings),
        zaxis=dict(title='u(x,t)', **axis_settings),
        camera=camera,
        dragmode='turntable',
        aspectmode='cube'
    )
    
    # Update layout
    fig.update_layout(
        title='Heat Equation Solutions Comparison',
        scene=scene1,
        scene2=scene2,
        height=1200,
        width=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    # Save or show
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()
