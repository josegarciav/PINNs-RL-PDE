"""Utility functions for the PINN framework."""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List, Union, Any
import os
import json
import logging
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Type aliases for better readability
Array = Union[np.ndarray, torch.Tensor]
ArrayLike = Union[Array, List[float], List[int], Tuple[float, ...], Tuple[int, ...]]


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """
    Setup logging configuration.

    :param log_dir: Directory to store log files
    :return: Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, f"training_{timestamp}.log")),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def generate_collocation_points(
    num_points: int,
    domain: Tuple[float, float],
    device: Optional[torch.device] = None,
    distribution: str = "uniform",
    **kwargs,
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
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    if distribution == "uniform":
        points = (
            torch.rand(num_points, 1, device=device) * (domain[1] - domain[0])
            + domain[0]
        )

    elif distribution == "latin_hypercube":
        # Latin Hypercube Sampling for better space-filling
        n_bins = int(np.sqrt(num_points))
        points = torch.zeros(num_points, 1, device=device)
        for i in range(num_points):
            bin_idx = i % n_bins
            bin_size = (domain[1] - domain[0]) / n_bins
            points[i] = (
                domain[0] + bin_idx * bin_size + torch.rand(1, device=device) * bin_size
            )

    elif distribution == "sobol":
        # Sobol sequence for quasi-random sampling
        from scipy.stats import qmc

        sampler = qmc.Sobol(d=1)
        points = torch.tensor(
            qmc.scale(
                sampler.random_base2(m=int(np.log2(num_points))), domain[0], domain[1]
            ),
            device=device,
        )

    else:
        raise ValueError(f"Unsupported distribution: {distribution}")

    return points


def save_model(
    model: Union[torch.nn.Module, "RLAgent"], path: str, config: Optional[Dict] = None
):
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
    elif hasattr(model, "save_state"):
        model.save_state(path)
    else:
        raise ValueError("Model must be a PyTorch model or RLAgent")

    # Save configuration if provided
    if config is not None:
        config_path = path.replace(".pth", "_config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)


def load_model(
    model: torch.nn.Module,
    path: str,
    device: Optional[torch.device] = None,
    load_config: bool = True,
) -> Tuple[torch.nn.Module, Optional[Dict]]:
    """
    Load a trained model and its configuration.

    :param model: Model instance to load weights into
    :param path: Path to the saved model
    :param device: Device to load the model to
    :param load_config: Whether to load the configuration
    :return: Tuple of (loaded model, configuration dictionary)
    """
    device = device or torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )

    # Load model state
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device).eval()

    # Load configuration if available
    config = None
    if load_config:
        config_path = path.replace(".pth", "_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)

    return model, config


def plot_solution(
    model: "PINNModel",
    pde: "PDEBase",
    num_points: int = 1000,
    save_path: Optional[str] = None,
    use_rl: bool = False,
    rl_agent: Optional["RLAgent"] = None,
    return_figs: bool = False,
    time_point: Optional[float] = None,
):
    """Plot the solution of a PDE using the trained model with interactive 3D visualization.

    Args:
        model: The trained PINN model
        pde: The PDE instance
        num_points: Number of points to use for plotting
        save_path: Optional path to save the plot
        use_rl: Whether to use RL agent for adaptive sampling
        rl_agent: The RL agent for adaptive sampling
        return_figs: Whether to return the figures instead of showing/saving them
        time_point: Optional specific time point for 2D visualization
    """
    if pde.dimension == 2:
        # 2D PDE case (x, y, t)
        x = torch.linspace(
            pde.domain[0][0],
            pde.domain[0][1],
            int(np.sqrt(num_points)),
            device=model.device,
        )
        y = torch.linspace(
            pde.domain[1][0],
            pde.domain[1][1],
            int(np.sqrt(num_points)),
            device=model.device,
        )

        if time_point is not None:
            t = torch.tensor([time_point], device=model.device)
        else:
            t = torch.linspace(
                pde.config.time_domain[0],
                pde.config.time_domain[1],
                int(np.sqrt(num_points)),
                device=model.device,
            )

        X, Y = torch.meshgrid(x, y, indexing="ij")
        points = []
        predictions = []
        exact_solutions = []

        for t_val in t:
            # Create grid for current time
            t_grid = torch.full_like(X, t_val)
            xyz = torch.stack([X.flatten(), Y.flatten(), t_grid.flatten()], dim=1)

            # Get predictions and exact solutions
            with torch.no_grad():
                pred = model(xyz).reshape(X.shape)
                exact = pde.exact_solution(xyz).reshape(X.shape)

            points.append(xyz)
            predictions.append(pred)
            exact_solutions.append(exact)

        # Convert to numpy arrays
        X = X.cpu().numpy()
        Y = Y.cpu().numpy()
        T = t.cpu().numpy()
        predictions = torch.stack(predictions).cpu().numpy()
        exact_solutions = torch.stack(exact_solutions).cpu().numpy()

        # Create exact solution figure
        exact_fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=exact_solutions[0],
                    colorscale="viridis",
                    name="Exact",
                    showscale=False,
                    hoverinfo="x+y+z",
                )
            ]
        )

        # Create predicted solution figure
        predicted_fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=Y,
                    z=predictions[0],
                    colorscale="plasma",
                    name="PINN",
                    showscale=False,
                    hoverinfo="x+y+z",
                )
            ]
        )

        # Add time slider if multiple time points
        if len(t) > 1:
            frames_exact = [
                go.Frame(
                    data=[go.Surface(x=X, y=Y, z=exact_solutions[i])],
                    name=f"t={T[i]:.2f}",
                )
                for i in range(len(t))
            ]
            frames_pred = [
                go.Frame(
                    data=[go.Surface(x=X, y=Y, z=predictions[i])], name=f"t={T[i]:.2f}"
                )
                for i in range(len(t))
            ]
            exact_fig.frames = frames_exact
            predicted_fig.frames = frames_pred

            # Add slider
            sliders = [
                dict(
                    steps=[
                        dict(
                            method="animate",
                            args=[[f"t={T[i]:.2f}"]],
                            label=f"t={T[i]:.2f}",
                        )
                        for i in range(len(t))
                    ],
                    currentvalue=dict(visible=True, prefix="Time: "),
                    len=0.9,
                )
            ]
            exact_fig.update_layout(sliders=sliders)
            predicted_fig.update_layout(sliders=sliders)

    else:
        # 1D PDE case (x, t)
        x = torch.linspace(
            pde.domain[0][0],
            pde.domain[0][1],
            int(np.sqrt(num_points)),
            device=model.device,
        )

        if time_point is not None:
            t = torch.tensor([time_point], device=model.device)
        else:
            t = torch.linspace(
                pde.config.time_domain[0],
                pde.config.time_domain[1],
                int(np.sqrt(num_points)),
                device=model.device,
            )

        X, T = torch.meshgrid(x, t, indexing="ij")
        xt = torch.stack([X.flatten(), T.flatten()], dim=1)

        # Get predictions and exact solutions
        with torch.no_grad():
            u_pred = model(xt).reshape(X.shape)
            u_exact = pde.exact_solution(xt).reshape(X.shape)

        # Move tensors to CPU for plotting
        X = X.cpu().numpy()
        T = T.cpu().numpy()
        U_pred = u_pred.cpu().numpy()
        U_exact = u_exact.cpu().numpy()

        # Create exact solution figure
        exact_fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=T,
                    z=U_exact,
                    colorscale="viridis",
                    name="Exact",
                    showscale=False,
                    hoverinfo="x+y+z",
                )
            ]
        )

        # Create predicted solution figure
        predicted_fig = go.Figure(
            data=[
                go.Surface(
                    x=X,
                    y=T,
                    z=U_pred,
                    colorscale="plasma",
                    name="PINN",
                    showscale=False,
                    hoverinfo="x+y+z",
                )
            ]
        )

    # Common camera settings
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5),
    )

    # Common axis settings
    axis_settings = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
        showspikes=False,
        showbackground=True,
        backgroundcolor="rgba(240, 240, 240, 0.5)",
    )

    # Update layout for exact solution
    exact_fig.update_layout(
        title="Exact Solution",
        scene=dict(
            xaxis=dict(title="x", **axis_settings),
            yaxis=dict(title="t" if pde.dimension == 1 else "y", **axis_settings),
            zaxis=dict(
                title="u(x,t)" if pde.dimension == 1 else "u(x,y,t)", **axis_settings
            ),
            camera=camera,
            dragmode="turntable",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # Update layout for predicted solution
    predicted_fig.update_layout(
        title="PINN Prediction",
        scene=dict(
            xaxis=dict(title="x", **axis_settings),
            yaxis=dict(title="t" if pde.dimension == 1 else "y", **axis_settings),
            zaxis=dict(
                title="u(x,t)" if pde.dimension == 1 else "u(x,y,t)", **axis_settings
            ),
            camera=camera,
            dragmode="turntable",
            aspectmode="cube",
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )

    if return_figs:
        return exact_fig, predicted_fig
    else:
        if save_path:
            exact_fig.write_html(save_path.replace(".html", "_exact.html"))
            predicted_fig.write_html(save_path.replace(".html", "_predicted.html"))
        else:
            exact_fig.show()
            predicted_fig.show()


def plot_architecture_comparison(
    model: "PINNModel",
    pde: "PDEBase",
    num_points: int = 1000,
    save_path: Optional[str] = None,
):
    """Plot comparison of different architecture outputs.

    Args:
        model: The PINN model
        pde: The PDE instance
        num_points: Number of points to use for plotting
        save_path: Optional path to save the plot
    """
    # Generate grid points
    x = torch.linspace(
        pde.domain[0], pde.domain[1], int(np.sqrt(num_points)), device=model.device
    )
    t = torch.linspace(
        pde.config.time_domain[0],
        pde.config.time_domain[1],
        int(np.sqrt(num_points)),
        device=model.device,
    )
    X, T = torch.meshgrid(x, t, indexing="ij")

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

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            f"Exact Solution",
            f"{model.architecture.title()} Network Prediction",
            "Error Distribution",
            "Error Surface",
        ),
        specs=[
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "scene"}, {"type": "scene"}],
        ],
        vertical_spacing=0.1,
        horizontal_spacing=0.1,
    )

    # Plot exact solution
    fig.add_trace(
        go.Surface(
            x=X.numpy(),
            y=T.numpy(),
            z=U_exact,
            colorscale="viridis",
            name="Exact",
            showscale=False,
            hoverinfo="x+y+z",
        ),
        row=1,
        col=1,
    )

    # Plot PINN prediction
    fig.add_trace(
        go.Surface(
            x=X.numpy(),
            y=T.numpy(),
            z=U_pred,
            colorscale="viridis",
            name="PINN",
            showscale=False,
            hoverinfo="x+y+z",
        ),
        row=1,
        col=2,
    )

    # Compute error
    error = np.abs(U_pred - U_exact)

    # Plot error distribution
    fig.add_trace(
        go.Histogram(
            x=error.flatten(),
            name="Error Distribution",
            nbinsx=50,
            histnorm="probability",
        ),
        row=2,
        col=1,
    )

    # Plot error surface
    fig.add_trace(
        go.Surface(
            x=X.numpy(),
            y=T.numpy(),
            z=error,
            colorscale="hot",
            name="Error",
            showscale=False,
            hoverinfo="x+y+z",
        ),
        row=2,
        col=2,
    )

    # Common camera settings
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=1.5),
    )

    # Common axis settings
    axis_settings = dict(
        showgrid=True,
        zeroline=True,
        showline=True,
        showticklabels=True,
        showspikes=False,
        showbackground=True,
        backgroundcolor="rgba(240, 240, 240, 0.5)",
    )

    # Scene settings for each subplot
    scene1 = dict(
        xaxis=dict(title="x", **axis_settings),
        yaxis=dict(title="t", **axis_settings),
        zaxis=dict(title="u(x,t)", **axis_settings),
        camera=camera,
        dragmode="turntable",
        aspectmode="cube",
    )

    scene2 = dict(
        xaxis=dict(title="x", **axis_settings),
        yaxis=dict(title="t", **axis_settings),
        zaxis=dict(title="u(x,t)", **axis_settings),
        camera=camera,
        dragmode="turntable",
        aspectmode="cube",
    )

    scene3 = dict(
        xaxis=dict(title="Error", **axis_settings),
        yaxis=dict(title="Probability", **axis_settings),
        zaxis=dict(title="", **axis_settings),
        camera=camera,
        dragmode="turntable",
        aspectmode="cube",
    )

    scene4 = dict(
        xaxis=dict(title="x", **axis_settings),
        yaxis=dict(title="t", **axis_settings),
        zaxis=dict(title="Error", **axis_settings),
        camera=camera,
        dragmode="turntable",
        aspectmode="cube",
    )

    # Update layout
    fig.update_layout(
        title=f"Architecture Comparison: {model.architecture.title()} Network",
        scene=scene1,
        scene2=scene2,
        scene3=scene3,
        scene4=scene4,
        height=1200,
        width=1200,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=0, r=0, t=50, b=0),
    )

    # Save or show
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def create_interactive_report(
    experiment_dir: str,
    pdes: List["PDEBase"],
    architectures: List[Dict],
    metrics: Dict,
    config: Dict,
    save_path: Optional[str] = None,
):
    """Creates an interactive HTML report with comprehensive experiment results.

    Args:
        experiment_dir: Directory containing experiment results
        pdes: List of PDE instances used in experiments
        architectures: List of architecture configurations
        metrics: Dictionary containing experiment metrics
        config: Configuration dictionary
        save_path: Optional path to save the report
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px

    # Create main figure with tabs
    fig = go.Figure()

    # Add PDE selection dropdown
    pde_names = [pde.__class__.__name__ for pde in pdes]
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=[
                    dict(
                        label=pde_name,
                        method="update",
                        args=[{"visible": [True] * len(pdes)}],
                    )
                    for pde_name in pde_names
                ],
                direction="down",
                showactive=True,
                x=0.1,
                y=1.1,
                xanchor="left",
                yanchor="top",
            ),
            dict(
                buttons=[
                    dict(
                        label=arch["name"],
                        method="update",
                        args=[{"visible": [True] * len(architectures)}],
                    )
                    for arch in architectures
                ],
                direction="down",
                showactive=True,
                x=0.4,
                y=1.1,
                xanchor="left",
                yanchor="top",
            ),
        ]
    )

    # Create tabs for different visualizations
    tabs = [
        dict(label="Solution Comparison", value="solution"),
        dict(label="Training Metrics", value="metrics"),
        dict(label="Computational Efficiency", value="efficiency"),
        dict(label="Configuration", value="config"),
    ]

    # Add solution comparison plots
    for pde in pdes:
        for arch in architectures:
            # Generate solution comparison plot
            plot_solution(model=arch["model"], pde=pde, num_points=1000, save_path=None)

    # Add training metrics plot
    metrics_fig = go.Figure()
    for metric_name, values in metrics.items():
        if metric_name != "computation_time":
            metrics_fig.add_trace(go.Scatter(y=values, name=metric_name))
    metrics_fig.update_layout(title="Training Metrics")

    # Add computational efficiency plot
    efficiency_fig = go.Figure()
    for pde in pdes:
        for arch in architectures:
            efficiency_fig.add_trace(
                go.Bar(
                    x=[pde.__class__.__name__],
                    y=[
                        metrics["computation_time"][
                            f"{pde.__class__.__name__}_{arch['name']}"
                        ]
                    ],
                    name=arch["name"],
                )
            )
    efficiency_fig.update_layout(
        title="Computation Time by PDE and Architecture",
        xaxis_title="PDE",
        yaxis_title="Time (seconds)",
    )

    # Add configuration details
    config_fig = go.Figure()
    config_fig.add_trace(
        go.Table(
            header=dict(values=["Parameter", "Value"]),
            cells=dict(values=[list(config.keys()), list(config.values())]),
        )
    )

    # Combine all plots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Solution Comparison",
            "Training Metrics",
            "Computational Efficiency",
            "Configuration",
        ),
        specs=[
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "table"}],
        ],
    )

    # Update layout
    fig.update_layout(
        title="PINN Experiment Results",
        height=1200,
        width=1600,
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    # Save or show
    if save_path:
        fig.write_html(save_path)
    else:
        fig.show()


def save_training_metrics(
    history: Dict[str, Any],
    experiment_dir: str,
    metadata: Optional[Dict[str, Any]] = None,
):
    """Save training metrics to JSON file.

    Args:
        history: Dictionary containing training metrics
        experiment_dir: Directory to save the metrics
        metadata: Optional metadata about the experiment
    """
    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)

    # Save metrics.json directly in the experiment directory
    metrics_file = os.path.join(experiment_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(history, f, indent=2)

    # Save history.json directly in the experiment directory
    history_file = os.path.join(experiment_dir, "history.json")
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    # Save metadata if provided
    if metadata:
        metadata_file = os.path.join(experiment_dir, "metadata.json")

        # Update existing metadata if it exists
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    existing_metadata = json.load(f)
                # Update with new metadata
                existing_metadata.update(metadata)
                metadata = existing_metadata
            except:
                pass

        # Add timestamp
        metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

    return history_file
