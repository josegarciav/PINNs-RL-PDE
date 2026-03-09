import argparse
import glob
import json
import os
import subprocess
import sys
from datetime import datetime

import dash
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import yaml
from dash import dcc, html
from dash.dependencies import Input, Output, State


# Parse command line arguments for port
def parse_args():
    parser = argparse.ArgumentParser(description="PINNs-RL-PDE Training Monitor Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    return parser.parse_args()


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    [
        html.H1("PINNs-RL-PDE Training Monitor", style={"textAlign": "center"}),
        # Launch trainer button
        html.Div(
            [
                html.Button(
                    "Launch Interactive Trainer",
                    id="launch-trainer-button",
                    style={
                        "backgroundColor": "#2196F3",
                        "color": "white",
                        "padding": "12px 24px",
                        "border": "none",
                        "borderRadius": "4px",
                        "cursor": "pointer",
                        "fontSize": "16px",
                    },
                ),
                html.Div(id="trainer-status", style={"marginTop": "8px", "color": "#666"}),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),
        # Experiment selector and download button row
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Experiment:"),
                        dcc.Dropdown(id="experiment-selector", placeholder="Select experiment..."),
                    ],
                    style={
                        "width": "70%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                    },
                ),
                html.Div(
                    [
                        html.Button(
                            "Download Report",
                            id="download-report-button",
                            style={
                                "backgroundColor": "#4CAF50",
                                "color": "white",
                                "padding": "10px 20px",
                                "border": "none",
                                "borderRadius": "4px",
                                "cursor": "pointer",
                                "fontSize": "16px",
                                "marginTop": "20px",
                            },
                        ),
                        dcc.Download(id="download-report"),
                    ],
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "textAlign": "right",
                    },
                ),
            ],
            style={"marginBottom": "20px", "width": "50%", "margin": "auto"},
        ),
        # Main metrics panel
        html.Div(
            [
                html.Div(
                    [
                        dcc.Graph(id="loss-graph"),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(id="collocation-evolution"),
                    ],
                    style={"width": "50%", "display": "inline-block"},
                ),
            ]
        ),
        # Solution visualization panel
        html.Div(
            [
                html.H2("Solution Visualization", className="section-title"),
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="exact-solution-3d", style={"height": "50vh"}),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="predicted-solution-3d", style={"height": "50vh"}),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                html.Label("Time Point:"),
                                dcc.Slider(
                                    id="time-slider",
                                    min=0,
                                    max=1,
                                    step=0.01,
                                    value=0.5,
                                    marks={i / 10: f"{i/10:.1f}" for i in range(11)},
                                ),
                            ],
                            style={"width": "100%", "padding": "20px"},
                        ),
                    ],
                    className="row",
                ),
            ],
            className="section",
        ),
        # Architecture comparison
        html.H2(
            "Architecture Comparison",
            style={"textAlign": "center", "marginTop": "30px"},
        ),
        dcc.Graph(id="architecture-comparison"),
        # PDE comparison
        html.H2("PDE Comparison", style={"textAlign": "center", "marginTop": "30px"}),
        dcc.Graph(id="pde-comparison"),
        # Experiment metadata
        html.Div(
            [
                html.H2("Experiment Details", style={"textAlign": "center"}),
                html.Pre(
                    id="experiment-details",
                    style={
                        "whiteSpace": "pre-wrap",
                        "wordBreak": "break-all",
                        "backgroundColor": "#f5f5f5",
                        "padding": "10px",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                        "maxHeight": "400px",
                        "overflow": "auto",
                    },
                ),
            ],
            style={"marginTop": "30px"},
        ),
        # Automatic update interval (refreshes experiment list and loss graphs)
        dcc.Interval(
            id="interval-component",
            interval=10 * 1000,  # 10 seconds
            n_intervals=0,
        ),
    ],
    style={"padding": "20px"},
)


# Callback to update experiment list
@app.callback(Output("experiment-selector", "options"), Input("interval-component", "n_intervals"))
def update_experiments(_):
    return get_experiments()


# Callback to launch interactive trainer
@app.callback(
    Output("trainer-status", "children"),
    Input("launch-trainer-button", "n_clicks"),
    prevent_initial_call=True,
)
def launch_trainer(n_clicks):
    if not n_clicks:
        return ""
    try:
        trainer_path = os.path.join(os.path.dirname(__file__), "interactive_trainer.py")

        # Find a Python interpreter with working tkinter.
        # uv-managed Python 3.14 ships without Tcl/Tk, so sys.executable may not work.
        python_cmd = None
        candidates = [
            sys.executable,
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "pinn", "bin", "python3"),
            "python3",
            "python",
        ]
        for candidate in candidates:
            try:
                result = subprocess.run(
                    [candidate, "-c", "import tkinter"],
                    capture_output=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    python_cmd = candidate
                    break
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        if python_cmd is None:
            return "Error: No Python with tkinter found. Install tk: brew install python-tk"

        proc = subprocess.Popen(
            [python_cmd, trainer_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        # Wait briefly to check if the process crashes on startup
        try:
            proc.wait(timeout=2)
            # If we get here, process exited within 2 seconds — it crashed
            stderr = proc.stderr.read().decode(errors="replace").strip()
            return f"Error: Trainer exited immediately. {stderr}"
        except subprocess.TimeoutExpired:
            # Process is still running after 2s — it launched successfully
            # Close stderr pipe so it doesn't block the subprocess
            proc.stderr.close()
            return "Interactive Trainer launched."
    except Exception as e:
        return f"Error launching trainer: {e}"


# Callback to update main graphs
@app.callback(
    [
        Output("loss-graph", "figure"),
        Output("collocation-evolution", "figure"),
        Output("experiment-details", "children"),
    ],
    [Input("experiment-selector", "value"), Input("interval-component", "n_intervals")],
)
def update_graphs(experiment, _):
    if not experiment:
        return {}, {}, "No experiment selected"

    # Load training data
    experiment_details = "Experiment not found"
    try:
        # Try to find history file
        history_file = os.path.join(experiment, "history.json")
        if not os.path.exists(history_file):
            # Only look in the experiment directory, not in subfolders
            print(f"history.json not found in experiment directory: {experiment}")
            return {}, {}, "No history.json found in experiment directory"

        with open(history_file, "r") as f:
            history = json.load(f)

        # Load metadata once and reuse
        metadata_file = os.path.join(experiment, "metadata.json")
        metadata = None
        early_stopping_triggered = False
        training_completed = False

        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

                # Add training time in minutes to the display if available
                if "training_time_minutes" in metadata:
                    time_in_minutes = metadata["training_time_minutes"]
                    if time_in_minutes is not None:
                        formatted_time = f"{time_in_minutes:.2f} minutes"
                        metadata["formatted_training_time"] = formatted_time

                experiment_details = json.dumps(metadata, indent=2)
                # Check if early stopping was triggered
                early_stopping_triggered = metadata.get("early_stopping_triggered", False)
                # Check if training is marked as completed
                training_completed = "end_time" in metadata
        else:
            # Create basic metadata from config
            config_file = os.path.join(experiment, "config.yaml")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    experiment_details = f"Experiment: {os.path.basename(experiment)}\nConfiguration:\n{json.dumps(config, indent=2)}"

        # Get data for x-axis (epochs)
        train_epochs = list(range(1, len(history["train_loss"]) + 1))

        # Determine if training has finished - check for both early stopping and natural completion
        final_epoch = len(history["train_loss"])
        total_epochs = final_epoch  # Default to final_epoch

        # Get total epochs and validation frequency from metadata or config
        val_frequency = 10  # Default validation frequency

        if metadata is not None:
            if "training_params" in metadata:
                if "num_epochs" in metadata["training_params"]:
                    total_epochs = metadata["training_params"]["num_epochs"]
                if "validation_frequency" in metadata["training_params"]:
                    val_frequency = metadata["training_params"]["validation_frequency"]
        else:
            config_file = os.path.join(experiment, "config.yaml")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    if "training" in config:
                        if "num_epochs" in config["training"]:
                            total_epochs = config["training"]["num_epochs"]
                        if "validation_frequency" in config["training"]:
                            val_frequency = config["training"]["validation_frequency"]

        # Create training loss trace
        loss_fig = go.Figure()
        loss_fig.add_trace(
            go.Scatter(x=train_epochs, y=history["train_loss"], name="Training Loss")
        )

        # Add validation loss if available
        if "val_loss" in history and history["val_loss"]:
            val_epochs = list(
                range(
                    val_frequency,
                    len(history["val_loss"]) * val_frequency + 1,
                    val_frequency,
                )
            )
            val_epochs = val_epochs[
                : len(history["val_loss"])
            ]  # Make sure we don't exceed the number of validation points

            loss_fig.add_trace(
                go.Scatter(x=val_epochs, y=history["val_loss"], name="Validation Loss")
            )

        # Add component losses if available
        for component in ["residual_loss", "boundary_loss", "initial_loss"]:
            if component in history and history[component]:
                # Use the same x-axis values as validation loss
                if "val_loss" in history and history["val_loss"]:
                    component_epochs = val_epochs[: len(history[component])]
                else:
                    component_epochs = list(range(1, len(history[component]) + 1))

                loss_fig.add_trace(
                    go.Scatter(
                        x=component_epochs,
                        y=history[component],
                        name=component.replace("_", " ").title(),
                    )
                )

        # Add training status annotation
        status_text = ""
        if early_stopping_triggered:
            status_text = f"Early stopping triggered at epoch {final_epoch}/{total_epochs}"
        elif training_completed:
            status_text = f"Training completed ({final_epoch} epochs)"

        if status_text:
            loss_fig.add_annotation(
                text=status_text,
                x=0.5,
                y=0.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=12, color="red"),
                bordercolor="red",
                borderwidth=1,
                borderpad=4,
                bgcolor="white",
            )

        loss_fig.update_layout(
            title="Training Progress",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        # Find the most recent collocation visualization
        vis_dir = "visualizations"
        if not os.path.exists(vis_dir):
            # Check if visualizations are stored in experiment directory
            vis_dir = os.path.join(experiment, "visualizations")
            if not os.path.exists(vis_dir):
                # Try to find any visualization files in the experiment directory
                vis_files = glob.glob(
                    os.path.join(experiment, "**", "*collocation*.png"), recursive=True
                )
                if vis_files:
                    # Use the most recent one
                    vis_files.sort(key=os.path.getmtime, reverse=True)
                    vis_path = vis_files[0]

                    # Create a figure with the image
                    try:
                        from PIL import Image

                        img = np.array(Image.open(vis_path))
                        colloc_fig = px.imshow(img)
                        colloc_fig.update_layout(
                            title="Collocation Points Distribution",
                            coloraxis_showscale=False,
                        )
                        # Remove axis labels and ticks
                        colloc_fig.update_xaxes(
                            showticklabels=False, showgrid=False, zeroline=False
                        )
                        colloc_fig.update_yaxes(
                            showticklabels=False, showgrid=False, zeroline=False
                        )

                        return loss_fig, colloc_fig, experiment_details
                    except Exception as e:
                        print(f"Error loading visualization image: {e}")

        # Look for visualization files in the visualization directory
        if os.path.exists(vis_dir):
            latest_vis = sorted(
                [
                    f
                    for f in os.listdir(vis_dir)
                    if f.endswith(".png")
                    and (
                        f.startswith("latest_collocation_evolution")
                        or f.startswith("latest_density_heatmap")
                        or f.startswith("collocation_evolution_epoch")
                        or f.startswith("final_collocation_evolution")
                    )
                ],
                key=lambda x: os.path.getmtime(os.path.join(vis_dir, x)),
                reverse=True,
            )

            if latest_vis:
                # Use the most recent visualization
                vis_path = os.path.join(vis_dir, latest_vis[0])

                try:
                    # Create a figure with the image
                    from PIL import Image

                    img = np.array(Image.open(vis_path))
                    colloc_fig = px.imshow(img)
                    colloc_fig.update_layout(
                        title="Collocation Points Distribution",
                        coloraxis_showscale=False,
                    )
                    # Remove axis labels and ticks
                    colloc_fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                    colloc_fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

                    return loss_fig, colloc_fig, experiment_details
                except Exception as e:
                    print(f"Error loading visualization image: {e}")

        # Create empty figure if no visualization available
        colloc_fig = go.Figure()
        colloc_fig.add_annotation(
            text="No collocation visualization available",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16),
        )
        colloc_fig.update_layout(
            title="Collocation Points Distribution",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        )

        return loss_fig, colloc_fig, experiment_details

    except Exception as e:
        print(f"Error loading data: {e}")
        return {}, {}, f"Error loading experiment data: {str(e)}"


# Callback to update architecture comparison
@app.callback(
    Output("architecture-comparison", "figure"),
    [Input("interval-component", "n_intervals")],
)
def update_architecture_comparison(_):
    # Find all experiments and group by architecture
    architectures = {}

    # Search for experiment directories
    experiment_dirs = []
    if os.path.exists("experiments"):
        experiment_dirs.extend(
            [
                os.path.join("experiments", d)
                for d in os.listdir("experiments")
                if os.path.isdir(os.path.join("experiments", d))
            ]
        )

    # Group experiments by architecture
    for exp_dir in experiment_dirs:
        # Try to find architecture information in metadata or config
        architecture = None

        # Check metadata.json first
        metadata_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if "architecture" in metadata:
                        architecture = metadata.get("architecture")
                    elif "config" in metadata and "model" in metadata["config"]:
                        architecture = metadata["config"]["model"].get("architecture")
            except Exception:
                pass

        # If not found, check config.yaml
        if not architecture:
            config_file = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        if "model" in config and "architecture" in config["model"]:
                            architecture = config["model"]["architecture"]
                except Exception:
                    pass

        # If still not found, try to infer from directory name
        if not architecture:
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split("_")
            if len(parts) >= 3:
                architecture = parts[2]  # Assuming architecture is in this position
            else:
                # Use unknown as fallback
                architecture = "unknown"

        # Load history.json
        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            # Skip this experiment if history.json not found
            print(f"history.json not found in experiment directory: {exp_dir}")
            continue

        try:
            with open(history_file, "r") as f:
                history = json.load(f)

            if architecture not in architectures:
                architectures[architecture] = []

            architectures[architecture].append(
                {
                    "name": os.path.basename(exp_dir),
                    "train_loss": history.get("train_loss", []),
                    "val_loss": (history.get("val_loss", []) if "val_loss" in history else []),
                }
            )
        except Exception:
            continue

    # Create comparison graph
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for arch, experiments in architectures.items():
        for i, exp in enumerate(experiments):
            # Add training loss
            if exp["train_loss"]:
                fig.add_trace(
                    go.Scatter(
                        y=exp["train_loss"],
                        name=f"{arch} - {exp['name']}",
                        line=dict(color=colors[color_idx % len(colors)], dash="solid"),
                    )
                )

                # Add validation loss if available
                if exp["val_loss"]:
                    fig.add_trace(
                        go.Scatter(
                            y=exp["val_loss"],
                            name=f"{arch} - {exp['name']} (Val)",
                            line=dict(color=colors[color_idx % len(colors)], dash="dash"),
                        )
                    )

                color_idx += 1

    fig.update_layout(
        title="Architecture Comparison - Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",  # Log scale for better visualization
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
    )

    return fig


# Callback to update PDE comparison
@app.callback(Output("pde-comparison", "figure"), [Input("interval-component", "n_intervals")])
def update_pde_comparison(_):
    # Find all experiments and group by PDE type
    pdes = {}
    computation_times = {}

    # Search for experiment directories
    experiment_dirs = []
    if os.path.exists("experiments"):
        experiment_dirs.extend(
            [
                os.path.join("experiments", d)
                for d in os.listdir("experiments")
                if os.path.isdir(os.path.join("experiments", d))
            ]
        )

    # Group experiments by PDE type
    for exp_dir in experiment_dirs:
        # Try to find PDE type in metadata or config
        pde_type = None

        # Check metadata.json first
        metadata_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if "pde_type" in metadata:
                        pde_type = metadata.get("pde_type")
                    elif "config" in metadata and "pde" in metadata["config"]:
                        # Try to construct PDE type from config
                        pde_params = metadata["config"]["pde"]
                        if "diffusion_coefficient" in pde_params:
                            pde_type = f"heat_eq_a{pde_params['diffusion_coefficient']}"
            except Exception:
                pass

        # If not found, check config.yaml
        if not pde_type:
            config_file = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        if "pde_type" in config:
                            pde_type = config["pde_type"]
                        elif "pde" in config:
                            # Try to construct PDE type from config
                            pde_params = config["pde"]
                            if "diffusion_coefficient" in pde_params:
                                pde_type = f"heat_eq_a{pde_params['diffusion_coefficient']}"
                except Exception:
                    pass

        # If still not found, try to infer from directory name
        if not pde_type:
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split("_")
            if len(parts) >= 2:
                pde_type = f"{parts[0]}_{parts[1]}"  # Assuming PDE type is in first two parts
            else:
                pde_type = dir_name  # Use directory name as fallback

        # Load history.json
        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            # Try alternative locations
            alt_files = glob.glob(os.path.join(exp_dir, "**", "history.json"), recursive=True)
            if alt_files:
                history_file = alt_files[0]
            else:
                continue

        try:
            with open(history_file, "r") as f:
                history = json.load(f)

            # Load computation time if available
            comp_time = None
            hover_text = f"Experiment: {os.path.basename(exp_dir)}"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        comp_time = metadata.get("training_time_minutes")
                        if comp_time:
                            computation_times[pde_type] = comp_time
                            hover_text = f"Experiment: {os.path.basename(exp_dir)}<br>Training time: {comp_time:.2f} minutes"
                except Exception:
                    pass

            if pde_type not in pdes:
                pdes[pde_type] = []

            pdes[pde_type].append(
                {
                    "name": os.path.basename(exp_dir),
                    "train_loss": history.get("train_loss", []),
                    "val_loss": (history.get("val_loss", []) if "val_loss" in history else []),
                    "computation_time": comp_time,
                    "hover_text": hover_text,
                }
            )
        except Exception:
            continue

    # Create comparison graph
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for pde_type, experiments in pdes.items():
        for i, exp in enumerate(experiments):
            # Add training loss
            if exp["train_loss"]:
                name = f"{pde_type} - {exp['name']}"
                if exp["computation_time"]:
                    name += f" ({exp['computation_time']:.2f} minutes)"

                fig.add_trace(
                    go.Scatter(
                        y=exp["train_loss"],
                        name=name,
                        line=dict(color=colors[color_idx % len(colors)]),
                        hovertext=exp["hover_text"],
                    )
                )

                color_idx += 1

    fig.update_layout(
        title="PDE Comparison - Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",  # Log scale for better visualization
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
    )

    return fig


# Callback to update 3D solution visualizations (only on user interaction, not interval)
@app.callback(
    [
        Output("exact-solution-3d", "figure"),
        Output("predicted-solution-3d", "figure"),
    ],
    [
        Input("experiment-selector", "value"),
        Input("time-slider", "value"),
    ],
)
def update_solution_visualizations(experiment, time_point):
    if not experiment:
        return create_empty_3d_figure("Select an experiment"), create_empty_3d_figure(
            "Select an experiment"
        )

    try:
        # Import necessary modules
        import sys

        import torch

        sys.path.append(".")
        from src.neural_networks import PINNModel
        from src.pdes.pde_base import PDEConfig
        from src.utils.utils import plot_solution

        # Look for model directly in the experiment directory
        model_path = os.path.join(experiment, "final_model.pt")
        config_path = os.path.join(experiment, "config.yaml")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print("Error: Model or config file not found")
            return create_empty_3d_figure("No model data available"), create_empty_3d_figure(
                "No model data available"
            )

        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Create PDE configuration
        pde_type = config.get("pde_type", "unknown").lower()

        # Extract PDE-specific configuration
        pde_config_dict = config.get("pde_configs", {}).get(pde_type, {})
        if not pde_config_dict:
            return create_empty_3d_figure(
                f"No configuration for PDE type: {pde_type}"
            ), create_empty_3d_figure(f"No configuration for PDE type: {pde_type}")

        # Create PDEConfig instance with the correct configuration
        pde_config = PDEConfig(
            name=pde_config_dict.get("name", pde_type),
            dimension=pde_config_dict.get("dimension", 1),
            domain=pde_config_dict.get("domain", [[0, 1]]),
            time_domain=pde_config_dict.get("time_domain", [0, 1]),
            initial_condition=pde_config_dict.get("initial_condition", {}),
            boundary_conditions=pde_config_dict.get("boundary_conditions", {}),
            parameters=pde_config_dict.get("parameters", {}),
            exact_solution=pde_config_dict.get("exact_solution", {}),
        )

        # Import and create appropriate PDE instance
        if "heat" in pde_type:
            from src.pdes.heat_equation import HeatEquation

            pde = HeatEquation(config=pde_config)
        elif "wave" in pde_type:
            from src.pdes.wave_equation import WaveEquation

            pde = WaveEquation(config=pde_config)
        elif "burgers" in pde_type:
            from src.pdes.burgers_equation import BurgersEquation

            pde = BurgersEquation(config=pde_config)
        elif "convection" in pde_type:
            from src.pdes.convection_equation import ConvectionEquation

            pde = ConvectionEquation(config=pde_config)
        elif "kdv" in pde_type:
            from src.pdes.kdv_equation import KdVEquation

            pde = KdVEquation(config=pde_config)
        elif "allen" in pde_type and "cahn" in pde_type:
            from src.pdes.allen_cahn import AllenCahnEquation

            pde = AllenCahnEquation(config=pde_config)
        elif "cahn" in pde_type and "hilliard" in pde_type:
            from src.pdes.cahn_hilliard import CahnHilliardEquation

            pde = CahnHilliardEquation(config=pde_config)
        elif "black" in pde_type or "scholes" in pde_type:
            from src.pdes.black_scholes import BlackScholesEquation

            pde = BlackScholesEquation(config=pde_config)
        elif "pendulum" in pde_type:
            from src.pdes.pendulum_equation import PendulumEquation

            pde = PendulumEquation(config=pde_config)
        else:
            return create_empty_3d_figure(
                f"Unsupported PDE type: {pde_type}"
            ), create_empty_3d_figure(f"Unsupported PDE type: {pde_type}")

        # Load model
        device = torch.device("cpu")

        # Create model configuration using Config
        from src.config import Config, ModelConfig

        # Read model params from the saved config's "model" section (exact training params)
        saved_model = config.get("model", {})
        architecture = saved_model.get("architecture", "feedforward")
        input_dim = saved_model.get("input_dim", 2 if pde_config.dimension == 1 else 3)
        output_dim = saved_model.get("output_dim", 1)

        model_cfg = ModelConfig(
            input_dim=input_dim,
            output_dim=output_dim,
            architecture=architecture,
            hidden_dim=saved_model.get("hidden_dim", 128),
            num_layers=saved_model.get("num_layers", 4),
            activation=saved_model.get("activation", "tanh"),
            dropout=saved_model.get("dropout", 0.0),
            layer_norm=saved_model.get("layer_norm", False),
        )
        # Inject architecture-specific fields from saved model config
        for key in [
            "mapping_size",
            "scale",
            "omega_0",
            "num_heads",
            "num_blocks",
            "latent_dim",
            "hidden_dims",
            "periodic",
        ]:
            if key in saved_model:
                setattr(model_cfg, key, saved_model[key])

        cfg = Config.__new__(Config)
        cfg.device = device
        cfg.model = model_cfg

        try:
            model = PINNModel(config=cfg, device=device)
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
            model.eval()
        except Exception as e:
            return create_empty_3d_figure(f"Error loading model: {str(e)}"), create_empty_3d_figure(
                f"Error loading model: {str(e)}"
            )

        # Generate visualization data using plot_solution
        num_points = 1000
        try:
            exact_fig, predicted_fig = plot_solution(
                model=model,
                pde=pde,
                num_points=num_points,
                time_point=time_point,
                return_figs=True,
            )
            return exact_fig, predicted_fig
        except Exception as e:
            return create_empty_3d_figure(
                f"Error generating visualization: {str(e)}"
            ), create_empty_3d_figure(f"Error generating visualization: {str(e)}")

    except Exception as e:
        return create_empty_3d_figure(f"Error: {str(e)}"), create_empty_3d_figure(
            f"Error: {str(e)}"
        )


def create_empty_3d_figure(message):
    """Create an empty 3D figure with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        title="Solution Visualization",
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        ),
    )
    return fig


def get_experiment_name(experiment_path):
    """Extract a readable name from the experiment path."""
    try:
        # Get the last part of the path
        name = os.path.basename(experiment_path)

        # Try to parse the components (timestamp_architecture_rl-status)
        parts = name.split("_")
        if len(parts) >= 3:
            timestamp = parts[0]
            arch = parts[1]
            rl_status = parts[2]

            # Format timestamp
            timestamp = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"

            return f"{timestamp} ({arch}, {rl_status})"
        else:
            return name
    except Exception:
        return name


@app.callback(
    Output("download-report", "data"),
    Input("download-report-button", "n_clicks"),
    State("experiment-selector", "value"),
    prevent_initial_call=True,
)
def download_report(n_clicks, experiment):
    if not experiment or not n_clicks:
        return None

    try:
        # Get current figures and data
        loss_fig = update_graphs(experiment, None)[0]
        collocation_fig = update_graphs(experiment, None)[1]
        exact_solution, predicted_solution = update_solution_visualizations(experiment, 0.5)

        # Load metadata
        metadata = {}
        metadata_file = os.path.join(experiment, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)

        # Convert figures to HTML/JavaScript
        figures = {
            "loss_plot": f"Plotly.newPlot('loss-plot', {loss_fig.to_json()})",
            "collocation_plot": f"Plotly.newPlot('collocation-plot', {collocation_fig.to_json()})",
            "exact_solution": f"Plotly.newPlot('exact-solution', {exact_solution.to_json()})",
            "predicted_solution": f"Plotly.newPlot('predicted-solution', {predicted_solution.to_json()})",
        }

        # Generate HTML report
        html_content = generate_html_report(experiment, figures, metadata)

        # Get experiment name for the filename
        exp_name = os.path.basename(experiment)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pinns_report_{exp_name}_{timestamp}.html"

        return dict(content=html_content, filename=filename, type="text/html")

    except Exception as e:
        print(f"Error generating report: {e}")
        return None


def get_experiments():
    """Get list of all experiments with their details."""
    experiments = []
    experiments_dir = "experiments"

    if os.path.exists(experiments_dir):
        for exp_dir in os.listdir(experiments_dir):
            # Skip .DS_Store and other hidden files
            if exp_dir.startswith("."):
                continue

            exp_path = os.path.join(experiments_dir, exp_dir)
            if os.path.isdir(exp_path):
                # Check if this is a running experiment
                # Check if experiment is truly running
                running_file = os.path.join(exp_path, ".running")
                is_running = os.path.exists(running_file)
                if is_running:
                    stale = False
                    meta_file = os.path.join(exp_path, "metadata.json")
                    history_file = os.path.join(exp_path, "history.json")

                    if os.path.exists(meta_file):
                        # If metadata has end_time, training finished — stale marker
                        try:
                            with open(meta_file, "r") as mf:
                                meta = json.load(mf)
                                if "end_time" in meta:
                                    stale = True
                        except Exception:
                            stale = True
                    elif not os.path.exists(history_file):
                        # No metadata AND no history — crashed/aborted run
                        stale = True
                    else:
                        # Has history but no metadata — check if .running file is old (>1 hour)
                        running_age = datetime.now().timestamp() - os.path.getmtime(running_file)
                        if running_age > 3600:
                            stale = True

                    if stale:
                        try:
                            os.remove(running_file)
                        except OSError:
                            pass
                        is_running = False

                # Get experiment details from directory name
                # Format: "YYYYMMDD_HHMMSS_PDE Name_arch_rl_status"
                # Timestamp is always YYYYMMDD_HHMMSS (15 chars), then underscore
                # PDE name may contain spaces, arch and rl_status are at the end
                parts = exp_dir.split("_")
                if len(parts) >= 3 and len(parts[0]) == 8 and len(parts[1]) == 6:
                    timestamp = f"{parts[0]}_{parts[1]}"
                    remainder = exp_dir[len(timestamp) + 1 :]

                    # Check for known RL suffixes to parse from the end
                    if remainder.endswith("_no_rl"):
                        rl_status = "no_rl"
                        middle = remainder[: -len("_no_rl")]
                    elif remainder.endswith("_rl"):
                        rl_status = "rl"
                        middle = remainder[: -len("_rl")]
                    else:
                        rl_status = None
                        middle = remainder

                    # Middle should be "PDE Name_arch" — arch is the last underscore-token
                    if "_" in middle:
                        last_underscore = middle.rfind("_")
                        pde_name = middle[:last_underscore]
                        arch_name = middle[last_underscore + 1 :]
                    else:
                        pde_name = middle if middle else "Unknown PDE"
                        arch_name = "unknown"

                    # Format the display name
                    display_name = f"{timestamp} - {pde_name} ({arch_name})"
                    if rl_status:
                        display_name += f" [{rl_status}]"
                else:
                    display_name = exp_dir

                if is_running:
                    display_name += " (Running)"

                experiments.append({"label": display_name, "value": exp_path})

    # Sort experiments by name (which includes timestamp) in reverse order
    experiments.sort(key=lambda x: x["label"], reverse=True)
    return experiments


def generate_html_report(experiment_path, figures, metadata):
    """Generate an interactive HTML report for the experiment."""
    # Get experiment details
    config_path = os.path.join(experiment_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get experiment name with architecture and RL status
    exp_name = get_experiment_name(experiment_path)

    # Extract RL status
    rl_enabled = config.get("rl", {}).get("enabled", False)
    architecture = config.get("model", {}).get("architecture", "Unknown")

    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PINNs-RL-PDE Experiment Report: {exp_name}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 40px; }}
            .plot {{ width: 100%; height: 600px; }}
            pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 20px; }}
            .status-badge {{
                display: inline-block;
                padding: 5px 10px;
                border-radius: 15px;
                margin: 5px;
                color: white;
                font-weight: bold;
            }}
            .rl-enabled {{ background-color: #28a745; }}
            .rl-disabled {{ background-color: #dc3545; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>PINNs-RL-PDE Experiment Report</h1>
                <h2>{exp_name}</h2>
                <div>
                    <span class="status-badge {('rl-enabled' if rl_enabled else 'rl-disabled')}">
                        RL {('Enabled' if rl_enabled else 'Disabled')}
                    </span>
                    <span class="status-badge" style="background-color: #007bff;">
                        {architecture}
                    </span>
                </div>
            </div>
            <div class="section">
                <h2>Experiment Details</h2>
                <pre>{json.dumps(metadata, indent=2)}</pre>
            </div>
            <div class="section">
                <h2>Training Progress</h2>
                <div id="loss-plot" class="plot"></div>
            </div>
            <div class="section">
                <h2>Collocation Points Distribution</h2>
                <div id="collocation-plot" class="plot"></div>
            </div>
            <div class="section">
                <h2>Solution Visualization</h2>
                <div class="row">
                    <div id="exact-solution" class="plot"></div>
                    <div id="predicted-solution" class="plot"></div>
                </div>
            </div>
        </div>
        <script>
            {figures['loss_plot']}
            {figures['collocation_plot']}
            {figures['exact_solution']}
            {figures['predicted_solution']}
        </script>
    </body>
    </html>
    """
    return html_template


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    port = args.port

    print(f"Starting PINNs-RL-PDE Training Monitor on port {port}")
    print(f"Open http://127.0.0.1:{port}/ in your browser")

    # Try different ports if the specified one is in use
    max_retries = 3

    for attempt in range(max_retries):
        try:
            app.run(debug=False, port=port)
            break
        except Exception as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is in use. Trying port {port+1}...")
                port += 1
                if attempt == max_retries - 1:
                    print(f"Could not find available port after {max_retries} attempts.")
                    print("Please close any running dashboards and try again.")
            else:
                print(f"Error starting dashboard: {e}")
                break
