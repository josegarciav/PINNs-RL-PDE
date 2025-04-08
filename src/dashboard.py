import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
import glob
import argparse
import sys
import base64
from pathlib import Path
import yaml


# Parse command line arguments for port
def parse_args():
    parser = argparse.ArgumentParser(
        description="PINNs-RL-PDE Training Monitor Dashboard"
    )
    parser.add_argument(
        "--port", type=int, default=8050, help="Port to run the dashboard on"
    )
    return parser.parse_args()


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(
    [
        html.H1("PINNs-RL-PDE Training Monitor", style={"textAlign": "center"}),
        # Experiment selector and download button row
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Select Experiment:"),
                        dcc.Dropdown(
                            id="experiment-selector", placeholder="Select experiment..."
                        ),
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
                                dcc.Graph(
                                    id="exact-solution-3d", style={"height": "50vh"}
                                ),
                            ],
                            className="six columns",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="predicted-solution-3d", style={"height": "50vh"}
                                ),
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
        # Automatic update interval
        dcc.Interval(
            id="interval-component",
            interval=5 * 1000,  # in milliseconds
            n_intervals=0,
            disabled=True,
        ),
    ],
    style={"padding": "20px"},
)


# Callback to update experiment list
@app.callback(
    Output("experiment-selector", "options"),
    Input("interval-component", "n_intervals")
)
def update_experiments(_):
    return get_experiments()


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

        # Load metadata if available
        metadata_file = os.path.join(experiment, "metadata.json")
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
                early_stopping_triggered = metadata.get(
                    "early_stopping_triggered", False
                )
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

        # Try to get total epochs from metadata if available
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                if (
                    "training_params" in metadata
                    and "num_epochs" in metadata["training_params"]
                ):
                    total_epochs = metadata["training_params"]["num_epochs"]
        else:
            # Try to get from config
            config_file = os.path.join(experiment, "config.yaml")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = yaml.safe_load(f)
                    if "training" in config and "num_epochs" in config["training"]:
                        total_epochs = config["training"]["num_epochs"]

        # Create training loss trace
        loss_fig = go.Figure()
        loss_fig.add_trace(
            go.Scatter(x=train_epochs, y=history["train_loss"], name="Training Loss")
        )

        # Add validation loss if available
        if "val_loss" in history and history["val_loss"]:
            # For validation, we need to account for validation frequency
            val_frequency = 10  # Default validation frequency

            # Try to get validation frequency from metadata or config
            if os.path.exists(metadata_file):
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    if (
                        "training_params" in metadata
                        and "validation_frequency" in metadata["training_params"]
                    ):
                        val_frequency = metadata["training_params"][
                            "validation_frequency"
                        ]
            else:
                # Try to get from config
                config_file = os.path.join(experiment, "config.yaml")
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        config = yaml.safe_load(f)
                        if (
                            "training" in config
                            and "validation_frequency" in config["training"]
                        ):
                            val_frequency = config["training"]["validation_frequency"]

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
            status_text = (
                f"Early stopping triggered at epoch {final_epoch}/{total_epochs}"
            )
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
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
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
                    colloc_fig.update_xaxes(
                        showticklabels=False, showgrid=False, zeroline=False
                    )
                    colloc_fig.update_yaxes(
                        showticklabels=False, showgrid=False, zeroline=False
                    )

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
            except:
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
                except:
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
                    "val_loss": (
                        history.get("val_loss", []) if "val_loss" in history else []
                    ),
                }
            )
        except:
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
                            line=dict(
                                color=colors[color_idx % len(colors)], dash="dash"
                            ),
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
@app.callback(
    Output("pde-comparison", "figure"), [Input("interval-component", "n_intervals")]
)
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
            except:
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
                                pde_type = (
                                    f"heat_eq_a{pde_params['diffusion_coefficient']}"
                                )
                except:
                    pass

        # If still not found, try to infer from directory name
        if not pde_type:
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split("_")
            if len(parts) >= 2:
                pde_type = (
                    f"{parts[0]}_{parts[1]}"  # Assuming PDE type is in first two parts
                )
            else:
                pde_type = dir_name  # Use directory name as fallback

        # Load history.json
        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            # Try alternative locations
            alt_files = glob.glob(
                os.path.join(exp_dir, "**", "history.json"), recursive=True
            )
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
                except:
                    pass

            if pde_type not in pdes:
                pdes[pde_type] = []

            pdes[pde_type].append(
                {
                    "name": os.path.basename(exp_dir),
                    "train_loss": history.get("train_loss", []),
                    "val_loss": (
                        history.get("val_loss", []) if "val_loss" in history else []
                    ),
                    "computation_time": comp_time,
                    "hover_text": hover_text,
                }
            )
        except:
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


# Callback to update 3D solution visualizations
@app.callback(
    [
        Output("exact-solution-3d", "figure"),
        Output("predicted-solution-3d", "figure"),
    ],
    [
        Input("experiment-selector", "value"),
        Input("time-slider", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_solution_visualizations(experiment, time_point, _):
    if not experiment:
        return create_empty_3d_figure("Select an experiment"), create_empty_3d_figure(
            "Select an experiment"
        )

    try:
        print(f"\nAttempting to visualize solution for experiment: {experiment}")
        print(f"Time point: {time_point}")

        # Import necessary modules
        import torch
        import sys

        sys.path.append(".")
        from src.utils.utils import plot_solution
        from src.neural_networks import PINNModel
        from src.pdes.pde_base import PDEConfig

        # Look for model directly in the experiment directory
        model_path = os.path.join(experiment, "final_model.pt")
        config_path = os.path.join(experiment, "config.yaml")

        print(f"Model path: {model_path}")
        print(f"Config path: {config_path}")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            print("Error: Model or config file not found")
            return create_empty_3d_figure(
                "No model data available"
            ), create_empty_3d_figure("No model data available")

        # Load configuration
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"Loaded config: {json.dumps(config, indent=2)}")

        # Create PDE configuration
        pde_type = config.get("pde_type", "unknown").lower()
        print(f"PDE type: {pde_type}")

        # Extract PDE-specific configuration
        pde_config = config.get("pde_configs", {}).get(pde_type, {})
        if not pde_config:
            print(f"Error: No configuration found for PDE type {pde_type}")
            return create_empty_3d_figure(
                f"No configuration for PDE type: {pde_type}"
            ), create_empty_3d_figure(f"No configuration for PDE type: {pde_type}")

        # Create PDEConfig instance with the correct configuration
        pde_config = PDEConfig(
            name=pde_config.get("name", pde_type),
            dimension=pde_config.get("dimension", 1),
            domain=pde_config.get("domain", [[0, 1]]),
            time_domain=pde_config.get("time_domain", [0, 1]),
            initial_condition=pde_config.get("initial_condition", {}),
            boundary_conditions=pde_config.get("boundary_conditions", {}),
            parameters=pde_config.get("parameters", {}),
            exact_solution=pde_config.get("exact_solution", {}),
        )

        # Import and create appropriate PDE instance
        if "heat" in pde_type:
            from src.pdes.heat_equation import HeatEquation

            pde = HeatEquation(config=pde_config)
            print(
                f"Created HeatEquation instance with dimension {pde_config.dimension}"
            )
        elif "wave" in pde_type:
            from src.pdes.wave_equation import WaveEquation

            pde = WaveEquation(config=pde_config)
            print("Created WaveEquation instance")
        elif "burgers" in pde_type:
            from src.pdes.burgers_equation import BurgersEquation

            pde = BurgersEquation(config=pde_config)
            print("Created BurgersEquation instance")
        elif "convection" in pde_type:
            from src.pdes.convection_equation import ConvectionEquation

            pde = ConvectionEquation(config=pde_config)
            print("Created ConvectionEquation instance")
        elif "kdv" in pde_type:
            from src.pdes.kdv_equation import KdVEquation

            pde = KdVEquation(config=pde_config)
            print("Created KdVEquation instance")
        elif "allen" in pde_type and "cahn" in pde_type:
            from src.pdes.allen_cahn import AllenCahnEquation

            pde = AllenCahnEquation(config=pde_config)
            print("Created AllenCahnEquation instance")
        elif "cahn" in pde_type and "hilliard" in pde_type:
            from src.pdes.cahn_hilliard import CahnHilliardEquation

            pde = CahnHilliardEquation(config=pde_config)
            print("Created CahnHilliardEquation instance")
        elif "black" in pde_type or "scholes" in pde_type:
            from src.pdes.black_scholes import BlackScholesEquation

            pde = BlackScholesEquation(config=pde_config)
            print("Created BlackScholesEquation instance")
        elif "pendulum" in pde_type:
            from src.pdes.pendulum_equation import PendulumEquation

            pde = PendulumEquation(config=pde_config)
            print("Created PendulumEquation instance")
        else:
            print(f"Error: Unsupported PDE type: {pde_type}")
            return create_empty_3d_figure(
                f"Unsupported PDE type: {pde_type}"
            ), create_empty_3d_figure(f"Unsupported PDE type: {pde_type}")

        # Load model
        device = torch.device("cpu")  # Use CPU for visualization
        print(f"Using device: {device}")

        # Create model configuration
        model_config = {
            "architecture": config.get("architectures", {}).get(
                pde_config.get("architecture", "feedforward"), {}
            ),
            "device": device,
            "pde_type": pde_type,
            "input_dim": (
                2 if pde_config.dimension == 1 else 3
            ),  # x,t for 1D or x,y,t for 2D
            "output_dim": 1,
        }

        print(f"Model config: {model_config}")

        try:
            model = PINNModel(config=model_config, device=device)
            print("Successfully created PINNModel instance")
            model.load_state_dict(torch.load(model_path, map_location=device))
            print("Successfully loaded model state")
            model.eval()
            print("Model set to eval mode")
        except Exception as e:
            print(f"Error creating/loading model: {e}")
            return create_empty_3d_figure(
                f"Error loading model: {str(e)}"
            ), create_empty_3d_figure(f"Error loading model: {str(e)}")

        # Generate visualization data using plot_solution
        print("Generating visualization...")
        num_points = 1000  # Increased for better resolution
        try:
            exact_fig, predicted_fig = plot_solution(
                model=model,
                pde=pde,
                num_points=num_points,
                time_point=time_point,  # Pass the current time point
                return_figs=True,  # Return figures instead of showing them
            )
            print("Visualization generated successfully")
            return exact_fig, predicted_fig
        except Exception as e:
            print(f"Error generating visualization: {e}")
            return create_empty_3d_figure(
                f"Error generating visualization: {str(e)}"
            ), create_empty_3d_figure(f"Error generating visualization: {str(e)}")

    except Exception as e:
        print(f"Error in solution visualization: {e}")
        import traceback

        traceback.print_exc()
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


def generate_example_solution(X, Y, solution_type, pde_type, time=None):
    """Generate example solutions for visualization."""
    if time is None:
        time = Y  # Assume Y is time grid for 1D problems

    # Generate different solutions based on PDE type
    if pde_type and "heat" in pde_type.lower():
        if solution_type == "exact":
            # Heat equation exact solution
            return np.sin(np.pi * X) * np.exp(-np.pi**2 * time)
        else:
            # Add some noise to simulate prediction
            return (
                np.sin(np.pi * X)
                * np.exp(-np.pi**2 * time)
                * (1 + 0.1 * np.random.rand(*X.shape) - 0.05)
            )

    elif pde_type and "wave" in pde_type.lower():
        if solution_type == "exact":
            # Wave equation exact solution
            return np.sin(np.pi * X) * np.cos(np.pi * time)
        else:
            # Add some noise to simulate prediction
            return (
                np.sin(np.pi * X)
                * np.cos(np.pi * time)
                * (1 + 0.1 * np.random.rand(*X.shape) - 0.05)
            )

    elif pde_type and "burgers" in pde_type.lower():
        # Simplified Burgers' equation solution (example)
        if solution_type == "exact":
            return np.tanh((X - 0.5 - 0.5 * time) / (0.1 + 0.05 * time))
        else:
            return np.tanh((X - 0.5 - 0.5 * time) / (0.1 + 0.05 * time)) * (
                1 + 0.15 * np.random.rand(*X.shape) - 0.075
            )

    else:
        # Generic solution for unknown PDEs
        if solution_type == "exact":
            return np.sin(np.pi * X) * np.cos(2 * np.pi * time)
        else:
            return (
                np.sin(np.pi * X)
                * np.cos(2 * np.pi * time)
                * (1 + 0.1 * np.random.rand(*X.shape) - 0.05)
            )


def get_experiment_details(experiment_path):
    """Get experiment details from config file."""
    try:
        config_path = os.path.join(experiment_path, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Extract basic details
        pde_type = config.get("pde", {}).get("name", "Unknown")
        architecture = config.get("model", {}).get("architecture", "Unknown")
        hidden_dim = config.get("model", {}).get("hidden_dim", "Unknown")
        num_layers = config.get("model", {}).get("num_layers", "Unknown")
        learning_rate = (
            config.get("training", {})
            .get("optimizer_config", {})
            .get("learning_rate", "Unknown")
        )
        batch_size = config.get("training", {}).get("batch_size", "Unknown")
        num_epochs = config.get("training", {}).get("num_epochs", "Unknown")
        rl_enabled = config.get("rl", {}).get("enabled", False)

        # Add RL status to metadata for HTML report
        metadata = {
            "pde_type": pde_type,
            "architecture": architecture,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "rl_enabled": rl_enabled,
        }

        details = f"""
        **PDE Type:** {pde_type}
        **Architecture:** {architecture}
        **Hidden Dimension:** {hidden_dim}
        **Number of Layers:** {num_layers}
        **Learning Rate:** {learning_rate}
        **Batch Size:** {batch_size}
        **Number of Epochs:** {num_epochs}
        **RL Enabled:** {rl_enabled}
        """

        return details, metadata
    except Exception as e:
        return f"Error loading experiment details: {str(e)}", {}


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
    except:
        return name


def generate_report(experiment_path):
    """Generate a detailed report for the experiment."""
    try:
        config_path = os.path.join(experiment_path, "config.yaml")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Get experiment name
        exp_name = get_experiment_name(experiment_path)

        # Basic configuration
        pde_type = config.get("pde", {}).get("name", "Unknown")
        architecture = config.get("model", {}).get("architecture", "Unknown")
        hidden_dim = config.get("model", {}).get("hidden_dim", "Unknown")
        num_layers = config.get("model", {}).get("num_layers", "Unknown")
        learning_rate = (
            config.get("training", {})
            .get("optimizer_config", {})
            .get("learning_rate", "Unknown")
        )
        batch_size = config.get("training", {}).get("batch_size", "Unknown")
        num_epochs = config.get("training", {}).get("num_epochs", "Unknown")
        rl_enabled = config.get("rl", {}).get("enabled", False)

        report = f"""# Experiment Report: {exp_name}

## Configuration
- **PDE Type:** {pde_type}
- **Architecture:** {architecture}
- **Hidden Dimension:** {hidden_dim}
- **Number of Layers:** {num_layers}
- **Learning Rate:** {learning_rate}
- **Batch Size:** {batch_size}
- **Number of Epochs:** {num_epochs}
- **RL Enabled:** {rl_enabled}

## Training Results
"""
        # Add training results if available
        history_path = os.path.join(experiment_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)

            final_epoch = len(history.get("train_loss", []))
            best_val_loss = min(history.get("val_loss", [float("inf")]))

            report += f"""
- **Completed Epochs:** {final_epoch}
- **Best Validation Loss:** {best_val_loss:.6f}
"""

        return report
    except Exception as e:
        return f"Error generating report: {str(e)}"


@app.callback(
    Output("download-report", "data"),
    Input("download-report-button", "n_clicks"),
    State("experiment-selector", "value"),
    prevent_initial_call=True,
)
def generate_report(n_clicks, experiment):
    if not experiment or not n_clicks:
        return None

    try:
        # Get current figures and data
        loss_fig = update_graphs(experiment, None)[0]
        collocation_fig = update_graphs(experiment, None)[1]
        exact_solution, predicted_solution = update_solution_visualizations(
            experiment, 0.5, None
        )

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
            if exp_dir.startswith('.'):
                continue
                
            exp_path = os.path.join(experiments_dir, exp_dir)
            if os.path.isdir(exp_path):
                # Check if this is a running experiment
                is_running = os.path.exists(os.path.join(exp_path, ".running"))
                
                # Get experiment details from directory name
                parts = exp_dir.split('_')
                if len(parts) >= 2:
                    # Try to extract meaningful information from directory name
                    timestamp = parts[0]
                    if len(parts) >= 4:
                        pde_name = parts[1] + " " + parts[2]
                        arch_name = parts[3]
                    else:
                        pde_name = "Unknown PDE"
                        arch_name = parts[1] if len(parts) > 1 else "unknown"
                    
                    # Format the display name
                    display_name = f"{timestamp} - {pde_name} ({arch_name})"
                else:
                    display_name = exp_dir
                
                if is_running:
                    display_name += " (Running)"
                
                experiments.append({"label": display_name, "value": exp_path})

    # Sort experiments by name (which includes timestamp) in reverse order
    experiments.sort(key=lambda x: x["label"], reverse=True)
    return experiments


def update_plots(experiment_path):
    """Update all plots for the selected experiment."""
    plots = {}

    # Check if experiment is running
    is_running = os.path.exists(os.path.join(experiment_path, ".running"))

    try:
        # Load history
        history_path = os.path.join(experiment_path, "history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)

            # Training progress plot
            plots["training_progress"] = create_training_plot(history)

            # Load latest collocation plot if available
            latest_coll_plot = os.path.join(
                experiment_path, "visualizations", "latest_collocation_evolution.png"
            )
            if os.path.exists(latest_coll_plot):
                plots["collocation_points"] = latest_coll_plot

            # Load latest solution plots if available
            latest_solution = os.path.join(
                experiment_path, "visualizations", "latest_solution.png"
            )
            if os.path.exists(latest_solution):
                plots["solution"] = latest_solution

        # If experiment is running, set auto-refresh
        if is_running:
            plots["refresh"] = True
    except Exception as e:
        print(f"Error updating plots: {e}")

    return plots


@app.callback(
    [
        Output("experiment-details", "children"),
        Output("training-plot", "figure"),
        Output("collocation-plot", "src"),
        Output("solution-plot", "src"),
        Output("interval-component", "disabled"),
    ],
    [Input("experiment-dropdown", "value"), Input("interval-component", "n_intervals")],
)
def update_experiment_view(selected_experiment, n_intervals):
    """Update all components of the dashboard based on selected experiment."""
    if not selected_experiment:
        return "No experiment selected", {}, "", "", True

    # Get experiment details
    details, metadata = get_experiment_details(selected_experiment)

    # Update plots
    plots = update_plots(selected_experiment)

    # Check if experiment is running
    is_running = os.path.exists(os.path.join(selected_experiment, ".running"))

    return (
        details,
        plots.get("training_progress", {}),
        plots.get("collocation_points", ""),
        plots.get("solution", ""),
        not is_running,  # Enable auto-refresh only for running experiments
    )


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
            app.run(debug=True, port=port)
            break
        except Exception as e:
            if "Address already in use" in str(e):
                print(f"Port {port} is in use. Trying port {port+1}...")
                port += 1
                if attempt == max_retries - 1:
                    print(
                        f"Could not find available port after {max_retries} attempts."
                    )
                    print(f"Please close any running dashboards and try again.")
            else:
                print(f"Error starting dashboard: {e}")
                break
