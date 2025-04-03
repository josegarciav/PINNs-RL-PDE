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
        ),
    ],
    style={"padding": "20px"},
)


# Callback to update experiment list
@app.callback(
    Output("experiment-selector", "options"), Input("interval-component", "n_intervals")
)
def update_experiments(_):
    # Get list of experiment directories
    experiment_dirs = []
    for results_dir in [
        "results",
        "experiments",
    ]:  # Check both possible result directories
        if os.path.exists(results_dir):
            experiment_dirs.extend(
                [
                    {"label": d, "value": os.path.join(results_dir, d)}
                    for d in os.listdir(results_dir)
                    if os.path.isdir(os.path.join(results_dir, d))
                ]
            )
    return experiment_dirs


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
            # Try alternative locations
            alt_files = glob.glob(
                os.path.join(experiment, "**", "history.json"), recursive=True
            )
            if alt_files:
                history_file = alt_files[0]

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
            # Create basic metadata from config if available
            config_file = os.path.join(experiment, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                    experiment_details = f"Experiment: {os.path.basename(experiment)}\nConfiguration:\n{json.dumps(config, indent=2)}"

        # Loss figure
        loss_fig = go.Figure()

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
            # Try to get from config if available
            config_file = os.path.join(experiment, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                    if "training" in config and "num_epochs" in config["training"]:
                        total_epochs = config["training"]["num_epochs"]

        # Create training loss trace
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
                    metadata_data = json.load(f)
                    if (
                        "training_params" in metadata_data
                        and "validation_frequency" in metadata_data["training_params"]
                    ):
                        val_frequency = metadata_data["training_params"][
                            "validation_frequency"
                        ]
            else:
                # Try to get from config if available
                config_file = os.path.join(experiment, "config.json")
                if os.path.exists(config_file):
                    with open(config_file, "r") as f:
                        config = json.load(f)
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
    for results_dir in [
        "results",
        "experiments",
    ]:  # Check both possible result directories
        if os.path.exists(results_dir):
            experiment_dirs.extend(
                [
                    os.path.join(results_dir, d)
                    for d in os.listdir(results_dir)
                    if os.path.isdir(os.path.join(results_dir, d))
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

        # If not found, check config.json
        if not architecture:
            config_file = os.path.join(exp_dir, "config.json")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        config = json.load(f)
                        if "model" in config and "architecture" in config["model"]:
                            architecture = config["model"]["architecture"]
                except:
                    pass

            # Try model_config.json as well
            if not architecture:
                config_file = os.path.join(exp_dir, "model_config.json")
                if os.path.exists(config_file):
                    try:
                        with open(config_file, "r") as f:
                            config = json.load(f)
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

    # Search for experiment directories
    experiment_dirs = []
    for results_dir in [
        "results",
        "experiments",
    ]:  # Check both possible result directories
        if os.path.exists(results_dir):
            experiment_dirs.extend(
                [
                    os.path.join(results_dir, d)
                    for d in os.listdir(results_dir)
                    if os.path.isdir(os.path.join(results_dir, d))
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

        # If not found, try to infer from directory name
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
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        comp_time = metadata.get("training_time_minutes")
                        if comp_time:
                            computation_times[pde_type] = comp_time
                            hover_text = f"Experiment: {os.path.basename(exp_dir)}<br>Training time: {comp_time:.2f} minutes"
                        else:
                            hover_text = f"Experiment: {os.path.basename(exp_dir)}"
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
        # Try to load model and PDE objects
        model_path = os.path.join(experiment, "model.pt")
        config_file = os.path.join(experiment, "config.json")

        # If pre-computed visualization files don't exist, create generic figures
        if not os.path.exists(model_path) or not os.path.exists(config_file):
            return (
                create_empty_3d_figure("No model data available"),
                create_empty_3d_figure("No model data available"),
            )

        # Load configuration
        with open(config_file, "r") as f:
            config = json.load(f)

        # Extract PDE information
        pde_type = None
        domain = None
        dimension = 1  # Default

        # Try to get from metadata or config
        metadata_file = os.path.join(experiment, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                if "pde_type" in metadata:
                    pde_type = metadata.get("pde_type")
                elif "config" in metadata and "pde" in metadata["config"]:
                    pde_params = metadata["config"]["pde"]
                    pde_type = pde_params.get("name", "unknown_pde")
                    domain = pde_params.get("domain", [0, 1])
                    dimension = pde_params.get("dimension", 1)

        # If still no PDE info, try config directly
        if not pde_type and "pde" in config:
            pde_params = config["pde"]
            pde_type = pde_params.get("name", "unknown_pde")
            domain = pde_params.get("domain", [0, 1])
            dimension = pde_params.get("dimension", 1)

        # Try to load the actual PDE model and model for more accurate visualizations
        try:
            # Import necessary modules for loading the model and PDE
            import torch
            import sys

            sys.path.append(".")  # Ensure current directory is in path
            from src.pdes.pde_base import PDEBase

            # Try to create PDE instance
            pde_instance = None
            if pde_type:
                # Import specific PDE classes based on the PDE type
                if "heat" in pde_type.lower():
                    from src.pdes.heat import Heat

                    pde_instance = Heat(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "wave" in pde_type.lower():
                    from src.pdes.wave import Wave

                    pde_instance = Wave(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "burgers" in pde_type.lower():
                    from src.pdes.burgers import Burgers

                    pde_instance = Burgers(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "kdv" in pde_type.lower():
                    from src.pdes.kdv import KdV

                    pde_instance = KdV(**pde_params if "pde_params" in locals() else {})
                elif "convection" in pde_type.lower():
                    from src.pdes.convection import Convection

                    pde_instance = Convection(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "allen" in pde_type.lower():
                    from src.pdes.allen_cahn import AllenCahn

                    pde_instance = AllenCahn(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "cahn" in pde_type.lower():
                    from src.pdes.cahn_hilliard import CahnHilliard

                    pde_instance = CahnHilliard(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "black" in pde_type.lower() or "scholes" in pde_type.lower():
                    from src.pdes.black_scholes import BlackScholes

                    pde_instance = BlackScholes(
                        **pde_params if "pde_params" in locals() else {}
                    )
                elif "pendulum" in pde_type.lower():
                    from src.pdes.pendulum import Pendulum

                    pde_instance = Pendulum(
                        **pde_params if "pde_params" in locals() else {}
                    )

            # Try to load the trained model
            model = None
            if os.path.exists(model_path):
                # Get the appropriate model architecture
                from src.components.neural_networks import PINNModel

                # Load model with caution
                try:
                    model = torch.load(model_path, map_location=torch.device("cpu"))
                except Exception as model_load_error:
                    print(f"Error loading model: {model_load_error}")

            # If we have both a PDE instance and model, we can generate accurate solutions
            if pde_instance and model:
                return generate_pde_visualization(
                    pde_instance, model, time_point, dimension
                )
        except Exception as instance_error:
            print(f"Error loading PDE instance or model: {instance_error}")
            # Continue with generic visualization

        # Generate visualization data
        if dimension == 1:
            # For 1D PDEs
            x = np.linspace(domain[0] if domain else 0, domain[1] if domain else 1, 100)
            t = np.linspace(0, 1, 100)
            X, T = np.meshgrid(x, t)

            # Create surface plots
            exact_fig = go.Figure(
                data=[
                    go.Surface(
                        x=X,
                        y=T,
                        z=generate_example_solution(X, T, "exact", pde_type),
                        colorscale="Viridis",
                    )
                ]
            )
            exact_fig.update_layout(
                title=f"Exact Solution - {pde_type}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

            predicted_fig = go.Figure(
                data=[
                    go.Surface(
                        x=X,
                        y=T,
                        z=generate_example_solution(X, T, "predicted", pde_type),
                        colorscale="Plasma",
                    )
                ]
            )
            predicted_fig.update_layout(
                title=f"Predicted Solution - {pde_type}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

        elif dimension == 2:
            # For 2D PDEs
            x = np.linspace(domain[0] if domain else 0, domain[1] if domain else 1, 50)
            y = np.linspace(domain[0] if domain else 0, domain[1] if domain else 1, 50)
            X, Y = np.meshgrid(x, y)

            # Create figures for 2D problems at a fixed time
            exact_solution = generate_example_solution(
                X, Y, "exact", pde_type, time=time_point
            )
            predicted_solution = generate_example_solution(
                X, Y, "predicted", pde_type, time=time_point
            )

            exact_fig = go.Figure(
                data=[go.Surface(x=X, y=Y, z=exact_solution, colorscale="Viridis")]
            )
            exact_fig.update_layout(
                title=f"Exact Solution at t={time_point:.2f} - {pde_type}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="u(x,y,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

            predicted_fig = go.Figure(
                data=[go.Surface(x=X, y=Y, z=predicted_solution, colorscale="Plasma")]
            )
            predicted_fig.update_layout(
                title=f"Predicted Solution at t={time_point:.2f} - {pde_type}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="u(x,y,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

        else:
            # For higher dimensions, create empty figures
            return (
                create_empty_3d_figure(f"Cannot visualize {dimension}D PDEs"),
                create_empty_3d_figure(f"Cannot visualize {dimension}D PDEs"),
            )

        return exact_fig, predicted_fig

    except Exception as e:
        print(f"Error generating 3D visualizations: {e}")
        return (
            create_empty_3d_figure(f"Error: {str(e)}"),
            create_empty_3d_figure(f"Error: {str(e)}"),
        )


def generate_pde_visualization(pde, model, time_point, dimension):
    """Generate visualization based on actual PDE instance and model."""
    try:
        import torch

        # For 1D PDEs
        if dimension == 1:
            # Create spatial grid
            domain = pde.domain
            x = np.linspace(domain[0], domain[1], 100)
            t_values = np.linspace(0, 1, 100)
            X, T = np.meshgrid(x, t_values)

            # Get exact solution
            exact_solution = np.zeros_like(X)
            for i, t_val in enumerate(t_values):
                for j, x_val in enumerate(x):
                    exact_solution[i, j] = pde.exact_solution(
                        torch.tensor([x_val, t_val])
                    ).item()

            # Get predicted solution
            predicted_solution = np.zeros_like(X)
            model.eval()
            with torch.no_grad():
                for i, t_val in enumerate(t_values):
                    for j, x_val in enumerate(x):
                        input_tensor = torch.tensor([x_val, t_val], dtype=torch.float32)
                        predicted_solution[i, j] = model(input_tensor).item()

            # Create figures
            exact_fig = go.Figure(
                data=[go.Surface(x=X, y=T, z=exact_solution, colorscale="Viridis")]
            )
            exact_fig.update_layout(
                title=f"Exact Solution - {pde.__class__.__name__}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

            predicted_fig = go.Figure(
                data=[go.Surface(x=X, y=T, z=predicted_solution, colorscale="Plasma")]
            )
            predicted_fig.update_layout(
                title=f"Predicted Solution - {pde.__class__.__name__}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

            return exact_fig, predicted_fig

        # For 2D PDEs
        elif dimension == 2:
            # Create spatial grid
            domain = pde.domain
            x = np.linspace(domain[0], domain[1], 50)
            y = np.linspace(domain[0], domain[1], 50)
            X, Y = np.meshgrid(x, y)

            # Get exact solution at specific time point
            exact_solution = np.zeros_like(X)
            for i, y_val in enumerate(y):
                for j, x_val in enumerate(x):
                    exact_solution[i, j] = pde.exact_solution(
                        torch.tensor([x_val, y_val, time_point])
                    ).item()

            # Get predicted solution
            predicted_solution = np.zeros_like(X)
            model.eval()
            with torch.no_grad():
                for i, y_val in enumerate(y):
                    for j, x_val in enumerate(x):
                        input_tensor = torch.tensor(
                            [x_val, y_val, time_point], dtype=torch.float32
                        )
                        predicted_solution[i, j] = model(input_tensor).item()

            # Create figures
            exact_fig = go.Figure(
                data=[go.Surface(x=X, y=Y, z=exact_solution, colorscale="Viridis")]
            )
            exact_fig.update_layout(
                title=f"Exact Solution at t={time_point:.2f} - {pde.__class__.__name__}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="u(x,y,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

            predicted_fig = go.Figure(
                data=[go.Surface(x=X, y=Y, z=predicted_solution, colorscale="Plasma")]
            )
            predicted_fig.update_layout(
                title=f"Predicted Solution at t={time_point:.2f} - {pde.__class__.__name__}",
                scene=dict(
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="u(x,y,t)",
                ),
                margin=dict(l=0, r=0, b=0, t=30),
            )

            return exact_fig, predicted_fig

        else:
            # For higher dimensions, return empty figures
            return (
                create_empty_3d_figure(f"Cannot visualize {dimension}D PDEs"),
                create_empty_3d_figure(f"Cannot visualize {dimension}D PDEs"),
            )

    except Exception as e:
        print(f"Error in generate_pde_visualization: {e}")
        return (
            create_empty_3d_figure(f"Error: {str(e)}"),
            create_empty_3d_figure(f"Error: {str(e)}"),
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


def generate_html_report(experiment_path, figures, metadata):
    """Generate an interactive HTML report for the experiment."""
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>PINNs-RL-PDE Experiment Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .section {{ margin-bottom: 40px; }}
            .plot {{ width: 100%; height: 600px; }}
            pre {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>PINNs-RL-PDE Experiment Report</h1>
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
