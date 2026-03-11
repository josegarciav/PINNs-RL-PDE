"""Dash web dashboard for monitoring and comparing PINN training runs."""

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
    """Parse command-line arguments for the dashboard server."""
    parser = argparse.ArgumentParser(description="PINNs-RL-PDE Training Monitor Dashboard")
    parser.add_argument("--port", type=int, default=8050, help="Port to run the dashboard on")
    return parser.parse_args()


# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Shared styles
BUTTON_STYLE = {
    "backgroundColor": "#2196F3",
    "color": "white",
    "padding": "10px 20px",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "fontSize": "14px",
}
PRE_STYLE = {
    "whiteSpace": "pre-wrap",
    "wordBreak": "break-all",
    "backgroundColor": "#f5f5f5",
    "padding": "10px",
    "border": "1px solid #ddd",
    "borderRadius": "5px",
    "maxHeight": "400px",
    "overflow": "auto",
}
LABEL_STYLE = {"fontWeight": "bold", "marginBottom": "4px", "display": "block"}
FIELD_STYLE = {"marginBottom": "12px"}

# Load config.yaml defaults at startup for the training form
_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
if not os.path.exists(_CONFIG_PATH):
    _CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
try:
    with open(_CONFIG_PATH, "r") as _f:
        _YAML_CONFIG = yaml.safe_load(_f)
except Exception:
    _YAML_CONFIG = {}

_TRAINING_DEFAULTS = _YAML_CONFIG.get("training", {})
_PDE_CONFIGS = _YAML_CONFIG.get("pde_configs", {})
_ARCHITECTURES = list(_YAML_CONFIG.get("architectures", {}).keys())

# Map PDE display names to config keys
_PDE_OPTIONS = []
for _key, _cfg in _PDE_CONFIGS.items():
    if _key == "heat_2d":
        continue  # skip 2D variant for now
    _PDE_OPTIONS.append({"label": _cfg.get("name", _key), "value": _cfg.get("name", _key)})

# ============================================================
# Layout
# ============================================================
app.layout = html.Div(
    [
        html.Div(
            html.H1(
                "PINN-RL Training Monitor",
                style={
                    "color": "white",
                    "margin": "0",
                    "padding": "24px 0",
                    "fontWeight": "600",
                    "letterSpacing": "0.5px",
                },
            ),
            style={
                "background": "linear-gradient(135deg, #0d1b2a 0%, #1b2d4f 50%, #1a3a5c 100%)",
                "textAlign": "center",
                "marginBottom": "20px",
                "borderRadius": "0 0 8px 8px",
                "boxShadow": "0 2px 8px rgba(0,0,0,0.15)",
            },
        ),
        # Tabs
        dcc.Tabs(
            id="main-tabs",
            value="live-training",
            children=[
                # ==================== TAB 1: LIVE TRAINING ====================
                dcc.Tab(
                    label="Live Training",
                    value="live-training",
                    children=[
                        # Sub-tabs: Monitor / New Training
                        dcc.Tabs(
                            id="live-subtabs",
                            value="new-training",
                            children=[
                                # ---------- Sub-tab 1a: New Training ----------
                                dcc.Tab(
                                    label="New Training",
                                    value="new-training",
                                    children=[
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Launch New Training",
                                                    style={
                                                        "textAlign": "center",
                                                        "marginBottom": "20px",
                                                    },
                                                ),
                                                # Two-column layout
                                                html.Div(
                                                    [
                                                        # Left column: PDE & Architecture
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Model",
                                                                    style={
                                                                        "borderBottom": "2px solid #2196F3",
                                                                        "paddingBottom": "8px",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "PDE:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-pde-selector",
                                                                            options=_PDE_OPTIONS,
                                                                            value="Heat Equation",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Architecture:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-arch-selector",
                                                                            options=[
                                                                                {
                                                                                    "label": a.capitalize(),
                                                                                    "value": a,
                                                                                }
                                                                                for a in _ARCHITECTURES
                                                                            ],
                                                                            value=_PDE_CONFIGS.get(
                                                                                "heat", {}
                                                                            ).get(
                                                                                "architecture",
                                                                                "fourier",
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Device:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-device-selector",
                                                                            options=[
                                                                                {
                                                                                    "label": "CPU",
                                                                                    "value": "cpu",
                                                                                },
                                                                                {
                                                                                    "label": "MPS (Apple GPU)",
                                                                                    "value": "mps",
                                                                                },
                                                                                {
                                                                                    "label": "CUDA (NVIDIA GPU)",
                                                                                    "value": "cuda",
                                                                                },
                                                                            ],
                                                                            value=_YAML_CONFIG.get(
                                                                                "device", "cpu"
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "RL Adaptive Sampling:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Checklist(
                                                                            id="train-rl-toggle",
                                                                            options=[
                                                                                {
                                                                                    "label": " Enable RL",
                                                                                    "value": "rl",
                                                                                }
                                                                            ],
                                                                            value=[],
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                            ],
                                                            style={
                                                                "width": "48%",
                                                                "display": "inline-block",
                                                                "verticalAlign": "top",
                                                                "paddingRight": "2%",
                                                            },
                                                        ),
                                                        # Right column: Hyperparameters
                                                        html.Div(
                                                            [
                                                                html.H4(
                                                                    "Hyperparameters",
                                                                    style={
                                                                        "borderBottom": "2px solid #4CAF50",
                                                                        "paddingBottom": "8px",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Epochs:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-epochs-input",
                                                                            type="number",
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "num_epochs", 500
                                                                            ),
                                                                            min=10,
                                                                            step=10,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Learning Rate:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-lr-input",
                                                                            type="number",
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "optimizer_config",
                                                                                {},
                                                                            ).get(
                                                                                "learning_rate",
                                                                                0.005,
                                                                            ),
                                                                            min=1e-7,
                                                                            max=1.0,
                                                                            step=1e-4,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Batch Size:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-batch-size",
                                                                            options=[
                                                                                {
                                                                                    "label": str(v),
                                                                                    "value": v,
                                                                                }
                                                                                for v in [
                                                                                    256,
                                                                                    512,
                                                                                    1024,
                                                                                    2048,
                                                                                    4096,
                                                                                ]
                                                                            ],
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "batch_size", 2048
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Collocation Points:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-collocation-points",
                                                                            type="number",
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "num_collocation_points",
                                                                                5000,
                                                                            ),
                                                                            min=100,
                                                                            step=500,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Boundary Points:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-boundary-points",
                                                                            type="number",
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "num_boundary_points",
                                                                                5000,
                                                                            ),
                                                                            min=100,
                                                                            step=500,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Initial Points:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-initial-points",
                                                                            type="number",
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "num_initial_points",
                                                                                5000,
                                                                            ),
                                                                            min=100,
                                                                            step=500,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                            ],
                                                            style={
                                                                "width": "48%",
                                                                "display": "inline-block",
                                                                "verticalAlign": "top",
                                                                "paddingLeft": "2%",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                                # PDE-specific info box
                                                html.Div(
                                                    id="pde-info-box",
                                                    style={
                                                        "marginTop": "16px",
                                                        "padding": "12px",
                                                        "backgroundColor": "#f0f7ff",
                                                        "borderRadius": "6px",
                                                        "border": "1px solid #c0d8f0",
                                                    },
                                                ),
                                                # Launch button
                                                html.Div(
                                                    [
                                                        html.Button(
                                                            "Start Training",
                                                            id="start-training-button",
                                                            style={
                                                                **BUTTON_STYLE,
                                                                "backgroundColor": "#4CAF50",
                                                                "padding": "14px 40px",
                                                                "fontSize": "16px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="trainer-status",
                                                            style={
                                                                "marginTop": "12px",
                                                                "fontSize": "14px",
                                                            },
                                                        ),
                                                    ],
                                                    style={
                                                        "textAlign": "center",
                                                        "marginTop": "24px",
                                                    },
                                                ),
                                            ],
                                            style={
                                                "maxWidth": "800px",
                                                "margin": "20px auto",
                                                "padding": "24px",
                                                "backgroundColor": "#fafafa",
                                                "borderRadius": "8px",
                                                "border": "1px solid #e0e0e0",
                                            },
                                        ),
                                    ],
                                ),
                                # ---------- Sub-tab 1b: Monitor ----------
                                dcc.Tab(
                                    label="Monitor",
                                    value="monitor",
                                    children=[
                                        html.Div(
                                            [
                                                html.Div(
                                                    [
                                                        html.Label("Select Experiment:"),
                                                        dcc.Dropdown(
                                                            id="experiment-selector",
                                                            placeholder="Select running or recent experiment...",
                                                        ),
                                                    ],
                                                    style={
                                                        "width": "60%",
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
                                                                **BUTTON_STYLE,
                                                                "backgroundColor": "#4CAF50",
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
                                            style={"width": "60%", "margin": "20px auto"},
                                        ),
                                        # Epoch progress bar
                                        html.Div(
                                            [
                                                html.Div(
                                                    id="epoch-progress-text",
                                                    style={
                                                        "marginBottom": "4px",
                                                        "fontWeight": "bold",
                                                    },
                                                ),
                                                html.Div(
                                                    html.Div(
                                                        id="epoch-progress-bar-inner",
                                                        style={
                                                            "width": "0%",
                                                            "height": "20px",
                                                            "backgroundColor": "#2196F3",
                                                            "borderRadius": "4px",
                                                            "transition": "width 0.5s",
                                                        },
                                                    ),
                                                    style={
                                                        "width": "100%",
                                                        "backgroundColor": "#e0e0e0",
                                                        "borderRadius": "4px",
                                                    },
                                                ),
                                            ],
                                            style={"width": "60%", "margin": "10px auto"},
                                        ),
                                        # Loss view toggle
                                        html.Div(
                                            [
                                                dcc.RadioItems(
                                                    id="loss-view-toggle",
                                                    options=[
                                                        {
                                                            "label": "Total Loss",
                                                            "value": "total",
                                                        },
                                                        {
                                                            "label": "Loss Breakdown",
                                                            "value": "breakdown",
                                                        },
                                                    ],
                                                    value="total",
                                                    inline=True,
                                                    style={"textAlign": "center"},
                                                ),
                                            ],
                                            style={
                                                "width": "60%",
                                                "margin": "10px auto",
                                            },
                                        ),
                                        # Loss graph
                                        html.Div(
                                            [dcc.Graph(id="loss-graph")],
                                            style={"width": "100%"},
                                        ),
                                        # Experiment details
                                        html.Div(
                                            [
                                                html.H3("Experiment Details"),
                                                html.Pre(
                                                    id="experiment-details",
                                                    style=PRE_STYLE,
                                                ),
                                            ],
                                            style={"marginTop": "20px"},
                                        ),
                                        # Auto-refresh interval
                                        dcc.Interval(
                                            id="interval-component",
                                            interval=10 * 1000,
                                            n_intervals=0,
                                        ),
                                    ],
                                ),
                            ],
                            style={"marginTop": "10px"},
                        ),
                    ],
                ),
                # ==================== TAB 2: COMPARISON ====================
                dcc.Tab(
                    label="Comparison",
                    value="comparison",
                    children=[
                        html.Div(
                            [
                                html.Button(
                                    "Refresh Comparisons",
                                    id="refresh-comparisons-button",
                                    style={**BUTTON_STYLE, "marginTop": "20px"},
                                ),
                            ],
                            style={"textAlign": "center"},
                        ),
                        html.H2(
                            "Architecture Comparison",
                            style={"textAlign": "center", "marginTop": "20px"},
                        ),
                        dcc.Graph(id="architecture-comparison"),
                        html.H2(
                            "PDE Comparison",
                            style={"textAlign": "center", "marginTop": "30px"},
                        ),
                        dcc.Graph(id="pde-comparison"),
                    ],
                ),
                # ==================== TAB 3: COLLOCATION ====================
                dcc.Tab(
                    label="Collocation & Solution",
                    value="collocation",
                    children=[
                        html.Div(
                            [
                                html.Label("Select Experiment:"),
                                dcc.Dropdown(
                                    id="collocation-experiment-selector",
                                    placeholder="Select experiment...",
                                ),
                            ],
                            style={"width": "50%", "margin": "20px auto"},
                        ),
                        # Collocation evolution
                        html.Div(
                            [dcc.Graph(id="collocation-evolution")],
                            style={"width": "100%"},
                        ),
                        # Solution visualization
                        html.Div(
                            [
                                html.H2("Solution Visualization", style={"textAlign": "center"}),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="exact-solution-3d", style={"height": "50vh"}
                                                )
                                            ],
                                            style={"width": "50%", "display": "inline-block"},
                                        ),
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="predicted-solution-3d",
                                                    style={"height": "50vh"},
                                                )
                                            ],
                                            style={"width": "50%", "display": "inline-block"},
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
                                                    marks={
                                                        i / 10: f"{i/10:.1f}" for i in range(11)
                                                    },
                                                ),
                                            ],
                                            style={"width": "100%", "padding": "20px"},
                                        ),
                                    ],
                                ),
                            ],
                            style={"marginTop": "20px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
    style={"padding": "20px"},
)


# ============================================================
# Helper functions
# ============================================================


def get_experiments():
    """Get list of all experiments with their details."""
    experiments = []
    experiments_dir = "experiments"

    if os.path.exists(experiments_dir):
        for exp_dir in os.listdir(experiments_dir):
            if exp_dir.startswith("."):
                continue

            exp_path = os.path.join(experiments_dir, exp_dir)
            if os.path.isdir(exp_path):
                # Check if running
                running_file = os.path.join(exp_path, ".running")
                is_running = os.path.exists(running_file)
                if is_running:
                    stale = False
                    meta_file = os.path.join(exp_path, "metadata.json")
                    history_file = os.path.join(exp_path, "history.json")

                    if os.path.exists(meta_file):
                        try:
                            with open(meta_file, "r") as mf:
                                meta = json.load(mf)
                                if meta.get("status") == "completed" or "end_time" in meta:
                                    stale = True
                        except Exception:
                            stale = True
                    elif not os.path.exists(history_file):
                        stale = True
                    else:
                        running_age = datetime.now().timestamp() - os.path.getmtime(running_file)
                        if running_age > 3600:
                            stale = True

                    if stale:
                        try:
                            os.remove(running_file)
                        except OSError:
                            pass
                        is_running = False

                # Parse directory name
                parts = exp_dir.split("_")
                if len(parts) >= 3 and len(parts[0]) == 8 and len(parts[1]) == 6:
                    timestamp = f"{parts[0]}_{parts[1]}"
                    remainder = exp_dir[len(timestamp) + 1 :]

                    if remainder.endswith("_no_rl"):
                        rl_status = "no_rl"
                        middle = remainder[: -len("_no_rl")]
                    elif remainder.endswith("_rl"):
                        rl_status = "rl"
                        middle = remainder[: -len("_rl")]
                    else:
                        rl_status = None
                        middle = remainder

                    if "_" in middle:
                        last_underscore = middle.rfind("_")
                        pde_name = middle[:last_underscore]
                        arch_name = middle[last_underscore + 1 :]
                    else:
                        pde_name = middle if middle else "Unknown PDE"
                        arch_name = "unknown"

                    display_name = f"{timestamp} - {pde_name} ({arch_name})"
                    if rl_status:
                        display_name += f" [{rl_status}]"
                else:
                    display_name = exp_dir

                if is_running:
                    display_name += " [RUNNING]"

                experiments.append(
                    {
                        "label": display_name,
                        "value": exp_path,
                        "is_running": is_running,
                    }
                )

    experiments.sort(key=lambda x: x["label"], reverse=True)
    return experiments


def get_live_experiments():
    """Return only running or recently completed experiments."""
    all_exps = get_experiments()
    live = []
    for exp in all_exps:
        exp_path = exp["value"]
        if exp["is_running"]:
            live.append(exp)
            continue
        # Check if completed recently (within 2 hours)
        meta_file = os.path.join(exp_path, "metadata.json")
        if os.path.exists(meta_file):
            try:
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                end_time_str = meta.get("end_time") or meta.get("last_updated")
                if end_time_str:
                    end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
                    if (datetime.now() - end_time).total_seconds() < 7200:
                        live.append(exp)
            except Exception:
                pass
    return live


def create_empty_figure(message, title=""):
    """Create an empty figure with a message."""
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
        title=title,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def create_empty_3d_figure(message):
    """Create an empty 3D figure with a message."""
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=[0],
                y=[0],
                z=[0],
                mode="markers",
                marker=dict(size=0.1, opacity=0),
                showlegend=False,
            )
        ],
    )
    fig.update_layout(
        title="Solution Visualization",
        scene=dict(
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
            zaxis=dict(showticklabels=False, showgrid=False, zeroline=False, title=""),
            annotations=[
                dict(
                    text=message,
                    x=0,
                    y=0,
                    z=0,
                    showarrow=False,
                    font=dict(size=14, color="#666"),
                )
            ],
        ),
        margin=dict(l=0, r=0, b=0, t=30),
    )
    return fig


def get_experiment_name(experiment_path):
    """Extract a readable name from the experiment path."""
    try:
        name = os.path.basename(experiment_path)
        parts = name.split("_")
        if len(parts) >= 3:
            timestamp = parts[0]
            arch = parts[1]
            rl_status = parts[2]
            timestamp = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}"
            return f"{timestamp} ({arch}, {rl_status})"
        return name
    except Exception:
        return name


def load_experiment_data(experiment):
    """Load history, metadata, and config for an experiment. Returns (history, metadata, config)."""
    history, metadata, config = None, None, None

    history_file = os.path.join(experiment, "history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    metadata_file = os.path.join(experiment, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    config_file = os.path.join(experiment, "config.yaml")
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            pass

    return history, metadata, config


# ============================================================
# Tab 1: Live Training Callbacks
# ============================================================


@app.callback(
    Output("experiment-selector", "options"),
    Input("interval-component", "n_intervals"),
)
def update_live_experiments(_):
    """Refresh the list of live training experiments."""
    return get_live_experiments()


@app.callback(
    [
        Output("epoch-progress-text", "children"),
        Output("epoch-progress-bar-inner", "style"),
    ],
    [Input("experiment-selector", "value"), Input("interval-component", "n_intervals")],
)
def update_epoch_progress(experiment, _):
    """Update the epoch progress bar for the selected experiment."""
    base_style = {
        "height": "20px",
        "borderRadius": "4px",
        "transition": "width 0.5s",
    }
    if not experiment:
        return "No experiment selected", {**base_style, "width": "0%", "backgroundColor": "#2196F3"}

    _, metadata, _ = load_experiment_data(experiment)
    if not metadata:
        return "No metadata available", {**base_style, "width": "0%", "backgroundColor": "#2196F3"}

    current = metadata.get("current_epoch", 0)
    total = metadata.get("training_params", {}).get("num_epochs", 0) or metadata.get(
        "total_epochs", 0
    )
    status = metadata.get("status", "unknown")

    pct = min(100, int(current / total * 100)) if total > 0 else 0

    text = f"Epoch {current}/{total}"
    if status == "completed":
        text += " (Completed)"
        pct = 100
    elif status == "running":
        text += " (Running)"

    color = "#4CAF50" if status == "completed" else "#2196F3"

    return text, {**base_style, "width": f"{pct}%", "backgroundColor": color}


@app.callback(
    [
        Output("loss-graph", "figure"),
        Output("experiment-details", "children"),
    ],
    [
        Input("experiment-selector", "value"),
        Input("interval-component", "n_intervals"),
        Input("loss-view-toggle", "value"),
    ],
)
def update_graphs(experiment, _, loss_view):
    """Update the training loss and metric graphs for the selected experiment."""
    if not experiment:
        return (
            create_empty_figure("Select an experiment", "Training Progress"),
            "No experiment selected",
        )

    try:
        history, metadata, config = load_experiment_data(experiment)

        if not history or "train_loss" not in history:
            return (
                create_empty_figure("No training data yet", "Training Progress"),
                json.dumps(metadata, indent=2) if metadata else "No metadata available",
            )

        # Build experiment details string
        if metadata:
            if (
                "training_time_minutes" in metadata
                and metadata["training_time_minutes"] is not None
            ):
                metadata["formatted_training_time"] = (
                    f"{metadata['training_time_minutes']:.2f} minutes"
                )
            experiment_details = json.dumps(metadata, indent=2)
        elif config:
            experiment_details = f"Experiment: {os.path.basename(experiment)}\nConfiguration:\n{json.dumps(config, indent=2)}"
        else:
            experiment_details = "No details available"

        # Extract training params
        val_frequency = 10
        total_epochs = len(history["train_loss"])
        early_stopping_triggered = False
        training_completed = False

        if metadata:
            tp = metadata.get("training_params", {})
            val_frequency = tp.get("validation_frequency", val_frequency)
            total_epochs = tp.get("num_epochs", total_epochs)
            early_stopping_triggered = metadata.get("early_stopping_triggered", False)
            training_completed = metadata.get("status") == "completed" or "end_time" in metadata
        elif config and "training" in config:
            val_frequency = config["training"].get("validation_frequency", val_frequency)
            total_epochs = config["training"].get("num_epochs", total_epochs)

        # Build loss figure
        train_epochs = list(range(1, len(history["train_loss"]) + 1))
        final_epoch = len(history["train_loss"])

        # Compute validation epochs once
        val_epochs = []
        if "val_loss" in history and history["val_loss"]:
            val_epochs = list(
                range(
                    val_frequency,
                    len(history["val_loss"]) * val_frequency + 1,
                    val_frequency,
                )
            )
            val_epochs = val_epochs[: len(history["val_loss"])]

        show_breakdown = loss_view == "breakdown"

        loss_fig = go.Figure()

        if not show_breakdown:
            # Total loss view: training + validation only
            loss_fig.add_trace(
                go.Scatter(
                    x=train_epochs,
                    y=history["train_loss"],
                    name="Training Loss",
                    line=dict(color="#1f77b4"),
                )
            )
            if val_epochs:
                loss_fig.add_trace(
                    go.Scatter(
                        x=val_epochs,
                        y=history["val_loss"],
                        name="Validation Loss",
                        line=dict(color="#ff7f0e"),
                    )
                )
        else:
            # Breakdown view: residual, boundary, initial only
            component_colors = {
                "residual_loss": "#d62728",
                "boundary_loss": "#2ca02c",
                "initial_loss": "#9467bd",
            }
            for component in ["residual_loss", "boundary_loss", "initial_loss"]:
                if component in history and history[component]:
                    if val_epochs:
                        component_epochs = val_epochs[: len(history[component])]
                    else:
                        component_epochs = list(range(1, len(history[component]) + 1))
                    loss_fig.add_trace(
                        go.Scatter(
                            x=component_epochs,
                            y=history[component],
                            name=component.replace("_", " ").title(),
                            line=dict(color=component_colors[component]),
                        )
                    )

        # Status annotation
        status_text = ""
        if early_stopping_triggered:
            status_text = f"Early stopping at epoch {final_epoch}/{total_epochs}"
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

        return loss_fig, experiment_details

    except Exception as e:
        print(f"Error loading data: {e}")
        return create_empty_figure(f"Error: {e}", "Training Progress"), f"Error: {str(e)}"


# ============================================================
# Tab 1: Launch trainer
# ============================================================


# PDE name -> config key mapping
_PDE_NAME_TO_KEY = {}
for _k, _c in _PDE_CONFIGS.items():
    _PDE_NAME_TO_KEY[_c.get("name", _k)] = _k


@app.callback(
    Output("train-arch-selector", "value"),
    Output("pde-info-box", "children"),
    Input("train-pde-selector", "value"),
    prevent_initial_call=True,
)
def update_pde_selection(pde_name):
    """When PDE changes, auto-select recommended architecture and show PDE info."""
    pde_key = _PDE_NAME_TO_KEY.get(pde_name, "heat")
    pde_cfg = _PDE_CONFIGS.get(pde_key, {})
    recommended_arch = pde_cfg.get("architecture", "fourier")
    params = pde_cfg.get("parameters", {})
    domain = pde_cfg.get("domain", [])
    time_domain = pde_cfg.get("time_domain", [])
    dim = pde_cfg.get("dimension", 1)

    info = [
        html.Strong(f"{pde_name}"),
        html.Span(f"  |  {dim}D  |  Recommended arch: {recommended_arch}"),
        html.Br(),
        html.Span(f"Domain: {domain}  |  Time: {time_domain}"),
        html.Br(),
        html.Span(f"Parameters: {params}"),
    ]
    return recommended_arch, info


@app.callback(
    Output("trainer-status", "children"),
    Input("launch-trainer-button", "n_clicks"),
    State("train-pde-selector", "value"),
    State("train-arch-selector", "value"),
    State("train-epochs-input", "value"),
    State("train-lr-input", "value"),
    State("train-batch-size", "value"),
    State("train-collocation-points", "value"),
    State("train-boundary-points", "value"),
    State("train-initial-points", "value"),
    State("train-device-selector", "value"),
    State("train-rl-toggle", "value"),
    prevent_initial_call=True,
)
def launch_trainer(
    n_clicks,
    pde_name,
    arch,
    epochs,
    lr,
    batch_size,
    collocation_pts,
    boundary_pts,
    initial_pts,
    device,
    rl_toggle,
):
    """Launch a headless training subprocess with the selected parameters."""
    if not n_clicks:
        return ""
    try:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_script = os.path.join(project_root, "src", "training", "train.py")
        config_path = os.path.join(project_root, "src", "config", "config.yaml")
        if not os.path.exists(config_path):
            config_path = os.path.join(project_root, "config.yaml")
        log_file = os.path.join(project_root, "trainer_launch.log")

        if not os.path.exists(train_script):
            return "Error: src/training/train.py not found"

        use_rl = "rl" in (rl_toggle or [])

        cmd = [
            sys.executable,
            train_script,
            "--pde",
            pde_name,
            "--arch",
            arch,
            "--epochs",
            str(epochs or 500),
            "--lr",
            str(lr or 0.005),
            "--batch-size",
            str(batch_size or 2048),
            "--collocation-points",
            str(collocation_pts or 5000),
            "--boundary-points",
            str(boundary_pts or 5000),
            "--initial-points",
            str(initial_pts or 5000),
            "--device",
            device or "cpu",
            "--config",
            config_path,
        ]
        if use_rl:
            cmd.append("--rl")

        log_fh = open(log_file, "w")
        subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            cwd=project_root,
        )

        rl_str = " + RL" if use_rl else ""
        return html.Div(
            [
                html.Span("Training started! ", style={"color": "#4CAF50", "fontWeight": "bold"}),
                html.Span(f"{pde_name} / {arch}{rl_str} / {epochs} epochs / LR={lr}"),
                html.Br(),
                html.Span(
                    "Switch to the Monitor sub-tab to track progress.",
                    style={"fontSize": "12px", "color": "#888"},
                ),
            ]
        )
    except Exception as e:
        return f"Error launching training: {e}"


# ============================================================
# Tab 1: Download report
# ============================================================


@app.callback(
    Output("download-report", "data"),
    Input("download-report-button", "n_clicks"),
    State("experiment-selector", "value"),
    prevent_initial_call=True,
)
def download_report(n_clicks, experiment):
    """Generate and download a JSON report for the selected experiment."""
    if not experiment or not n_clicks:
        return None

    try:
        loss_fig = update_graphs(experiment, None, "total")[0]
        colloc_fig = update_collocation_plot(experiment)
        exact_solution, predicted_solution = update_solution_visualizations(experiment, 0.5)

        _, metadata, _ = load_experiment_data(experiment)
        metadata = metadata or {}

        figures = {
            "loss_plot": f"Plotly.newPlot('loss-plot', {loss_fig.to_json()})",
            "collocation_plot": f"Plotly.newPlot('collocation-plot', {colloc_fig.to_json()})",
            "exact_solution": f"Plotly.newPlot('exact-solution', {exact_solution.to_json()})",
            "predicted_solution": f"Plotly.newPlot('predicted-solution', {predicted_solution.to_json()})",
        }

        html_content = generate_html_report(experiment, figures, metadata)
        exp_name = os.path.basename(experiment)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pinns_report_{exp_name}_{timestamp}.html"

        return dict(content=html_content, filename=filename, type="text/html")

    except Exception as e:
        print(f"Error generating report: {e}")
        return None


# ============================================================
# Tab 2: Comparison Callbacks
# ============================================================


@app.callback(
    Output("architecture-comparison", "figure"),
    Input("refresh-comparisons-button", "n_clicks"),
    prevent_initial_call=False,
)
def update_architecture_comparison(_):
    """Refresh the architecture comparison chart from completed experiments."""
    architectures = {}

    experiment_dirs = []
    if os.path.exists("experiments"):
        experiment_dirs.extend(
            os.path.join("experiments", d)
            for d in os.listdir("experiments")
            if os.path.isdir(os.path.join("experiments", d))
        )

    for exp_dir in experiment_dirs:
        architecture = None

        metadata_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                    architecture = meta.get("architecture")
                    if not architecture and "config" in meta and "model" in meta["config"]:
                        architecture = meta["config"]["model"].get("architecture")
            except Exception:
                pass

        if not architecture:
            config_file = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        cfg = yaml.safe_load(f)
                        if "model" in cfg and "architecture" in cfg["model"]:
                            architecture = cfg["model"]["architecture"]
                except Exception:
                    pass

        if not architecture:
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split("_")
            architecture = parts[2] if len(parts) >= 3 else "unknown"

        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            continue

        try:
            with open(history_file, "r") as f:
                hist = json.load(f)

            if architecture not in architectures:
                architectures[architecture] = []
            architectures[architecture].append(
                {
                    "name": os.path.basename(exp_dir),
                    "train_loss": hist.get("train_loss", []),
                    "val_loss": hist.get("val_loss", []),
                }
            )
        except Exception:
            continue

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for arch, experiments in architectures.items():
        for exp in experiments:
            if exp["train_loss"]:
                fig.add_trace(
                    go.Scatter(
                        y=exp["train_loss"],
                        name=f"{arch} - {exp['name']}",
                        line=dict(color=colors[color_idx % len(colors)], dash="solid"),
                    )
                )
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
        yaxis_type="log",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
    )
    return fig


@app.callback(
    Output("pde-comparison", "figure"),
    Input("refresh-comparisons-button", "n_clicks"),
    prevent_initial_call=False,
)
def update_pde_comparison(_):
    """Refresh the PDE comparison chart from completed experiments."""
    pdes = {}

    experiment_dirs = []
    if os.path.exists("experiments"):
        experiment_dirs.extend(
            os.path.join("experiments", d)
            for d in os.listdir("experiments")
            if os.path.isdir(os.path.join("experiments", d))
        )

    for exp_dir in experiment_dirs:
        pde_type = None

        metadata_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, "r") as f:
                    meta = json.load(f)
                    pde_type = meta.get("pde_type") or meta.get("pde_name")
            except Exception:
                pass

        if not pde_type:
            config_file = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(config_file):
                try:
                    with open(config_file, "r") as f:
                        cfg = yaml.safe_load(f)
                        pde_type = cfg.get("pde_type") or cfg.get("pde", {}).get("name")
                except Exception:
                    pass

        if not pde_type:
            dir_name = os.path.basename(exp_dir)
            parts = dir_name.split("_")
            pde_type = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else dir_name

        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            alt_files = glob.glob(os.path.join(exp_dir, "**", "history.json"), recursive=True)
            if alt_files:
                history_file = alt_files[0]
            else:
                continue

        try:
            with open(history_file, "r") as f:
                hist = json.load(f)

            comp_time = None
            hover_text = f"Experiment: {os.path.basename(exp_dir)}"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        meta = json.load(f)
                        comp_time = meta.get("training_time_minutes")
                        if comp_time:
                            hover_text += f"<br>Training time: {comp_time:.2f} minutes"
                except Exception:
                    pass

            if pde_type not in pdes:
                pdes[pde_type] = []
            pdes[pde_type].append(
                {
                    "name": os.path.basename(exp_dir),
                    "train_loss": hist.get("train_loss", []),
                    "computation_time": comp_time,
                    "hover_text": hover_text,
                }
            )
        except Exception:
            continue

    fig = go.Figure()
    colors = px.colors.qualitative.Plotly
    color_idx = 0

    for pde_type, experiments in pdes.items():
        for exp in experiments:
            if exp["train_loss"]:
                name = f"{pde_type} - {exp['name']}"
                if exp["computation_time"]:
                    name += f" ({exp['computation_time']:.2f} min)"
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
        yaxis_type="log",
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05),
    )
    return fig


# ============================================================
# Tab 3: Collocation & Solution Callbacks
# ============================================================


@app.callback(
    Output("collocation-experiment-selector", "options"),
    Input("main-tabs", "value"),
)
def update_collocation_experiments(tab):
    """Populate the experiment selector when the collocation tab is active."""
    if tab == "collocation":
        return get_experiments()
    return dash.no_update


@app.callback(
    Output("collocation-evolution", "figure"),
    Input("collocation-experiment-selector", "value"),
)
def update_collocation_plot(experiment):
    """Render the collocation-point distribution plot for the selected experiment."""
    if not experiment:
        return create_empty_figure("Select an experiment", "Collocation Points Distribution")

    try:
        # Look for collocation visualizations
        vis_dir = os.path.join(experiment, "visualizations")
        vis_files = []

        if os.path.exists(vis_dir):
            vis_files = sorted(
                [
                    f
                    for f in os.listdir(vis_dir)
                    if f.endswith(".png") and ("collocation" in f or "density" in f)
                ],
                key=lambda x: os.path.getmtime(os.path.join(vis_dir, x)),
                reverse=True,
            )

        if not vis_files:
            # Try recursive search
            vis_files_full = glob.glob(
                os.path.join(experiment, "**", "*collocation*.png"), recursive=True
            )
            if vis_files_full:
                vis_files_full.sort(key=os.path.getmtime, reverse=True)
                vis_path = vis_files_full[0]
            else:
                return create_empty_figure(
                    "No collocation visualization available",
                    "Collocation Points Distribution",
                )
        else:
            vis_path = os.path.join(vis_dir, vis_files[0])

        try:
            from PIL import Image

            img = np.array(Image.open(vis_path))
            colloc_fig = px.imshow(img)
            colloc_fig.update_layout(
                title="Collocation Points Distribution",
                coloraxis_showscale=False,
            )
            colloc_fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
            colloc_fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
            return colloc_fig
        except Exception as e:
            return create_empty_figure(
                f"Error loading image: {e}",
                "Collocation Points Distribution",
            )

    except Exception as e:
        return create_empty_figure(
            f"Error: {e}",
            "Collocation Points Distribution",
        )


def _infer_model_params(state_dict, architecture, input_dim, output_dim):
    """Infer model architecture parameters from checkpoint state_dict shapes.

    The saved config.yaml may not match the actual trained model weights,
    so we inspect tensor shapes to reconstruct the correct configuration.
    """
    params = {}
    keys = list(state_dict.keys())

    if architecture == "fourier":
        b_key = [k for k in keys if "fourier.B" in k]
        if b_key:
            params["mapping_size"] = state_dict[b_key[0]].shape[1]
        layer_keys = sorted([k for k in keys if "layers" in k and "weight" in k])
        if layer_keys:
            params["hidden_dim"] = state_dict[layer_keys[0]].shape[0]
            params["num_layers"] = len(layer_keys)
            hidden_dims = [state_dict[lk].shape[0] for lk in layer_keys[:-1]]
            if hidden_dims:
                params["hidden_dims"] = hidden_dims

    elif architecture == "resnet":
        il_key = [k for k in keys if "input_layer.weight" in k]
        if il_key:
            params["hidden_dim"] = state_dict[il_key[0]].shape[0]
        block_keys = {k.split("blocks.")[1].split(".")[0] for k in keys if "blocks." in k}
        if block_keys:
            params["num_blocks"] = len(block_keys)

    elif architecture == "siren":
        layer_keys = sorted([k for k in keys if "layers" in k and "weight" in k])
        if layer_keys:
            hidden_dims = [state_dict[lk].shape[0] for lk in layer_keys[:-1]]
            if hidden_dims:
                params["hidden_dims"] = hidden_dims
                params["hidden_dim"] = hidden_dims[0]
            params["num_layers"] = len(layer_keys)

    elif architecture == "attention":
        ip_key = [k for k in keys if "input_proj.weight" in k]
        if ip_key:
            params["hidden_dim"] = state_dict[ip_key[0]].shape[0]
        attn_keys = {
            k.split("attention_layers.")[1].split(".")[0] for k in keys if "attention_layers." in k
        }
        if attn_keys:
            params["num_layers"] = len(attn_keys)

    elif architecture == "autoencoder":
        enc_keys = sorted([k for k in keys if "encoder" in k and "weight" in k])
        if enc_keys:
            hidden_dims = [state_dict[ek].shape[0] for ek in enc_keys]
            if hidden_dims:
                params["hidden_dims"] = hidden_dims[:-1]
                params["latent_dim"] = hidden_dims[-1]

    elif architecture == "fno":
        lift_key = [k for k in keys if "lift.0.weight" in k]
        if lift_key:
            params["hidden_dim"] = state_dict[lift_key[0]].shape[0]
        block_keys = {k.split("blocks.")[1].split(".")[0] for k in keys if "blocks." in k}
        if block_keys:
            params["num_blocks"] = len(block_keys)

    elif architecture == "feedforward":
        layer_keys = sorted([k for k in keys if "layers" in k and "weight" in k])
        if layer_keys:
            hidden_dims = [state_dict[lk].shape[0] for lk in layer_keys[:-1]]
            if hidden_dims:
                params["hidden_dims"] = hidden_dims
                params["hidden_dim"] = hidden_dims[0]
            params["num_layers"] = len(layer_keys)

    return params


@app.callback(
    [
        Output("exact-solution-3d", "figure"),
        Output("predicted-solution-3d", "figure"),
    ],
    [
        Input("collocation-experiment-selector", "value"),
        Input("time-slider", "value"),
    ],
)
def update_solution_visualizations(experiment, time_point):
    """Update the exact and predicted 3-D solution surfaces."""
    if not experiment:
        return create_empty_3d_figure("Select an experiment"), create_empty_3d_figure(
            "Select an experiment"
        )

    try:
        import torch

        sys.path.append(".")
        from pinnrl.neural_networks import PINNModel
        from pinnrl.pdes.pde_base import PDEConfig
        from pinnrl.utils.utils import plot_solution

        model_path = os.path.join(experiment, "final_model.pt")
        config_path = os.path.join(experiment, "config.yaml")

        if not os.path.exists(model_path) or not os.path.exists(config_path):
            missing = []
            if not os.path.exists(model_path):
                missing.append("final_model.pt")
            if not os.path.exists(config_path):
                missing.append("config.yaml")
            msg = f"Missing: {', '.join(missing)}"
            print(f"[Solution Viz] {msg} in {experiment}")
            return create_empty_3d_figure(msg), create_empty_3d_figure(msg)

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        pde_type = config.get("pde_type", "unknown").lower()

        pde_config_dict = config.get("pde_configs", {}).get(pde_type, {})
        if not pde_config_dict:
            msg = f"No configuration for PDE type: {pde_type}"
            return create_empty_3d_figure(msg), create_empty_3d_figure(msg)

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

        # Create PDE instance
        pde_classes = {
            "heat": ("src.pdes.heat_equation", "HeatEquation"),
            "wave": ("src.pdes.wave_equation", "WaveEquation"),
            "burgers": ("src.pdes.burgers_equation", "BurgersEquation"),
            "convection": ("src.pdes.convection_equation", "ConvectionEquation"),
            "kdv": ("src.pdes.kdv_equation", "KdVEquation"),
            "pendulum": ("src.pdes.pendulum_equation", "PendulumEquation"),
        }

        pde = None
        for key, (module, cls_name) in pde_classes.items():
            if key in pde_type:
                mod = __import__(module, fromlist=[cls_name])
                pde = getattr(mod, cls_name)(config=pde_config)
                break

        if pde is None:
            if "allen" in pde_type and "cahn" in pde_type:
                from pinnrl.pdes.allen_cahn import AllenCahnEquation

                pde = AllenCahnEquation(config=pde_config)
            elif "cahn" in pde_type and "hilliard" in pde_type:
                from pinnrl.pdes.cahn_hilliard import CahnHilliardEquation

                pde = CahnHilliardEquation(config=pde_config)
            elif "black" in pde_type or "scholes" in pde_type:
                from pinnrl.pdes.black_scholes import BlackScholesEquation

                pde = BlackScholesEquation(config=pde_config)
            else:
                msg = f"Unsupported PDE type: {pde_type}"
                return create_empty_3d_figure(msg), create_empty_3d_figure(msg)

        # Load model — infer architecture from checkpoint state_dict
        # because saved config.yaml may not match actual trained weights
        device = torch.device("cpu")
        from pinnrl.config import Config, ModelConfig

        saved_model = config.get("model", {})
        architecture = saved_model.get("architecture", "feedforward")
        input_dim = saved_model.get("input_dim", 2 if pde_config.dimension == 1 else 3)
        output_dim = saved_model.get("output_dim", 1)

        try:
            state_dict = torch.load(model_path, map_location=device, weights_only=False)

            # Infer model params from state_dict shapes
            inferred = _infer_model_params(state_dict, architecture, input_dim, output_dim)

            model_cfg = ModelConfig(
                input_dim=input_dim,
                output_dim=output_dim,
                architecture=architecture,
                hidden_dim=inferred.get("hidden_dim", saved_model.get("hidden_dim", 128)),
                num_layers=inferred.get("num_layers", saved_model.get("num_layers", 4)),
                activation=saved_model.get("activation", "tanh"),
                dropout=saved_model.get("dropout", 0.0),
                layer_norm=saved_model.get("layer_norm", False),
            )
            # Set architecture-specific params (inferred takes priority)
            for key in [
                "mapping_size",
                "scale",
                "omega_0",
                "num_heads",
                "num_blocks",
                "latent_dim",
                "hidden_dims",
                "modes",
                "periodic",
            ]:
                val = inferred.get(key, saved_model.get(key))
                if val is not None:
                    setattr(model_cfg, key, val)

            cfg = Config.__new__(Config)
            cfg.device = device
            cfg.model = model_cfg

            model = PINNModel(config=cfg, device=device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            return create_empty_3d_figure(msg), create_empty_3d_figure(msg)

        try:
            exact_fig, predicted_fig = plot_solution(
                model=model,
                pde=pde,
                num_points=1000,
                time_point=time_point,
                return_figs=True,
            )
            return exact_fig, predicted_fig
        except Exception as e:
            msg = f"Error generating visualization: {str(e)}"
            return create_empty_3d_figure(msg), create_empty_3d_figure(msg)

    except Exception as e:
        msg = f"Error: {str(e)}"
        return create_empty_3d_figure(msg), create_empty_3d_figure(msg)


# ============================================================
# HTML Report Generator
# ============================================================


def generate_html_report(experiment_path, figures, metadata):
    """Generate an interactive HTML report for the experiment."""
    config_path = os.path.join(experiment_path, "config.yaml")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception:
        config = {}

    exp_name = get_experiment_name(experiment_path)
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
                display: inline-block; padding: 5px 10px; border-radius: 15px;
                margin: 5px; color: white; font-weight: bold;
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
                    <span class="status-badge" style="background-color: #007bff;">{architecture}</span>
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
                <div id="exact-solution" class="plot"></div>
                <div id="predicted-solution" class="plot"></div>
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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()
    port = args.port

    print(f"Starting PINNs-RL-PDE Training Monitor on port {port}")
    print(f"Open http://127.0.0.1:{port}/ in your browser")

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
