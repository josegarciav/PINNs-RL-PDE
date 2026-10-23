"""Dash web dashboard for monitoring and comparing PINN training runs."""

import glob
import json
import os
import shutil
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

from pinnrl.config import DEFAULT_CONFIG_PATH
from pinnrl.datasets import WELL_REGISTRY
from pinnrl.pdes.registry import PDE_SPECS, get_pde_spec
from pinnrl.plots import loss_curve_epochs
from pinnrl.training.train import parse_run_name

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
_CONFIG_PATH = DEFAULT_CONFIG_PATH
try:
    with open(_CONFIG_PATH) as _f:
        _YAML_CONFIG = yaml.safe_load(_f)
except Exception:
    _YAML_CONFIG = {}

_TRAINING_DEFAULTS = _YAML_CONFIG.get("training", {})
_PDE_CONFIGS = _YAML_CONFIG.get("pde_configs", {})
_ARCHITECTURES = list(_YAML_CONFIG.get("architectures", {}).keys())

# Training runs are written here by pinnrl-train, which the dashboard launches
# from its own working directory; the path is resolved against that directory.
RESULTS_DIR = os.path.abspath(_YAML_CONFIG.get("paths", {}).get("results_dir", "runs"))

# PDE dropdown: display names from the shared registry (2D variant hidden for now).
_PDE_OPTIONS = [
    {"label": spec.display_name, "value": spec.display_name}
    for spec in PDE_SPECS
    if spec.key in _PDE_CONFIGS and spec.key != "heat_2d"
]
_PDE_NAME_TO_KEY = {spec.display_name: spec.key for spec in PDE_SPECS}
_DATASET_OPTIONS = [
    {
        "label": f"{_name} ({_entry.n_spatial_dims}D, {_entry.recommended_mode})",
        "value": _name,
    }
    for _name, _entry in sorted(WELL_REGISTRY.items())
]

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
                                                                            "Sampling Strategy:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-sampling-strategy",
                                                                            options=[
                                                                                {
                                                                                    "label": "Uniform (grid + jitter)",
                                                                                    "value": "uniform",
                                                                                },
                                                                                {
                                                                                    "label": "Stratified (Latin Hypercube)",
                                                                                    "value": "stratified",
                                                                                },
                                                                                {
                                                                                    "label": "Residual-based (resample where residual is high)",
                                                                                    "value": "residual_based",
                                                                                },
                                                                                {
                                                                                    "label": "RL bandit (experimental)",
                                                                                    "value": "rl_adaptive",
                                                                                },
                                                                            ],
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "collocation_distribution",
                                                                                "uniform",
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Mode:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-mode-selector",
                                                                            options=[
                                                                                {
                                                                                    "label": "Forward (solve)",
                                                                                    "value": "forward",
                                                                                },
                                                                                {
                                                                                    "label": "Inverse (identify)",
                                                                                    "value": "inverse",
                                                                                },
                                                                                {
                                                                                    "label": "Data-augmented (residual + data)",
                                                                                    "value": "data_augmented",
                                                                                },
                                                                                {
                                                                                    "label": "Data-only (regression)",
                                                                                    "value": "data_only",
                                                                                },
                                                                            ],
                                                                            value="forward",
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                # Dataset section: pick a Well dataset and the
                                                                # callbacks below auto-fill the PDE / arch / mode.
                                                                html.Div(
                                                                    [
                                                                        html.H4(
                                                                            "Dataset (optional)",
                                                                            style={
                                                                                "borderBottom": "2px solid #FF9800",
                                                                                "paddingBottom": "8px",
                                                                                "marginTop": "16px",
                                                                            },
                                                                        ),
                                                                        html.Div(
                                                                            [
                                                                                dcc.Checklist(
                                                                                    id="train-dataset-toggle",
                                                                                    options=[
                                                                                        {
                                                                                            "label": " Train on a Well benchmark dataset",
                                                                                            "value": "well",
                                                                                        }
                                                                                    ],
                                                                                    value=[],
                                                                                ),
                                                                            ],
                                                                            style=FIELD_STYLE,
                                                                        ),
                                                                        html.Div(
                                                                            id="train-dataset-controls",
                                                                            style={
                                                                                "display": "none"
                                                                            },
                                                                            children=[
                                                                                html.Div(
                                                                                    [
                                                                                        html.Label(
                                                                                            "Dataset:",
                                                                                            style=LABEL_STYLE,
                                                                                        ),
                                                                                        dcc.Dropdown(
                                                                                            id="train-dataset-selector",
                                                                                            options=_DATASET_OPTIONS,
                                                                                            value=(
                                                                                                _DATASET_OPTIONS[
                                                                                                    0
                                                                                                ][
                                                                                                    "value"
                                                                                                ]
                                                                                                if _DATASET_OPTIONS
                                                                                                else None
                                                                                            ),
                                                                                            clearable=False,
                                                                                        ),
                                                                                    ],
                                                                                    style=FIELD_STYLE,
                                                                                ),
                                                                                html.Div(
                                                                                    id="train-dataset-info",
                                                                                    style={
                                                                                        "fontSize": "12px",
                                                                                        "padding": "8px",
                                                                                        "backgroundColor": "#FFF3E0",
                                                                                        "borderRadius": "4px",
                                                                                        "marginBottom": "12px",
                                                                                    },
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Label(
                                                                                            "Source:",
                                                                                            style=LABEL_STYLE,
                                                                                        ),
                                                                                        dcc.RadioItems(
                                                                                            id="train-dataset-source",
                                                                                            options=[
                                                                                                {
                                                                                                    "label": " Hugging Face streaming",
                                                                                                    "value": "hf",
                                                                                                },
                                                                                                {
                                                                                                    "label": " Local download dir",
                                                                                                    "value": "local",
                                                                                                },
                                                                                            ],
                                                                                            value="hf",
                                                                                            inline=True,
                                                                                        ),
                                                                                    ],
                                                                                    style=FIELD_STYLE,
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Label(
                                                                                            "Local base path:",
                                                                                            style=LABEL_STYLE,
                                                                                        ),
                                                                                        dcc.Input(
                                                                                            id="train-dataset-base",
                                                                                            type="text",
                                                                                            placeholder="/path/to/the_well_data",
                                                                                            value="",
                                                                                            style={
                                                                                                "width": "100%"
                                                                                            },
                                                                                        ),
                                                                                    ],
                                                                                    id="train-dataset-base-row",
                                                                                    style={
                                                                                        "display": "none"
                                                                                    },
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Label(
                                                                                            "Trajectories:",
                                                                                            style=LABEL_STYLE,
                                                                                        ),
                                                                                        dcc.Input(
                                                                                            id="train-dataset-traj",
                                                                                            type="number",
                                                                                            value=1,
                                                                                            min=1,
                                                                                            step=1,
                                                                                            style={
                                                                                                "width": "100%"
                                                                                            },
                                                                                        ),
                                                                                    ],
                                                                                    style=FIELD_STYLE,
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Label(
                                                                                            "Sampled points:",
                                                                                            style=LABEL_STYLE,
                                                                                        ),
                                                                                        dcc.Input(
                                                                                            id="train-dataset-points",
                                                                                            type="number",
                                                                                            value=4096,
                                                                                            min=64,
                                                                                            step=64,
                                                                                            style={
                                                                                                "width": "100%"
                                                                                            },
                                                                                        ),
                                                                                    ],
                                                                                    style=FIELD_STYLE,
                                                                                ),
                                                                                html.Div(
                                                                                    [
                                                                                        html.Label(
                                                                                            "Time steps:",
                                                                                            style=LABEL_STYLE,
                                                                                        ),
                                                                                        dcc.Input(
                                                                                            id="train-dataset-steps",
                                                                                            type="number",
                                                                                            placeholder="dataset default",
                                                                                            min=1,
                                                                                            step=1,
                                                                                            style={
                                                                                                "width": "100%"
                                                                                            },
                                                                                        ),
                                                                                    ],
                                                                                    style=FIELD_STYLE,
                                                                                ),
                                                                            ],
                                                                        ),
                                                                    ],
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
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Optimizer:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-optimizer-selector",
                                                                            options=[
                                                                                {
                                                                                    "label": "Adam",
                                                                                    "value": "adam",
                                                                                },
                                                                                {
                                                                                    "label": "L-BFGS",
                                                                                    "value": "lbfgs",
                                                                                },
                                                                                {
                                                                                    "label": "Adam → L-BFGS (two-phase)",
                                                                                    "value": "adam_lbfgs",
                                                                                },
                                                                            ],
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "optimizer", "adam"
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                    ],
                                                                    style=FIELD_STYLE,
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Loss Function:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-loss-fn-selector",
                                                                            options=[
                                                                                {
                                                                                    "label": "MSE (mean squared error)",
                                                                                    "value": "mse",
                                                                                },
                                                                                {
                                                                                    "label": "MAE (mean absolute error)",
                                                                                    "value": "mae",
                                                                                },
                                                                                {
                                                                                    "label": "Huber (robust)",
                                                                                    "value": "huber",
                                                                                },
                                                                            ],
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "loss_function",
                                                                                "mse",
                                                                            ),
                                                                            clearable=False,
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
                                                # Training extras: silent defaults exposed as dropdowns.
                                                html.Div(
                                                    [
                                                        html.H4(
                                                            "Training Extras",
                                                            style={
                                                                "marginTop": "20px",
                                                                "borderBottom": "2px solid #9C27B0",
                                                                "paddingBottom": "8px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Adaptive Loss Weights:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-adaptive-weights",
                                                                            options=[
                                                                                {
                                                                                    "label": "Off (static loss weights)",
                                                                                    "value": "off",
                                                                                },
                                                                                {
                                                                                    "label": "LRW (gradient-balanced)",
                                                                                    "value": "lrw",
                                                                                },
                                                                                {
                                                                                    "label": "RBW (loss-balanced)",
                                                                                    "value": "rbw",
                                                                                },
                                                                            ],
                                                                            value=(
                                                                                _TRAINING_DEFAULTS.get(
                                                                                    "adaptive_weights",
                                                                                    {},
                                                                                ).get(
                                                                                    "strategy",
                                                                                    "rbw",
                                                                                )
                                                                                if _TRAINING_DEFAULTS.get(
                                                                                    "adaptive_weights",
                                                                                    {},
                                                                                ).get("enabled")
                                                                                else "off"
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                        html.Div(
                                                                            "LRW balances residual/BC/IC by gradient norms; RBW balances by loss magnitudes.",
                                                                            style={
                                                                                "fontSize": "11px",
                                                                                "color": "#888",
                                                                                "marginTop": "4px",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "48%",
                                                                        "display": "inline-block",
                                                                        "verticalAlign": "top",
                                                                        "paddingRight": "2%",
                                                                        "marginBottom": "12px",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "LR Scheduler:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-lr-scheduler",
                                                                            options=[
                                                                                {
                                                                                    "label": "Cosine annealing",
                                                                                    "value": "cosine",
                                                                                },
                                                                                {
                                                                                    "label": "Reduce on plateau",
                                                                                    "value": "reduce_lr",
                                                                                },
                                                                            ],
                                                                            value=_TRAINING_DEFAULTS.get(
                                                                                "scheduler_type",
                                                                                "cosine",
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                        html.Div(
                                                                            "Cosine: deterministic decay over a fixed epoch budget. Reduce-on-plateau: drops LR when the epoch training loss stalls.",
                                                                            style={
                                                                                "fontSize": "11px",
                                                                                "color": "#888",
                                                                                "marginTop": "4px",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "48%",
                                                                        "display": "inline-block",
                                                                        "verticalAlign": "top",
                                                                        "paddingLeft": "2%",
                                                                        "marginBottom": "12px",
                                                                    },
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Early Stopping:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-early-stopping",
                                                                            options=[
                                                                                {
                                                                                    "label": "On (stop when val loss plateaus)",
                                                                                    "value": "on",
                                                                                },
                                                                                {
                                                                                    "label": "Off (always run all epochs)",
                                                                                    "value": "off",
                                                                                },
                                                                            ],
                                                                            value=(
                                                                                "on"
                                                                                if _TRAINING_DEFAULTS.get(
                                                                                    "early_stopping",
                                                                                    {},
                                                                                ).get(
                                                                                    "enabled",
                                                                                    True,
                                                                                )
                                                                                else "off"
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                        html.Div(
                                                                            "Off is useful for benchmarking — every config sees the same epoch count.",
                                                                            style={
                                                                                "fontSize": "11px",
                                                                                "color": "#888",
                                                                                "marginTop": "4px",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "48%",
                                                                        "display": "inline-block",
                                                                        "verticalAlign": "top",
                                                                        "paddingRight": "2%",
                                                                        "marginBottom": "12px",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Gradient Clipping:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="train-gradient-clipping",
                                                                            options=[
                                                                                {
                                                                                    "label": "Off",
                                                                                    "value": 0.0,
                                                                                },
                                                                                {
                                                                                    "label": "0.5 (aggressive)",
                                                                                    "value": 0.5,
                                                                                },
                                                                                {
                                                                                    "label": "1.0 (default)",
                                                                                    "value": 1.0,
                                                                                },
                                                                                {
                                                                                    "label": "5.0 (loose)",
                                                                                    "value": 5.0,
                                                                                },
                                                                            ],
                                                                            value=float(
                                                                                _TRAINING_DEFAULTS.get(
                                                                                    "gradient_clipping",
                                                                                    1.0,
                                                                                )
                                                                            ),
                                                                            clearable=False,
                                                                        ),
                                                                        html.Div(
                                                                            "Caps the gradient norm. Tighter clipping stabilises stiff PDEs but slows learning.",
                                                                            style={
                                                                                "fontSize": "11px",
                                                                                "color": "#888",
                                                                                "marginTop": "4px",
                                                                            },
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "48%",
                                                                        "display": "inline-block",
                                                                        "verticalAlign": "top",
                                                                        "paddingLeft": "2%",
                                                                        "marginBottom": "12px",
                                                                    },
                                                                ),
                                                            ],
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Weight Decay:",
                                                                    style=LABEL_STYLE,
                                                                ),
                                                                dcc.Dropdown(
                                                                    id="train-weight-decay",
                                                                    options=[
                                                                        {
                                                                            "label": "Off (no L2 regularisation)",
                                                                            "value": 0.0,
                                                                        },
                                                                        {
                                                                            "label": "1e-4 (light)",
                                                                            "value": 1e-4,
                                                                        },
                                                                        {
                                                                            "label": "5e-4 (default)",
                                                                            "value": 5e-4,
                                                                        },
                                                                        {
                                                                            "label": "1e-3 (strong)",
                                                                            "value": 1e-3,
                                                                        },
                                                                    ],
                                                                    value=float(
                                                                        _TRAINING_DEFAULTS.get(
                                                                            "optimizer_config",
                                                                            {},
                                                                        ).get(
                                                                            "weight_decay",
                                                                            5e-4,
                                                                        )
                                                                    ),
                                                                    clearable=False,
                                                                ),
                                                                html.Div(
                                                                    "L2 penalty on weights. Helps generalisation but can blur sharp PDE features if too strong.",
                                                                    style={
                                                                        "fontSize": "11px",
                                                                        "color": "#888",
                                                                        "marginTop": "4px",
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "width": "48%",
                                                                "display": "inline-block",
                                                                "verticalAlign": "top",
                                                                "paddingRight": "2%",
                                                                "marginBottom": "12px",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                                # Inverse-mode options (revealed when Mode = Inverse)
                                                html.Div(
                                                    id="inverse-options-panel",
                                                    style={"display": "none"},
                                                    children=[
                                                        html.H4(
                                                            "Inverse Problem — Parameter Identification",
                                                            style={
                                                                "marginTop": "20px",
                                                                "borderBottom": "2px solid #FF9800",
                                                                "paddingBottom": "8px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Parameters to identify:",
                                                                    style=LABEL_STYLE,
                                                                ),
                                                                dcc.Checklist(
                                                                    id="train-identify-params",
                                                                    options=[],
                                                                    value=[],
                                                                ),
                                                            ],
                                                            style=FIELD_STYLE,
                                                        ),
                                                        html.Div(
                                                            id="train-initial-guess-inputs",
                                                            style=FIELD_STYLE,
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Observation source:",
                                                                    style=LABEL_STYLE,
                                                                ),
                                                                dcc.RadioItems(
                                                                    id="train-obs-source",
                                                                    options=[
                                                                        {
                                                                            "label": " Synthetic (from exact solution + noise)",
                                                                            "value": "synth",
                                                                        },
                                                                        {
                                                                            "label": " File (.npz with x,t,u)",
                                                                            "value": "file",
                                                                        },
                                                                    ],
                                                                    value="synth",
                                                                ),
                                                            ],
                                                            style=FIELD_STYLE,
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Observation points:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-obs-points",
                                                                            type="number",
                                                                            value=200,
                                                                            min=20,
                                                                            step=20,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "48%",
                                                                        "display": "inline-block",
                                                                        "paddingRight": "2%",
                                                                    },
                                                                ),
                                                                html.Div(
                                                                    [
                                                                        html.Label(
                                                                            "Noise std:",
                                                                            style=LABEL_STYLE,
                                                                        ),
                                                                        dcc.Input(
                                                                            id="train-obs-noise",
                                                                            type="number",
                                                                            value=0.01,
                                                                            min=0.0,
                                                                            step=0.001,
                                                                            style={"width": "100%"},
                                                                        ),
                                                                    ],
                                                                    style={
                                                                        "width": "48%",
                                                                        "display": "inline-block",
                                                                        "paddingLeft": "2%",
                                                                    },
                                                                ),
                                                            ],
                                                            style=FIELD_STYLE,
                                                        ),
                                                        html.Div(
                                                            [
                                                                html.Label(
                                                                    "Observation file path (.npz):",
                                                                    style=LABEL_STYLE,
                                                                ),
                                                                dcc.Input(
                                                                    id="train-obs-path",
                                                                    type="text",
                                                                    value="",
                                                                    placeholder="/absolute/path/to/observations.npz",
                                                                    style={"width": "100%"},
                                                                ),
                                                            ],
                                                            style=FIELD_STYLE,
                                                        ),
                                                    ],
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
                                                        html.Div(
                                                            [
                                                                html.Div(
                                                                    dcc.Dropdown(
                                                                        id="experiment-selector",
                                                                        placeholder="Select running or recent experiment...",
                                                                    ),
                                                                    style={"flex": "1"},
                                                                ),
                                                                html.Button(
                                                                    "✕",
                                                                    id="delete-experiment-button",
                                                                    title="Delete this experiment folder permanently",
                                                                    n_clicks=0,
                                                                    style={
                                                                        "marginLeft": "8px",
                                                                        "padding": "0 12px",
                                                                        "height": "36px",
                                                                        "border": "1px solid #d32f2f",
                                                                        "backgroundColor": "#fff",
                                                                        "color": "#d32f2f",
                                                                        "borderRadius": "4px",
                                                                        "cursor": "pointer",
                                                                        "fontWeight": "bold",
                                                                    },
                                                                ),
                                                            ],
                                                            style={
                                                                "display": "flex",
                                                                "alignItems": "center",
                                                            },
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
                                                dcc.ConfirmDialog(
                                                    id="confirm-delete-experiment",
                                                    message="",
                                                ),
                                                html.Div(
                                                    id="delete-experiment-status",
                                                    style={
                                                        "marginTop": "8px",
                                                        "fontSize": "13px",
                                                        "textAlign": "center",
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
                                        # Live 3D solution snapshot
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Live Solution Snapshot",
                                                    style={"textAlign": "center"},
                                                ),
                                                html.Div(
                                                    id="live-snapshot-epoch",
                                                    style={
                                                        "textAlign": "center",
                                                        "fontStyle": "italic",
                                                        "marginBottom": "8px",
                                                    },
                                                ),
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            [
                                                                dcc.Graph(
                                                                    id="live-prediction-3d",
                                                                    style={"height": "55vh"},
                                                                )
                                                            ],
                                                            style={
                                                                "width": "50%",
                                                                "display": "inline-block",
                                                                "verticalAlign": "top",
                                                            },
                                                        ),
                                                        html.Div(
                                                            [
                                                                dcc.Graph(
                                                                    id="live-residual-3d",
                                                                    style={"height": "55vh"},
                                                                )
                                                            ],
                                                            style={
                                                                "width": "50%",
                                                                "display": "inline-block",
                                                                "verticalAlign": "top",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                            ],
                                            style={"marginTop": "20px"},
                                        ),
                                        # Identified-parameter trajectories (inverse mode only)
                                        html.Div(
                                            [
                                                dcc.Graph(id="param-trajectory-graph"),
                                            ],
                                            style={"width": "100%", "marginTop": "10px"},
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
                                                        i / 10: f"{i / 10:.1f}" for i in range(11)
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
    """List every run folder in ``RESULTS_DIR`` as dropdown options.

    Returns ``{"label", "value", "is_running"}`` dicts, newest first. Stale
    ``.running`` markers (run finished, no metadata or history, or older than
    an hour without metadata) are deleted on the way.
    """
    experiments = []
    experiments_dir = RESULTS_DIR

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
                            with open(meta_file) as mf:
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

                display_name = describe_run(exp_dir)
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
    """Return running experiments and those whose metadata changed in the last two hours."""
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
                with open(meta_file) as f:
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


def describe_run(name):
    """``"YYYY-MM-DD HH:MM - <subject> (<arch>) [rl|no_rl]"``, or ``name`` if it is not a run name."""
    parsed = parse_run_name(name)
    if parsed is None:
        return name
    rl = "rl" if parsed["rl"] else "no_rl"
    return f"{parsed['started'][:16]} - {parsed['subject']} ({parsed['architecture']}) [{rl}]"


def get_experiment_name(experiment_path):
    """Readable name of a run directory (see :func:`describe_run`)."""
    return describe_run(os.path.basename(experiment_path))


def load_experiment_data(experiment):
    """Load history, metadata, and config for an experiment. Returns (history, metadata, config)."""
    history, metadata, config = None, None, None

    history_file = os.path.join(experiment, "history.json")
    if os.path.exists(history_file):
        try:
            with open(history_file) as f:
                history = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    metadata_file = os.path.join(experiment, "metadata.json")
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    config_file = os.path.join(experiment, "config.yaml")
    if os.path.exists(config_file):
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
        except (yaml.YAMLError, OSError):
            pass

    return history, metadata, config


# ============================================================
# Tab 1: Live Training Callbacks
# ============================================================


@app.callback(
    Output("experiment-selector", "options"),
    [
        Input("interval-component", "n_intervals"),
        Input("delete-experiment-status", "children"),
    ],
)
def update_live_experiments(_n, _status):
    """Refresh the list of live training experiments."""
    return get_live_experiments()


@app.callback(
    [
        Output("confirm-delete-experiment", "displayed"),
        Output("confirm-delete-experiment", "message"),
    ],
    Input("delete-experiment-button", "n_clicks"),
    State("experiment-selector", "value"),
    prevent_initial_call=True,
)
def show_delete_experiment_confirm(n_clicks, experiment_path):
    """Open the confirmation dialog when the delete button is clicked."""
    if not n_clicks or not experiment_path:
        return False, ""
    folder = os.path.basename(os.path.normpath(experiment_path))
    message = (
        f"Permanently delete experiment folder '{folder}'?\n\n"
        "This removes the entire directory and all files inside it. "
        "This action cannot be undone."
    )
    return True, message


@app.callback(
    [
        Output("delete-experiment-status", "children"),
        Output("experiment-selector", "value"),
    ],
    Input("confirm-delete-experiment", "submit_n_clicks"),
    State("experiment-selector", "value"),
    prevent_initial_call=True,
)
def delete_experiment_folder(submit_n_clicks, experiment_path):
    """Delete the experiment folder when the user confirms."""
    if not submit_n_clicks or not experiment_path:
        return dash.no_update, dash.no_update

    experiments_root = RESULTS_DIR
    target = os.path.abspath(experiment_path)
    # Refuse to delete anything outside the results directory.
    if (
        os.path.commonpath([experiments_root, target]) != experiments_root
        or target == experiments_root
    ):
        return (
            html.Span(
                f"Refused to delete: path is outside {RESULTS_DIR}.",
                style={"color": "#d32f2f"},
            ),
            dash.no_update,
        )

    if not os.path.isdir(target):
        return (
            html.Span("Folder no longer exists.", style={"color": "#888"}),
            None,
        )

    try:
        shutil.rmtree(target)
    except OSError as exc:
        return (
            html.Span(f"Failed to delete: {exc}", style={"color": "#d32f2f"}),
            dash.no_update,
        )

    return (
        html.Span(
            f"Deleted '{os.path.basename(target)}'.",
            style={"color": "#4CAF50"},
        ),
        None,
    )


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
        train_epochs, val_epochs = loss_curve_epochs(history, val_frequency)
        final_epoch = len(train_epochs)

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
            # Breakdown view: the per-epoch loss components
            component_colors = {
                "residual_loss": "#d62728",
                "boundary_loss": "#2ca02c",
                "initial_loss": "#9467bd",
                "data_loss": "#8c564b",
            }
            for component in component_colors:
                if history.get(component) and any(history[component]):
                    loss_fig.add_trace(
                        go.Scatter(
                            x=train_epochs[: len(history[component])],
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
    Output("train-dataset-controls", "style"),
    Input("train-dataset-toggle", "value"),
)
def toggle_dataset_controls(toggle_value):
    """Show or hide the Well-dataset controls based on the master toggle."""
    if toggle_value and "well" in toggle_value:
        return {"display": "block"}
    return {"display": "none"}


@app.callback(
    Output("train-dataset-base-row", "style"),
    Input("train-dataset-source", "value"),
)
def toggle_dataset_base_row(source):
    """Reveal the local-path text input only when the source is 'local'."""
    if source == "local":
        return {**FIELD_STYLE, "display": "block"}
    return {"display": "none"}


@app.callback(
    Output("train-dataset-info", "children"),
    Output("train-pde-selector", "value"),
    Output("train-mode-selector", "value"),
    Input("train-dataset-selector", "value"),
    State("train-dataset-toggle", "value"),
    prevent_initial_call=True,
)
def populate_from_well_dataset(dataset_name, toggle_value):
    """Auto-fill PDE / mode from the registry when a dataset is picked."""
    if not dataset_name or "well" not in (toggle_value or []):
        # Toggle off — leave info blank and don't touch the other dropdowns.
        return "", dash.no_update, dash.no_update
    entry = WELL_REGISTRY.get(dataset_name)
    if entry is None:
        return f"Unknown dataset: {dataset_name}", dash.no_update, dash.no_update

    info_children = [
        html.Strong(entry.name),
        html.Span(f"  ·  {entry.n_spatial_dims}D  ·  fields: {len(entry.fields)}"),
        html.Span(f"  ·  reads up to {entry.steps_to_read()} steps per trajectory by default"),
        html.Br(),
        html.Span(entry.description),
        html.Br(),
        html.Em(
            f"Recommended: mode={entry.recommended_mode}, arch={entry.default_architecture}"
            + (
                f", PDE={entry.default_pde_key}"
                if entry.default_pde_key
                else " (no matched analytical PDE)"
            )
        ),
    ]

    # Drive the PDE dropdown only when the dataset matches an existing PDE
    # in the framework. Unmatched datasets keep whatever PDE the user had
    # picked — the data_only loss recipe makes the choice irrelevant.
    pde_value = dash.no_update
    if entry.default_pde_key:
        pde_value = get_pde_spec(entry.default_pde_key).display_name

    return info_children, pde_value, entry.recommended_mode


@app.callback(
    Output("trainer-status", "children"),
    Input("start-training-button", "n_clicks"),
    State("train-pde-selector", "value"),
    State("train-arch-selector", "value"),
    State("train-epochs-input", "value"),
    State("train-lr-input", "value"),
    State("train-batch-size", "value"),
    State("train-collocation-points", "value"),
    State("train-boundary-points", "value"),
    State("train-initial-points", "value"),
    State("train-device-selector", "value"),
    State("train-sampling-strategy", "value"),
    State("train-optimizer-selector", "value"),
    State("train-mode-selector", "value"),
    State("train-identify-params", "value"),
    State("train-initial-guess-inputs", "children"),
    State("train-obs-source", "value"),
    State("train-obs-points", "value"),
    State("train-obs-noise", "value"),
    State("train-obs-path", "value"),
    State("train-loss-fn-selector", "value"),
    State("train-adaptive-weights", "value"),
    State("train-lr-scheduler", "value"),
    State("train-early-stopping", "value"),
    State("train-gradient-clipping", "value"),
    State("train-weight-decay", "value"),
    State("train-dataset-toggle", "value"),
    State("train-dataset-selector", "value"),
    State("train-dataset-source", "value"),
    State("train-dataset-base", "value"),
    State("train-dataset-traj", "value"),
    State("train-dataset-points", "value"),
    State("train-dataset-steps", "value"),
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
    sampling_strategy,
    optimizer,
    mode,
    identify_params,
    initial_guess_children,
    obs_source,
    obs_points,
    obs_noise,
    obs_path,
    loss_fn,
    adaptive_weights,
    lr_scheduler,
    early_stopping,
    gradient_clipping,
    weight_decay,
    dataset_toggle,
    dataset_name,
    dataset_source,
    dataset_base,
    dataset_traj,
    dataset_points,
    dataset_steps=None,
):
    """Launch a headless training subprocess with the selected parameters."""
    if not n_clicks:
        return ""
    try:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        log_file = os.path.join(RESULTS_DIR, "trainer_launch.log")

        use_rl = sampling_strategy == "rl_adaptive"

        cmd = [
            sys.executable,
            "-m",
            "pinnrl.training.train",
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
            _CONFIG_PATH,
        ]
        if use_rl:
            cmd.append("--rl")
        elif sampling_strategy:
            cmd += ["--collocation-distribution", sampling_strategy]
        if optimizer:
            cmd += ["--optimizer", optimizer]
        if loss_fn:
            cmd += ["--loss-function", loss_fn]
        if adaptive_weights is not None:
            cmd += ["--adaptive-weights", str(adaptive_weights)]
        if lr_scheduler:
            cmd += ["--lr-scheduler", lr_scheduler]
        if early_stopping is not None:
            cmd += ["--early-stopping", early_stopping]
        if gradient_clipping is not None:
            cmd += ["--gradient-clipping", str(float(gradient_clipping))]
        if weight_decay is not None:
            cmd += ["--weight-decay", str(float(weight_decay))]

        # Inverse-mode flags: only add when inverse mode is selected and at least
        # one parameter is checked, so the trainer falls back to the forward path
        # otherwise.
        mode = mode or "forward"
        if mode == "inverse" and identify_params:
            cmd += ["--mode", "inverse"]
            for name in identify_params:
                cmd += ["--identify", str(name)]
            # Initial guesses come from the dynamically-rendered inputs.
            for guess_spec in _extract_initial_guesses(initial_guess_children, identify_params):
                cmd += ["--initial-guess", guess_spec]
            if obs_source == "file" and obs_path:
                cmd += ["--obs-path", str(obs_path)]
            else:
                if obs_points:
                    cmd += ["--obs-points", str(int(obs_points))]
                if obs_noise is not None:
                    cmd += ["--obs-noise", str(float(obs_noise))]
        elif mode in ("data_only", "data_augmented"):
            # The dataset block carries the data source; apply the chosen
            # mode here so train.py overrides the yaml default.
            cmd += ["--mode", mode]

        # Well dataset flags.
        use_dataset = bool(dataset_toggle and "well" in dataset_toggle and dataset_name)
        if use_dataset:
            cmd += ["--dataset", str(dataset_name)]
            if dataset_traj:
                cmd += ["--dataset-traj", str(int(dataset_traj))]
            if dataset_points:
                cmd += ["--dataset-points", str(int(dataset_points))]
            if dataset_steps:
                cmd += ["--dataset-steps", str(int(dataset_steps))]
            if dataset_source == "local" and dataset_base:
                cmd += ["--dataset-base", str(dataset_base)]

        log_fh = open(log_file, "w")
        subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=log_fh,
            cwd=os.getcwd(),
        )

        rl_str = " + RL" if use_rl else ""
        mode_str = f" ({mode})" if mode != "forward" else ""
        dataset_str = f" / dataset={dataset_name}" if use_dataset else ""
        return html.Div(
            [
                html.Span("Training started! ", style={"color": "#4CAF50", "fontWeight": "bold"}),
                html.Span(
                    f"{pde_name} / {arch}{rl_str}{mode_str}{dataset_str} / {epochs} epochs / LR={lr} / opt={optimizer or 'adam'} / loss={loss_fn or 'mse'}"
                ),
                html.Br(),
                html.Span(
                    "Switch to the Monitor sub-tab to track progress.",
                    style={"fontSize": "12px", "color": "#888"},
                ),
            ]
        )
    except Exception as e:
        return f"Error launching training: {e}"


def _extract_initial_guesses(children, selected_params):
    """Pull ``name=value`` strings out of the dynamically-built guess inputs.

    The Mode=Inverse panel renders one numeric ``dcc.Input`` per checked
    parameter with id ``{"role": "initial-guess", "param": <name>}``. Dash
    serialises that tree as nested dicts under ``children`` State; this helper
    walks the tree to recover the ``name=value`` pairs the CLI expects.
    """
    if not children or not selected_params:
        return []
    selected = set(selected_params)
    guesses = []

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        props = node.get("props", {}) or {}
        node_id = props.get("id")
        if isinstance(node_id, dict) and node_id.get("role") == "initial-guess":
            param = node_id.get("param")
            value = props.get("value")
            if param in selected and value is not None:
                guesses.append(f"{param}={value}")
        # Recurse into children.
        sub = props.get("children")
        if sub is not None:
            visit(sub)

    visit(children)
    return guesses


# ============================================================
# Tab 1: Inverse-mode panel callbacks
# ============================================================


@app.callback(
    Output("inverse-options-panel", "style"),
    Input("train-mode-selector", "value"),
)
def toggle_inverse_panel(mode):
    """Show the parameter-identification panel only when Inverse mode is active."""
    if mode == "inverse":
        return {"display": "block", "marginTop": "12px"}
    return {"display": "none"}


@app.callback(
    [
        Output("train-identify-params", "options"),
        Output("train-identify-params", "value"),
    ],
    Input("train-pde-selector", "value"),
)
def populate_identify_options(pde_name):
    """Refresh the identifiable-parameter checklist whenever the PDE changes."""
    pde_key = _PDE_NAME_TO_KEY.get(pde_name, "heat")
    params = _PDE_CONFIGS.get(pde_key, {}).get("parameters", {}) or {}
    options = [
        {"label": f" {name} (true: {value})", "value": name} for name, value in params.items()
    ]
    return options, []


@app.callback(
    Output("train-initial-guess-inputs", "children"),
    [
        Input("train-identify-params", "value"),
        Input("train-pde-selector", "value"),
    ],
)
def render_initial_guess_inputs(selected_params, pde_name):
    """Build one numeric input per checked parameter, prefilled with the true value.

    Each input gets a pattern-matching id ``{"role": "initial-guess",
    "param": <name>}`` so ``_extract_initial_guesses`` can recover the values
    from the launch callback's ``State`` payload without a per-parameter
    callback.
    """
    if not selected_params:
        return []
    pde_key = _PDE_NAME_TO_KEY.get(pde_name, "heat")
    true_params = _PDE_CONFIGS.get(pde_key, {}).get("parameters", {}) or {}
    children = [html.Label("Initial guesses:", style=LABEL_STYLE)]
    for name in selected_params:
        prefill = float(true_params.get(name, 1.0))
        children.append(
            html.Div(
                [
                    html.Span(f"{name}:", style={"display": "inline-block", "width": "120px"}),
                    dcc.Input(
                        id={"role": "initial-guess", "param": name},
                        type="number",
                        value=prefill,
                        step=0.01,
                        style={"width": "200px"},
                    ),
                    html.Span(
                        f"  (true: {prefill})",
                        style={"marginLeft": "8px", "color": "#888", "fontStyle": "italic"},
                    ),
                ],
                style={"marginBottom": "6px"},
            )
        )
    return children


# ============================================================
# Tab 1 Monitor: live snapshot + parameter trajectory callbacks
# ============================================================


def _empty_3d_figure(message: str) -> go.Figure:
    """Placeholder 3D figure shown when no live snapshot exists yet."""
    fig = go.Figure()
    fig.update_layout(
        annotations=[
            dict(
                text=message,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="#888"),
            )
        ],
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


@app.callback(
    [
        Output("live-prediction-3d", "figure"),
        Output("live-residual-3d", "figure"),
        Output("live-snapshot-epoch", "children"),
    ],
    [
        Input("experiment-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_live_snapshot(experiment, _n_intervals):
    """Read ``live_snapshot.npz`` for the selected experiment and render two surfaces.

    The trainer writes this file every ``validation_frequency`` epochs (plus
    once before the first epoch and once after the last), so the Monitor tab
    updates roughly in step with the loss curve while training is running.
    """
    if not experiment:
        empty = _empty_3d_figure("Select an experiment to view live snapshot")
        return empty, empty, ""
    snapshot_path = os.path.join(experiment, "live_snapshot.npz")
    if not os.path.exists(snapshot_path):
        empty = _empty_3d_figure("Snapshot not available yet — waiting for first save")
        return empty, empty, ""
    try:
        data = np.load(snapshot_path)
        axis_x = data["axis_x"]
        axis_y = data["axis_y"]
        u_pred = data["u_pred"]
        residual = data["residual"]
        epoch = int(data["epoch"]) if "epoch" in data.files else 0
        dimension = int(data["dimension"]) if "dimension" in data.files else 1
        x_label = str(data["x_label"]) if "x_label" in data.files else "x"
        y_label = str(data["y_label"]) if "y_label" in data.files else "t"
        fixed_t = float(data["fixed_t"]) if "fixed_t" in data.files else float("nan")
    except Exception as exc:
        empty = _empty_3d_figure(f"Snapshot read error: {exc}")
        return empty, empty, ""

    pred_fig = go.Figure(data=[go.Surface(x=axis_x, y=axis_y, z=u_pred, colorscale="viridis")])
    pred_fig.update_layout(
        title="Predicted solution u",
        scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title="u"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    res_fig = go.Figure(
        data=[go.Surface(x=axis_x, y=axis_y, z=residual, colorscale="RdBu", reversescale=True)]
    )
    res_fig.update_layout(
        title="PDE residual",
        scene=dict(xaxis_title=x_label, yaxis_title=y_label, zaxis_title="residual"),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    caption = f"Snapshot @ epoch {epoch}"
    if dimension == 2 and not np.isnan(fixed_t):
        caption += f"  (slice at t = {fixed_t:.3f})"
    return pred_fig, res_fig, caption


@app.callback(
    Output("param-trajectory-graph", "figure"),
    [
        Input("experiment-selector", "value"),
        Input("interval-component", "n_intervals"),
    ],
)
def update_param_trajectories(experiment, _n_intervals):
    """Plot the trajectory of each identified PDE parameter over training epochs.

    Reads ``history.json`` for the ``param_<name>`` series produced by the
    trainer and draws a dashed reference line at the true value (recovered
    from ``metadata.json["true_parameters"]``). Without trainable parameters
    it shows a "No trainable PDE parameters" placeholder.
    """
    fig = go.Figure()
    if not experiment:
        fig.update_layout(
            title="Identified Parameters (inverse mode)",
            annotations=[
                dict(
                    text="Select an experiment",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
            height=320,
        )
        return fig
    history, metadata, _ = load_experiment_data(experiment)
    history = history or {}
    metadata = metadata or {}
    true_params = metadata.get("true_parameters", {}) or {}

    param_keys = [k for k in history.keys() if k.startswith("param_")]
    if not param_keys:
        fig.update_layout(
            title="Identified Parameters (forward run — none)",
            annotations=[
                dict(
                    text="No trainable PDE parameters in this run",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(color="#888"),
                )
            ],
            height=320,
        )
        return fig

    for key in param_keys:
        name = key[len("param_") :]
        series = history[key]
        epochs = list(range(1, len(series) + 1))
        fig.add_trace(go.Scatter(x=epochs, y=series, mode="lines", name=name))
        if name in true_params:
            fig.add_hline(
                y=float(true_params[name]),
                line_dash="dash",
                line_color="#888",
                annotation_text=f"true {name}={true_params[name]}",
                annotation_position="top right",
            )
    fig.update_layout(
        title="Identified Parameters",
        xaxis_title="Epoch",
        yaxis_title="Value",
        height=360,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


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
    """Refresh the architecture comparison chart from every run with a ``history.json``."""
    architectures = {}

    experiment_dirs = []
    if os.path.exists(RESULTS_DIR):
        experiment_dirs.extend(
            os.path.join(RESULTS_DIR, d)
            for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))
        )

    for exp_dir in experiment_dirs:
        architecture = None

        metadata_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file) as f:
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
                    with open(config_file) as f:
                        cfg = yaml.safe_load(f)
                        if "model" in cfg and "architecture" in cfg["model"]:
                            architecture = cfg["model"]["architecture"]
                except Exception:
                    pass

        if not architecture:
            parsed = parse_run_name(os.path.basename(exp_dir))
            architecture = parsed["architecture"] if parsed else "unknown"

        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            continue

        try:
            with open(history_file) as f:
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
    """Refresh the PDE comparison chart from every run with a ``history.json``."""
    pdes = {}

    experiment_dirs = []
    if os.path.exists(RESULTS_DIR):
        experiment_dirs.extend(
            os.path.join(RESULTS_DIR, d)
            for d in os.listdir(RESULTS_DIR)
            if os.path.isdir(os.path.join(RESULTS_DIR, d))
        )

    for exp_dir in experiment_dirs:
        pde_type = None

        metadata_file = os.path.join(exp_dir, "metadata.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file) as f:
                    meta = json.load(f)
                    pde_type = meta.get("pde_type") or meta.get("pde_name")
            except Exception:
                pass

        if not pde_type:
            config_file = os.path.join(exp_dir, "config.yaml")
            if os.path.exists(config_file):
                try:
                    with open(config_file) as f:
                        cfg = yaml.safe_load(f)
                        pde_type = cfg.get("pde_type") or cfg.get("pde", {}).get("name")
                except Exception:
                    pass

        if not pde_type:
            parsed = parse_run_name(os.path.basename(exp_dir))
            pde_type = parsed["subject"] if parsed else os.path.basename(exp_dir)

        history_file = os.path.join(exp_dir, "history.json")
        if not os.path.exists(history_file):
            alt_files = glob.glob(os.path.join(exp_dir, "**", "history.json"), recursive=True)
            if alt_files:
                history_file = alt_files[0]
            else:
                continue

        try:
            with open(history_file) as f:
                hist = json.load(f)

            comp_time = None
            hover_text = f"Experiment: {os.path.basename(exp_dir)}"
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file) as f:
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
    """Show the newest saved collocation PNG (e.g. ``visualizations/collocation_evolution.png``)."""
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
        from pinnrl.plots import plot_solution
        from pinnrl.training.train import load_run

        missing = [
            name
            for name in ("final_model.pt", "config.yaml")
            if not os.path.exists(os.path.join(experiment, name))
        ]
        if missing:
            msg = f"Missing: {', '.join(missing)}"
            return create_empty_3d_figure(msg), create_empty_3d_figure(msg)

        try:
            pde, model, _ = load_run(experiment)
        except Exception as e:
            msg = f"Error loading run: {e}"
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
        with open(config_path) as f:
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
                    <span class="status-badge {("rl-enabled" if rl_enabled else "rl-disabled")}">
                        RL {("Enabled" if rl_enabled else "Disabled")}
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
            {figures["loss_plot"]}
            {figures["collocation_plot"]}
            {figures["exact_solution"]}
            {figures["predicted_solution"]}
        </script>
    </body>
    </html>
    """
    return html_template
