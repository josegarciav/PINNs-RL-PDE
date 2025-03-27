import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
import glob

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div([
    html.H1("PINNs-RL-PDE Training Monitor", style={'textAlign': 'center'}),
    
    # Experiment selector
    html.Div([
        html.Label("Select Experiment:"),
        dcc.Dropdown(id='experiment-selector', placeholder='Select experiment...'),
    ], style={'marginBottom': '20px', 'width': '50%', 'margin': 'auto'}),
    
    # Main metrics panel
    html.Div([
        html.Div([
            dcc.Graph(id='loss-graph'),
        ], style={'width': '50%', 'display': 'inline-block'}),
        html.Div([
            dcc.Graph(id='collocation-evolution'),
        ], style={'width': '50%', 'display': 'inline-block'}),
    ]),
    
    # Architecture comparison
    html.H2("Architecture Comparison", style={'textAlign': 'center', 'marginTop': '30px'}),
    dcc.Graph(id='architecture-comparison'),
    
    # PDE comparison
    html.H2("PDE Comparison", style={'textAlign': 'center', 'marginTop': '30px'}),
    dcc.Graph(id='pde-comparison'),
    
    # Experiment metadata
    html.Div([
        html.H2("Experiment Details", style={'textAlign': 'center'}),
        html.Pre(id='experiment-details', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all',
            'backgroundColor': '#f5f5f5',
            'padding': '10px',
            'border': '1px solid #ddd',
            'borderRadius': '5px',
            'maxHeight': '400px',
            'overflow': 'auto'
        })
    ], style={'marginTop': '30px'}),
    
    # Automatic update interval
    dcc.Interval(id='interval-component', interval=5*1000, n_intervals=0),
], style={'padding': '20px'})

# Callback to update experiment list
@app.callback(
    Output('experiment-selector', 'options'),
    Input('interval-component', 'n_intervals')
)
def update_experiments(_):
    # Get list of experiment directories
    experiment_dirs = []
    for results_dir in ["results", "experiments"]:  # Check both possible result directories
        if os.path.exists(results_dir):
            experiment_dirs.extend([
                {"label": d, "value": os.path.join(results_dir, d)}
                for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))
            ])
    return experiment_dirs

# Callback to update main graphs
@app.callback(
    [Output('loss-graph', 'figure'),
     Output('collocation-evolution', 'figure'),
     Output('experiment-details', 'children')],
    [Input('experiment-selector', 'value'),
     Input('interval-component', 'n_intervals')]
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
            alt_files = glob.glob(os.path.join(experiment, "**", "history.json"), recursive=True)
            if alt_files:
                history_file = alt_files[0]
        
        with open(history_file, "r") as f:
            history = json.load(f)
        
        # Load metadata if available
        metadata_file = os.path.join(experiment, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                experiment_details = json.dumps(metadata, indent=2)
        else:
            # Create basic metadata from config if available
            config_file = os.path.join(experiment, "config.json")
            if os.path.exists(config_file):
                with open(config_file, "r") as f:
                    config = json.load(f)
                    experiment_details = f"Experiment: {os.path.basename(experiment)}\nConfiguration:\n{json.dumps(config, indent=2)}"
        
        # Loss figure
        loss_fig = go.Figure()
        loss_fig.add_trace(go.Scatter(y=history["train_loss"], name="Training Loss"))
        if "val_loss" in history and history["val_loss"]:
            loss_fig.add_trace(go.Scatter(y=history["val_loss"], name="Validation Loss"))
        
        # Add component losses if available
        for component in ["residual_loss", "boundary_loss", "initial_loss"]:
            if component in history and history[component]:
                loss_fig.add_trace(go.Scatter(y=history[component], name=component.replace("_", " ").title()))
                
        loss_fig.update_layout(
            title="Training Progress", 
            xaxis_title="Epoch", 
            yaxis_title="Loss",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        # Find the most recent collocation visualization
        vis_dir = "visualizations"
        if not os.path.exists(vis_dir):
            # Check if visualizations are stored in experiment directory
            vis_dir = os.path.join(experiment, "visualizations")
            if not os.path.exists(vis_dir):
                # Try to find any visualization files in the experiment directory
                vis_files = glob.glob(os.path.join(experiment, "**", "*collocation*.png"), recursive=True)
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
                            coloraxis_showscale=False
                        )
                        # Remove axis labels and ticks
                        colloc_fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
                        colloc_fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)
                        
                        return loss_fig, colloc_fig, experiment_details
                    except Exception as e:
                        print(f"Error loading visualization image: {e}")
        
        # Look for visualization files in the visualization directory
        if os.path.exists(vis_dir):
            latest_vis = sorted(
                [f for f in os.listdir(vis_dir) 
                if f.endswith(".png") and (
                    f.startswith("latest_collocation_evolution") or 
                    f.startswith("latest_density_heatmap") or
                    f.startswith("collocation_evolution_epoch") or
                    f.startswith("final_collocation_evolution")
                )],
                key=lambda x: os.path.getmtime(os.path.join(vis_dir, x)),
                reverse=True
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
                        coloraxis_showscale=False
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
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16)
        )
        colloc_fig.update_layout(
            title="Collocation Points Distribution",
            xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
            yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
        )
            
        return loss_fig, colloc_fig, experiment_details
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}, {}, f"Error loading experiment data: {str(e)}"

# Callback to update architecture comparison
@app.callback(
    Output('architecture-comparison', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_architecture_comparison(_):
    # Find all experiments and group by architecture
    architectures = {}
    
    # Search for experiment directories
    experiment_dirs = []
    for results_dir in ["results", "experiments"]:  # Check both possible result directories
        if os.path.exists(results_dir):
            experiment_dirs.extend([
                os.path.join(results_dir, d)
                for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))
            ])
    
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
            parts = dir_name.split('_')
            if len(parts) >= 3:
                architecture = parts[2]  # Assuming architecture is in this position
            else:
                # Use unknown as fallback
                architecture = "unknown"
        
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
                
            if architecture not in architectures:
                architectures[architecture] = []
                
            architectures[architecture].append({
                "name": os.path.basename(exp_dir),
                "train_loss": history.get("train_loss", []),
                "val_loss": history.get("val_loss", []) if "val_loss" in history else []
            })
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
                fig.add_trace(go.Scatter(
                    y=exp["train_loss"],
                    name=f"{arch} - {exp['name']}",
                    line=dict(color=colors[color_idx % len(colors)], dash="solid")
                ))
                
                # Add validation loss if available
                if exp["val_loss"]:
                    fig.add_trace(go.Scatter(
                        y=exp["val_loss"],
                        name=f"{arch} - {exp['name']} (Val)",
                        line=dict(color=colors[color_idx % len(colors)], dash="dash")
                    ))
                    
                color_idx += 1
    
    fig.update_layout(
        title="Architecture Comparison - Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",  # Log scale for better visualization
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
    )
    
    return fig

# Callback to update PDE comparison
@app.callback(
    Output('pde-comparison', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_pde_comparison(_):
    # Find all experiments and group by PDE type
    pdes = {}
    
    # Search for experiment directories
    experiment_dirs = []
    for results_dir in ["results", "experiments"]:  # Check both possible result directories
        if os.path.exists(results_dir):
            experiment_dirs.extend([
                os.path.join(results_dir, d)
                for d in os.listdir(results_dir) 
                if os.path.isdir(os.path.join(results_dir, d))
            ])
    
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
            parts = dir_name.split('_')
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
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                        comp_time = metadata.get("training_time")
                except:
                    pass
                
            if pde_type not in pdes:
                pdes[pde_type] = []
                
            pdes[pde_type].append({
                "name": os.path.basename(exp_dir),
                "train_loss": history.get("train_loss", []),
                "val_loss": history.get("val_loss", []) if "val_loss" in history else [],
                "computation_time": comp_time
            })
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
                    name += f" ({exp['computation_time']:.2f}s)"
                    
                fig.add_trace(go.Scatter(
                    y=exp["train_loss"],
                    name=name,
                    line=dict(color=colors[color_idx % len(colors)])
                ))
                
                color_idx += 1
    
    fig.update_layout(
        title="PDE Comparison - Training Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_type="log",  # Log scale for better visualization
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.05)
    )
    
    return fig

if __name__ == '__main__':
    print("Starting PINNs-RL-PDE Training Monitor")
    print("Open http://127.0.0.1:8050/ in your browser")
    app.run_server(debug=True) 