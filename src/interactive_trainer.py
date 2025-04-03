import os
import sys
import tkinter as tk
from tkinter import ttk
import torch
import yaml
import threading
from datetime import datetime
from pathlib import Path
import numpy as np
import json
import subprocess
import webbrowser
import time
import logging

# Make sure src is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.neural_networks import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.wave_equation import WaveEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils.utils import save_model
from src.pdes.pde_base import PDEConfig
from src.config import ModelConfig, Config


class InteractiveTrainer:
    def __init__(self, root):
        self.root = root
        self.root.title("PINN Training Interface")
        self.root.geometry("800x800")

        # Initial configuration
        self.setup_variables()
        self.create_ui()
        
        # Load configuration
        self.load_config("config.yaml")

    def setup_variables(self):
        """Initialize application variables"""
        # Load config to get default values
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                training_config = config.get("training", {})
                arch_config = config.get("architectures", {}).get("fourier", {})
        except Exception as e:
            print(f"Error loading config.yaml: {e}")
            training_config = {}
            arch_config = {}

        # Training parameters with defaults from config.yaml
        self.epochs = tk.IntVar(value=training_config.get("num_epochs", 100))
        self.batch_size = tk.IntVar(value=training_config.get("batch_size", 32))
        self.num_points = tk.IntVar(value=training_config.get("num_collocation_points", 10000))
        self.learning_rate = tk.DoubleVar(
            value=training_config.get("optimizer_config", {}).get("learning_rate", 0.001)
        )
        self.hidden_dim = tk.IntVar(value=arch_config.get("hidden_dims", [124])[0] if isinstance(arch_config.get("hidden_dims"), list) else 64)
        self.num_layers = tk.IntVar(value=len(arch_config.get("hidden_dims", [])) if isinstance(arch_config.get("hidden_dims"), list) else 4)

        # Load PDE-specific architectures
        try:
            pde_configs = config.get("pde_configs", {})
        except Exception as e:
            print(f"Error loading config.yaml: {e}")
            pde_configs = {}

        # Options for dropdown lists
        self.pde_types = [
            "Heat Equation",  # First in list
            "Wave Equation",
            "Burgers Equation",
            "KdV Equation",
            "Convection Equation",
            "Allen-Cahn Equation",
            "Cahn-Hilliard Equation",
            "Black-Scholes Equation",
            "Pendulum Equation",
        ]

        # Map PDE names to their config keys
        self.pde_name_to_key = {
            "Heat Equation": "heat",
            "Wave Equation": "wave",
            "Burgers Equation": "burgers",
            "KdV Equation": "kdv",
            "Convection Equation": "convection",
            "Allen-Cahn Equation": "allen_cahn",
            "Cahn-Hilliard Equation": "cahn_hilliard",
            "Black-Scholes Equation": "black_scholes",
            "Pendulum Equation": "pendulum",
        }

        # Get all unique architectures from config
        self.architectures = list(
            set(
                pde_config.get("architecture", "fourier")
                for pde_config in pde_configs.values()
            )
        )
        if not self.architectures:  # Fallback if config loading failed
            self.architectures = [
                "fourier",
                "siren",
                "resnet",
                "feedforward",
                "attention",
                "autoencoder",
            ]

        self.device_options = ["cpu", "cuda", "mps"]

        # Control variables
        self.selected_pde = tk.StringVar(
            value="Heat Equation"
        )  # Explicitly set default to Heat Equation

        # Get the default architecture for the selected PDE
        default_arch = pde_configs.get(
            self.pde_name_to_key.get("Heat Equation", "heat"), {}
        ).get("architecture", "fourier")

        self.selected_arch = tk.StringVar(value=default_arch)
        self.selected_device = tk.StringVar(value=self.device_options[0])
        self.use_rl = tk.BooleanVar(value=False)

        # PDE-specific parameters
        self.alpha = tk.DoubleVar(value=0.01)  # Heat/Wave equation
        self.frequency = tk.DoubleVar(value=2.0)  # Exact solution
        self.viscosity = tk.DoubleVar(value=0.01)  # Burgers equation
        self.velocity = tk.DoubleVar(value=1.0)  # Convection equation
        self.epsilon = tk.DoubleVar(value=0.1)  # Allen-Cahn/Cahn-Hilliard
        self.sigma = tk.DoubleVar(value=0.2)  # Black-Scholes
        self.r = tk.DoubleVar(value=0.05)  # Black-Scholes
        self.gravity = tk.DoubleVar(value=9.81)  # Pendulum
        self.length = tk.DoubleVar(value=1.0)  # Pendulum

        # Training state
        self.training_running = False
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_loss = float("inf")

    def create_ui(self):
        """Create the user interface"""
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill="both", expand=True)

        # PDE selection section
        pde_frame = ttk.LabelFrame(main_frame, text="Partial Differential Equation")
        pde_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(pde_frame, text="Equation Type:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        pde_combo = ttk.Combobox(
            pde_frame,
            textvariable=self.selected_pde,
            values=self.pde_types,
            state="readonly",
        )
        pde_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        pde_combo.bind("<<ComboboxSelected>>", self.on_pde_selected)

        # PDE-specific parameters
        self.pde_params_frame = ttk.Frame(pde_frame)
        self.pde_params_frame.grid(
            row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )
        self.update_pde_params()

        # Network architecture section
        arch_frame = ttk.LabelFrame(main_frame, text="Network Architecture")
        arch_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(arch_frame, text="Architecture:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        arch_combo = ttk.Combobox(
            arch_frame,
            textvariable=self.selected_arch,
            values=self.architectures,
            state="readonly",
        )
        arch_combo.grid(row=0, column=1, sticky="ew", padx=5, pady=5)

        ttk.Label(arch_frame, text="Hidden Dimension:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(arch_frame, textvariable=self.hidden_dim, width=10).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )

        ttk.Label(arch_frame, text="Number of Layers:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(arch_frame, textvariable=self.num_layers, width=10).grid(
            row=2, column=1, sticky="w", padx=5, pady=5
        )

        # Training section
        train_frame = ttk.LabelFrame(main_frame, text="Training Parameters")
        train_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(train_frame, text="Epochs:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(train_frame, textvariable=self.epochs, width=10).grid(
            row=0, column=1, sticky="w", padx=5, pady=5
        )

        ttk.Label(train_frame, text="Batch Size:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(train_frame, textvariable=self.batch_size, width=10).grid(
            row=1, column=1, sticky="w", padx=5, pady=5
        )

        ttk.Label(train_frame, text="Collocation Points:").grid(
            row=2, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(train_frame, textvariable=self.num_points, width=10).grid(
            row=2, column=1, sticky="w", padx=5, pady=5
        )

        ttk.Label(train_frame, text="Learning Rate:").grid(
            row=3, column=0, sticky="w", padx=5, pady=5
        )
        ttk.Entry(train_frame, textvariable=self.learning_rate, width=10).grid(
            row=3, column=1, sticky="w", padx=5, pady=5
        )

        ttk.Label(train_frame, text="Device:").grid(
            row=4, column=0, sticky="w", padx=5, pady=5
        )
        device_combo = ttk.Combobox(
            train_frame,
            textvariable=self.selected_device,
            values=self.device_options,
            state="readonly",
        )
        device_combo.grid(row=4, column=1, sticky="ew", padx=5, pady=5)

        # RL option
        rl_frame = ttk.LabelFrame(main_frame, text="Reinforcement Learning")
        rl_frame.pack(fill="x", padx=5, pady=5)

        ttk.Checkbutton(
            rl_frame, text="Use RL for adaptive sampling", variable=self.use_rl
        ).pack(padx=5, pady=5, anchor="w")

        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill="x", padx=5, pady=10)

        self.start_btn = ttk.Button(
            btn_frame, text="Start Training", command=self.start_training
        )
        self.start_btn.pack(side="left", padx=5)

        self.stop_btn = ttk.Button(
            btn_frame, text="Stop", command=self.stop_training, state="disabled"
        )
        self.stop_btn.pack(side="left", padx=5)

        self.dashboard_btn = ttk.Button(
            btn_frame, text="Open Dashboard", command=self.open_dashboard
        )
        self.dashboard_btn.pack(side="right", padx=5)

        # Progress indicators
        progress_frame = ttk.LabelFrame(main_frame, text="Progress")
        progress_frame.pack(fill="x", padx=5, pady=5)

        ttk.Label(progress_frame, text="Epoch:").grid(
            row=0, column=0, sticky="w", padx=5, pady=5
        )
        self.epoch_label = ttk.Label(progress_frame, text="0/0")
        self.epoch_label.grid(row=0, column=1, sticky="w", padx=5, pady=5)

        ttk.Label(progress_frame, text="Best Val Loss:").grid(
            row=1, column=0, sticky="w", padx=5, pady=5
        )
        self.loss_label = ttk.Label(progress_frame, text="--")
        self.loss_label.grid(row=1, column=1, sticky="w", padx=5, pady=5)

        self.progress_bar = ttk.Progressbar(
            progress_frame, orient="horizontal", length=300, mode="determinate"
        )
        self.progress_bar.grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5
        )

        # Status label at the bottom
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill="x", padx=5, pady=5)
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(fill="x", padx=5, pady=5)

        # Set default PDE and trigger update
        pde_combo.set("Heat Equation")  # Explicitly set default value
        self.on_pde_selected(None)  # Trigger the event to update parameters

    def update_pde_params(self):
        """Update PDE-specific parameters based on selected PDE"""
        # Clear current parameters
        for widget in self.pde_params_frame.winfo_children():
            widget.destroy()

        # Get the current PDE type
        pde_type = self.selected_pde.get().lower().replace(" ", "_")

        # Load config to get PDE-specific parameters
        try:
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                pde_configs = config.get("pde_configs", {})
                pde_config = pde_configs.get(pde_type, {})

                # Update architecture based on config
                if "architecture" in pde_config:
                    self.selected_arch.set(pde_config["architecture"])

                # Show parameters based on PDE type
                if pde_type == "heat_equation":
                    ttk.Label(self.pde_params_frame, text="Alpha:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.alpha, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                elif pde_type == "burgers_equation":
                    ttk.Label(self.pde_params_frame, text="Viscosity:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.viscosity, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                elif pde_type == "wave_equation":
                    ttk.Label(self.pde_params_frame, text="Wave Speed:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.alpha, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                elif pde_type == "convection_equation":
                    ttk.Label(self.pde_params_frame, text="Velocity:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.velocity, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                elif pde_type == "allen_cahn_equation":
                    ttk.Label(self.pde_params_frame, text="Epsilon:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.epsilon, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                elif pde_type == "black_scholes_equation":
                    ttk.Label(self.pde_params_frame, text="Sigma:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.sigma, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                    ttk.Label(self.pde_params_frame, text="Risk-free Rate:").grid(
                        row=1, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.r, width=10
                    ).grid(row=1, column=1, sticky="w", padx=5, pady=5)
                elif pde_type == "pendulum_equation":
                    ttk.Label(self.pde_params_frame, text="Gravity:").grid(
                        row=0, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.gravity, width=10
                    ).grid(row=0, column=1, sticky="w", padx=5, pady=5)
                    ttk.Label(self.pde_params_frame, text="Length:").grid(
                        row=1, column=0, sticky="w", padx=5, pady=5
                    )
                    ttk.Entry(
                        self.pde_params_frame, textvariable=self.length, width=10
                    ).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        except Exception as e:
            print(f"Error updating PDE parameters: {e}")

    def on_pde_selected(self, event):
        """Event handler when a PDE is selected"""
        try:
            # Load config to get PDE-specific architecture
            with open("config.yaml", "r") as f:
                config = yaml.safe_load(f)
                pde_configs = config.get("pde_configs", {})

            # Get the selected PDE's config key
            selected_pde = self.selected_pde.get()
            pde_key = self.pde_name_to_key.get(selected_pde)

            if pde_key and pde_key in pde_configs:
                # Update architecture based on PDE configuration
                pde_config = pde_configs[pde_key]
                self.selected_arch.set(pde_config.get("architecture", "fourier"))

        except Exception as e:
            print(f"Error updating architecture for selected PDE: {e}")

        # Update PDE parameters
        self.update_pde_params()

    def load_config(self, config_path="config.yaml"):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Get the default PDE type from config
            default_pde = config.get("pde_type", "heat")
            pde_configs = config.get("pde_configs", {})

            # Map config PDE names to UI names
            pde_name_map = {
                "heat": "Heat Equation",
                "wave": "Wave Equation",
                "burgers": "Burgers Equation",
                "convection": "Convection Equation",
                "kdv": "KdV Equation",
                "allen_cahn": "Allen-Cahn Equation",
                "cahn_hilliard": "Cahn-Hilliard Equation",
                "black_scholes": "Black-Scholes Equation",
                "pendulum": "Pendulum Equation",
            }

            # Set the default PDE and its architecture
            if default_pde in pde_configs:
                pde_config = pde_configs[default_pde]
                self.selected_pde.set(pde_name_map.get(default_pde, "Heat Equation"))
                self.selected_arch.set(pde_config.get("architecture", "fourier"))

                # Update PDE-specific parameters
                if "parameters" in pde_config:
                    params = pde_config["parameters"]
                    if default_pde == "heat":
                        self.alpha.set(params.get("alpha", 0.01))
                    elif default_pde == "burgers":
                        self.viscosity.set(params.get("viscosity", 0.01))
                    elif default_pde == "wave":
                        self.alpha.set(params.get("c", 1.0))
                    elif default_pde == "convection":
                        self.velocity.set(params.get("velocity", 1.0))
                    elif default_pde == "allen_cahn":
                        self.epsilon.set(params.get("epsilon", 0.1))
                    elif default_pde == "black_scholes":
                        self.sigma.set(params.get("sigma", 0.2))
                        self.r.set(params.get("r", 0.05))
                    elif default_pde == "pendulum":
                        self.gravity.set(params.get("gravity", 9.81))
                        self.length.set(params.get("length", 1.0))

            # Update UI with loaded parameters
            self.update_pde_params()

        except Exception as e:
            print(f"Error loading configuration: {e}")

    def create_config(self):
        """Create a configuration dictionary based on current values and config.yaml"""
        # Load config.yaml
        with open("config.yaml", "r") as f:
            yaml_config = yaml.safe_load(f)
            
        # Get PDE-specific configuration from config.yaml
        pde_name = self.selected_pde.get().lower().replace(" ", "_")
        pde_key = pde_name.split("_")[0]  # Get base name (e.g., 'heat' from 'heat_equation')
        pde_config = yaml_config.get(pde_key, {})  # Changed from yaml_config["pde_configs"] to direct access

        # Get architecture configuration
        arch_config = yaml_config.get("architectures", {}).get(
            self.selected_arch.get(), {}
        )

        # Use global device from config
        device = yaml_config.get("device", "cpu")

        config = {
            "device": device,  # Use global device setting
            "model": {
                "architecture": self.selected_arch.get(),
                "input_dim": pde_config.get("input_dim", 2),
                "hidden_dim": self.hidden_dim.get(),
                "output_dim": pde_config.get("output_dim", 1),
                "num_layers": self.num_layers.get(),
                "activation": arch_config.get("activation", "tanh"),
                "fourier_features": self.selected_arch.get() == "fourier",
                "fourier_scale": arch_config.get("scale", 1.0) if self.selected_arch.get() == "fourier" else None,
                "dropout": arch_config.get("dropout", 0.0),
                "layer_norm": arch_config.get("layer_norm", True),
            },
            "training": yaml_config.get("training", {}),  # Get full training config
            "rl": {"enabled": self.use_rl.get(), **yaml_config.get("rl", {})},
            "evaluation": yaml_config.get("evaluation", {}),
            "paths": yaml_config.get("paths", {}),
            "logging": yaml_config.get("logging", {}),
        }

        # Only override specific training parameters that can be set in UI
        config["training"].update({
            "num_epochs": self.epochs.get(),
            "batch_size": self.batch_size.get(),
            "num_collocation_points": self.num_points.get(),
            "optimizer_config": {
                **config["training"].get("optimizer_config", {}),
                "learning_rate": self.learning_rate.get(),
            }
        })

        # Add PDE-specific configuration
        config["pde"] = {
            "name": pde_config.get("name", self.selected_pde.get()),
            "parameters": pde_config.get("parameters", {}),
            "domain": pde_config.get("domain", [[0, 1]]),
            "time_domain": pde_config.get("time_domain", [0, 1]),
            "boundary_conditions": pde_config.get("boundary_conditions", {}),
            "initial_condition": pde_config.get("initial_condition", {}),
            "exact_solution": pde_config.get("exact_solution", {}),
            "dimension": pde_config.get("dimension", 1),
        }

        return config

    def create_pde(self, config, device):
        """Create a PDE instance based on configuration"""
        pde_type = self.selected_pde.get()
        pde_config = PDEConfig(
            name=config["pde"]["name"],
            domain=config["pde"]["domain"],
            time_domain=config["pde"]["time_domain"],
            parameters=config["pde"]["parameters"],
            boundary_conditions=config["pde"]["boundary_conditions"],
            initial_condition=config["pde"]["initial_condition"],
            exact_solution=config["pde"]["exact_solution"],
        )

        if pde_type == "Heat Equation":
            return HeatEquation(config=pde_config, device=device)
        elif pde_type == "Burgers Equation":
            return BurgersEquation(config=pde_config, device=device)
        elif pde_type == "Wave Equation":
            return WaveEquation(config=pde_config, device=device)
        else:
            raise ValueError(f"Unsupported PDE: {pde_type}")

    def start_training(self):
        """Start training in a separate thread"""
        if self.training_running:
            return

        # Update UI
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.training_running = True

        # Create a training instance in a separate thread
        self.training_thread = threading.Thread(target=self.run_training)
        self.training_thread.daemon = True
        self.training_thread.start()

        # Start progress updates
        self.root.after(1000, self.update_progress)

    def run_training(self):
        """Run the training process in a separate thread."""
        try:
            # Create configuration
            config = self.create_config()

            # Set device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Create experiment directory with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_dir = Path(config["paths"]["results_dir"]) / timestamp
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Create subdirectories
            (experiment_dir / "visualizations").mkdir(exist_ok=True)
            (experiment_dir / "models").mkdir(exist_ok=True)
            (experiment_dir / "logs").mkdir(exist_ok=True)

            # Save configuration
            with open(experiment_dir / "config.yaml", "w") as f:
                yaml.dump(config, f)

            # Create PDE
            pde = self.create_pde(config, device)

            # Create model config
            model_config = ModelConfig(
                input_dim=config["model"]["input_dim"],
                hidden_dim=config["model"]["hidden_dim"],
                output_dim=config["model"]["output_dim"],
                num_layers=config["model"]["num_layers"],
                activation=config["model"]["activation"],
                fourier_features=config["model"]["fourier_features"],
                fourier_scale=config["model"]["fourier_scale"],
                dropout=config["model"]["dropout"],
                layer_norm=config["model"]["layer_norm"],
                architecture=config["model"]["architecture"],
            )

            # Create config object with model attribute
            config_obj = Config()
            config_obj.model = model_config

            # Initialize model
            model = PINNModel(config=config_obj, device=device).to(device)

            # Initialize trainer
            trainer = PDETrainer(
                model=model,
                pde=pde,
                optimizer_config=config["training"]["optimizer_config"],
                device=device,
                rl_agent=(
                    None if not config["rl"]["enabled"] else None
                ),  # TODO: Implement RL agent
                validation_frequency=config["training"]["validation_frequency"],
                early_stopping_config=config["training"]["early_stopping"],
            )

            # Save trainer and configuration in attributes for access from update_progress
            self.current_trainer = trainer
            self.total_epochs = config["training"]["num_epochs"]
            self.experiment_dir = experiment_dir

            # Run training
            history = trainer.train(
                num_epochs=config["training"]["num_epochs"],
                batch_size=config["training"]["batch_size"],
                num_points=config["training"]["num_collocation_points"],
                experiment_dir=str(experiment_dir),
            )

            # Save the model
            model_path = experiment_dir / "models" / "final_model.pt"
            save_model(model, str(model_path))

            # Open the experiment directory
            if sys.platform == "darwin":  # macOS
                subprocess.run(["open", str(experiment_dir)])
            elif sys.platform == "win32":  # Windows
                os.startfile(str(experiment_dir))
            else:  # Linux
                subprocess.run(["xdg-open", str(experiment_dir)])

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.training_running = False
            self.root.after(0, self.on_training_complete)

    def update_progress(self):
        """Update training progress in the UI"""
        if not self.training_running:
            return

        try:
            # Get current values from trainer
            if hasattr(self, "current_trainer"):
                # Update progress values
                current_epoch = len(self.current_trainer.history.get("train_loss", []))
                self.current_epoch = current_epoch

                # Update best loss
                if (
                    "val_loss" in self.current_trainer.history
                    and self.current_trainer.history["val_loss"]
                ):
                    self.best_loss = min(self.current_trainer.history["val_loss"])

                # Check if early stopping was triggered
                early_stopped = (
                    hasattr(self.current_trainer, "patience_counter")
                    and self.current_trainer.patience_counter
                    >= self.current_trainer.patience
                )

                # Update UI
                if early_stopped:
                    # Set progress to 100% if early stopping was triggered
                    self.epoch_label.config(
                        text=f"{current_epoch}/{self.total_epochs} (Early Stop)"
                    )
                    self.progress_bar["value"] = 100
                    # Update status label
                    self.status_label.config(
                        text=f"Early stopping at epoch {current_epoch}/{self.total_epochs}"
                    )
                else:
                    self.epoch_label.config(text=f"{current_epoch}/{self.total_epochs}")
                    # Update progress bar as a percentage
                    progress = (current_epoch / self.total_epochs) * 100
                    self.progress_bar["value"] = progress

                self.loss_label.config(text=f"{self.best_loss:.6f}")
        except Exception as e:
            print(f"Error updating progress: {e}")

        # Schedule next update
        if self.training_running:
            self.root.after(1000, self.update_progress)

    def on_training_complete(self):
        """Method called when training completes"""
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

        # Set progress bar to 100% to indicate completion
        self.progress_bar["value"] = 100

        # Update epoch display with final values
        if hasattr(self, "current_trainer"):
            current_epoch = len(self.current_trainer.history.get("train_loss", []))
            total_epochs = self.total_epochs

            # Check if early stopping was triggered
            early_stopped = (
                hasattr(self.current_trainer, "patience_counter")
                and self.current_trainer.patience_counter
                >= self.current_trainer.patience
            )

            if early_stopped:
                self.epoch_label.config(
                    text=f"{current_epoch}/{total_epochs} (Early Stop)"
                )
            else:
                self.epoch_label.config(
                    text=f"{current_epoch}/{total_epochs} (Completed)"
                )

            # Update best loss display
            if (
                "val_loss" in self.current_trainer.history
                and self.current_trainer.history["val_loss"]
            ):
                self.best_loss = min(self.current_trainer.history["val_loss"])
                self.loss_label.config(text=f"{self.best_loss:.6f}")

        # Show final message
        if hasattr(self, "experiment_dir"):
            msg = f"Training completed. Results saved in: {self.experiment_dir}"
            print(msg)
            self.status_label.config(text=msg)

    def stop_training(self):
        """Stop ongoing training"""
        self.training_running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def open_dashboard(self):
        """Opens the training monitor dashboard in a web browser"""
        try:
            self.status_label.config(text="Starting dashboard...")

            # Kill any existing dashboard process first
            try:
                subprocess.run("pkill -f 'python dashboard.py' || true", shell=True)
            except Exception:
                pass

            # Run the dashboard on port 8051 (different from the default 8050)
            # to avoid conflict with the GUI
            dashboard_cmd = f"python dashboard.py --port 8051"

            # Set environment variable to ensure PYTHONPATH is correct
            env = os.environ.copy()
            if "PYTHONPATH" not in env:
                env["PYTHONPATH"] = os.getcwd()

            # Start the dashboard process
            proc = subprocess.Popen(
                    dashboard_cmd,
                shell=True,
                cwd=os.getcwd(),
                env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
            )

            # Give dashboard time to start
            time.sleep(2)

            # Check if the process is still running
            if proc.poll() is None:
                # Process is running, open browser
                webbrowser.open("http://127.0.0.1:8051/")
                self.status_label.config(text="Dashboard opened in browser (port 8051)")
            else:
                # Process failed, get error message
                stderr = proc.stderr.read().decode("utf-8")
                if "ModuleNotFoundError: No module named 'dash'" in stderr:
                    self.status_label.config(
                        text="Error: Package 'dash' not found. Install with 'pip install dash'"
                    )
                else:
                    self.status_label.config(text=f"Error starting dashboard: {stderr}")
        except Exception as e:
            self.status_label.config(text=f"Error opening dashboard: {e}")
            print(f"Error opening dashboard: {e}")


def main():
    """Entry point for the interactive trainer"""
    root = tk.Tk()
    app = InteractiveTrainer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
