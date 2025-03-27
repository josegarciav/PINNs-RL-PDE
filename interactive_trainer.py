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

# Make sure src is in the PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.neural_networks.neural_networks import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.wave_equation import WaveEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils.utils import setup_logging, save_model


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
        # Options for dropdown lists
        self.pde_types = ["Heat Equation", "Burgers Equation", "Wave Equation"]
        self.architectures = ["standard", "fourier", "residual", "attention"]
        self.device_options = ["cpu", "cuda", "mps"]

        # Control variables
        self.selected_pde = tk.StringVar(value=self.pde_types[0])
        self.selected_arch = tk.StringVar(value=self.architectures[0])
        self.selected_device = tk.StringVar(value=self.device_options[0])
        self.use_rl = tk.BooleanVar(value=False)

        # Training parameters
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=32)
        self.num_points = tk.IntVar(value=10000)
        self.learning_rate = tk.DoubleVar(value=0.001)
        self.hidden_dim = tk.IntVar(value=64)
        self.num_layers = tk.IntVar(value=4)

        # PDE-specific parameters
        self.alpha = tk.DoubleVar(value=0.01)  # Heat/Wave equation
        self.frequency = tk.DoubleVar(value=2.0)  # Exact solution
        self.viscosity = tk.DoubleVar(value=0.01)  # Burgers equation

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

        ttk.Label(progress_frame, text="Best Loss:").grid(
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

    def update_pde_params(self):
        """Update specific parameters based on selected PDE"""
        # Clear current frame
        for widget in self.pde_params_frame.winfo_children():
            widget.destroy()

        # Show parameters based on selected PDE
        if self.selected_pde.get() == "Heat Equation":
            ttk.Label(self.pde_params_frame, text="Alpha (Diffusivity):").grid(
                row=0, column=0, sticky="w", padx=5, pady=5
            )
            ttk.Entry(self.pde_params_frame, textvariable=self.alpha, width=10).grid(
                row=0, column=1, sticky="w", padx=5, pady=5
            )
            ttk.Label(self.pde_params_frame, text="Frequency:").grid(
                row=1, column=0, sticky="w", padx=5, pady=5
            )
            ttk.Entry(
                self.pde_params_frame, textvariable=self.frequency, width=10
            ).grid(row=1, column=1, sticky="w", padx=5, pady=5)

        elif self.selected_pde.get() == "Burgers Equation":
            ttk.Label(self.pde_params_frame, text="Viscosity:").grid(
                row=0, column=0, sticky="w", padx=5, pady=5
            )
            ttk.Entry(
                self.pde_params_frame, textvariable=self.viscosity, width=10
            ).grid(row=0, column=1, sticky="w", padx=5, pady=5)

        elif self.selected_pde.get() == "Wave Equation":
            ttk.Label(self.pde_params_frame, text="Wave Speed:").grid(
                row=0, column=0, sticky="w", padx=5, pady=5
            )
            ttk.Entry(self.pde_params_frame, textvariable=self.alpha, width=10).grid(
                row=0, column=1, sticky="w", padx=5, pady=5
            )
            ttk.Label(self.pde_params_frame, text="Frequency:").grid(
                row=1, column=0, sticky="w", padx=5, pady=5
            )
            ttk.Entry(
                self.pde_params_frame, textvariable=self.frequency, width=10
            ).grid(row=1, column=1, sticky="w", padx=5, pady=5)

    def on_pde_selected(self, event):
        """Event handler when a PDE is selected"""
        self.update_pde_params()

    def load_config(self, config_path="config.yaml"):
        """Load configuration from YAML file"""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)

            # Update variables with config values
            if "model" in config:
                self.selected_arch.set(config["model"].get("architecture", "standard"))
                self.hidden_dim.set(config["model"].get("hidden_dim", 64))
                self.num_layers.set(config["model"].get("num_layers", 4))

            if "training" in config:
                self.epochs.set(config["training"].get("num_epochs", 100))
                self.batch_size.set(config["training"].get("batch_size", 32))
                self.num_points.set(
                    config["training"].get("num_collocation_points", 10000)
                )
                if "optimizer_config" in config["training"]:
                    self.learning_rate.set(
                        config["training"]["optimizer_config"].get(
                            "learning_rate", 0.001
                        )
                    )

            if "pde" in config and "parameters" in config["pde"]:
                pde_type = config["pde"].get("name", "Heat Equation")
                self.selected_pde.set(pde_type)

                if pde_type == "Heat Equation":
                    self.alpha.set(config["pde"]["parameters"].get("alpha", 0.01))
                elif pde_type == "Burgers Equation":
                    self.viscosity.set(
                        config["pde"]["parameters"].get("viscosity", 0.01)
                    )
                elif pde_type == "Wave Equation":
                    self.alpha.set(config["pde"]["parameters"].get("wave_speed", 1.0))

                if "exact_solution" in config["pde"]:
                    self.frequency.set(
                        config["pde"]["exact_solution"].get("frequency", 2.0)
                    )

            if "rl" in config:
                self.use_rl.set(config["rl"].get("enabled", False))

            if "device" in config:
                self.selected_device.set(config.get("device", "cpu"))

            # Update UI
            self.update_pde_params()

        except Exception as e:
            print(f"Error loading configuration: {e}")

    def create_config(self):
        """Create a configuration dictionary based on current values"""
        config = {
            "device": self.selected_device.get(),
            "model": {
                "architecture": self.selected_arch.get(),
                "input_dim": 2,  # For 1D PDE + time
                "hidden_dim": self.hidden_dim.get(),
                "output_dim": 1,
                "num_layers": self.num_layers.get(),
                "activation": "tanh",
                "fourier_features": self.selected_arch.get() == "fourier",
                "fourier_scale": 10.0,
                "dropout": 0.0,
                "layer_norm": True,
            },
            "training": {
                "num_epochs": self.epochs.get(),
                "batch_size": self.batch_size.get(),
                "num_collocation_points": self.num_points.get(),
                "validation_frequency": 10,
                "optimizer_config": {
                    "learning_rate": self.learning_rate.get(),
                    "weight_decay": 1e-6,
                    "patience": 10,
                },
            },
            "rl": {
                "enabled": self.use_rl.get(),
                "state_dim": 2,
                "action_dim": 1,
                "hidden_dim": 64,
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon_start": 1.0,
                "epsilon_end": 0.01,
                "epsilon_decay": 0.995,
                "memory_size": 10000,
                "batch_size": 64,
                "target_update": 100,
                "reward_weights": {
                    "residual": 1.0,
                    "boundary": 1.0,
                    "initial": 1.0,
                    "exploration": 0.1,
                },
            },
            "evaluation": {"num_points": 1000},
            "paths": {"results_dir": "results", "model_dir": "models"},
        }

        # PDE-specific configuration
        if self.selected_pde.get() == "Heat Equation":
            config["pde"] = {
                "name": "Heat Equation",
                "parameters": {"alpha": self.alpha.get()},
                "domain": [0, 1],
                "time_domain": [0, 1],
                "boundary_conditions": {
                    "dirichlet": {"type": "zero", "points": [0, 1]}
                },
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": self.frequency.get(),
                },
                "exact_solution": {"amplitude": 1.0, "frequency": self.frequency.get()},
            }
        elif self.selected_pde.get() == "Burgers Equation":
            config["pde"] = {
                "name": "Burgers Equation",
                "parameters": {"viscosity": self.viscosity.get()},
                "domain": [-1, 1],
                "time_domain": [0, 1],
                "boundary_conditions": {
                    "dirichlet": {"type": "zero", "points": [-1, 1]}
                },
                "initial_condition": {
                    "type": "sine",
                    "amplitude": -1.0,
                    "frequency": 1.0,
                },
                "exact_solution": {},  # No simple exact solution for Burgers
            }
        elif self.selected_pde.get() == "Wave Equation":
            config["pde"] = {
                "name": "Wave Equation",
                "parameters": {"wave_speed": self.alpha.get()},
                "domain": [0, 1],
                "time_domain": [0, 2],
                "boundary_conditions": {
                    "dirichlet": {"type": "zero", "points": [0, 1]}
                },
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": self.frequency.get(),
                },
                "exact_solution": {"amplitude": 1.0, "frequency": self.frequency.get()},
            }

        return config

    def create_pde(self, config, device):
        """Create a PDE instance based on configuration"""
        pde_type = self.selected_pde.get()

        if pde_type == "Heat Equation":
            return HeatEquation(
                alpha=config["pde"]["parameters"]["alpha"],
                domain=config["pde"]["domain"],
                time_domain=config["pde"]["time_domain"],
                boundary_conditions=config["pde"]["boundary_conditions"],
                initial_condition=config["pde"]["initial_condition"],
                exact_solution=config["pde"]["exact_solution"],
                device=device,
            )
        elif pde_type == "Burgers Equation":
            return BurgersEquation(
                viscosity=config["pde"]["parameters"]["viscosity"],
                domain=config["pde"]["domain"],
                time_domain=config["pde"]["time_domain"],
                boundary_conditions=config["pde"]["boundary_conditions"],
                initial_condition=config["pde"]["initial_condition"],
                device=device,
            )
        elif pde_type == "Wave Equation":
            return WaveEquation(
                wave_speed=config["pde"]["parameters"]["wave_speed"],
                domain=config["pde"]["domain"],
                time_domain=config["pde"]["time_domain"],
                boundary_conditions=config["pde"]["boundary_conditions"],
                initial_condition=config["pde"]["initial_condition"],
                exact_solution=config["pde"]["exact_solution"],
                device=device,
            )
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
        """Main function to run training"""
        try:
            # Create configuration
            config = self.create_config()

            # Configure device
            device_name = self.selected_device.get()
            if device_name == "mps" and torch.backends.mps.is_available():
                device = torch.device("mps")
            elif device_name == "cuda" and torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")

            # Create experiment directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pde_name = self.selected_pde.get().lower().replace(" ", "_")
            exp_name = f"{pde_name}_{config['model']['architecture']}_{'rl' if self.use_rl.get() else 'uniform'}_{timestamp}"
            experiment_dir = Path(f"{config['paths']['results_dir']}/{exp_name}")
            experiment_dir.mkdir(parents=True, exist_ok=True)

            # Save configuration
            with open(experiment_dir / "config.json", "w") as f:
                json.dump(config, f, indent=2)

            # Create PDE
            pde = self.create_pde(config, device)

            # Create model
            model = PINNModel(
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
                device=device,
            ).to(device)

            # Create RL agent if needed
            rl_agent = None
            if config["rl"]["enabled"]:
                rl_agent = RLAgent(
                    state_dim=config["rl"]["state_dim"],
                    action_dim=config["rl"]["action_dim"],
                    hidden_dim=config["rl"]["hidden_dim"],
                    learning_rate=config["rl"]["learning_rate"],
                    gamma=config["rl"]["gamma"],
                    epsilon_start=config["rl"]["epsilon_start"],
                    epsilon_end=config["rl"]["epsilon_end"],
                    epsilon_decay=config["rl"]["epsilon_decay"],
                    memory_size=config["rl"]["memory_size"],
                    batch_size=config["rl"]["batch_size"],
                    target_update=config["rl"]["target_update"],
                    reward_weights=config["rl"]["reward_weights"],
                    device=device,
                )

                # Update PDE with RL agent
                pde.rl_agent = rl_agent

            # Create trainer
            trainer = PDETrainer(
                model=model,
                pde=pde,
                optimizer_config=config["training"]["optimizer_config"],
                device=device,
                rl_agent=rl_agent,
            )

            # Save trainer and configuration in attributes for access from update_progress
            self.current_trainer = trainer
            self.total_epochs = config["training"]["num_epochs"]
            self.experiment_dir = experiment_dir

            # Start training
            history = trainer.train(
                num_epochs=config["training"]["num_epochs"],
                batch_size=config["training"]["batch_size"],
                num_points=config["training"]["num_collocation_points"],
                validation_frequency=config["training"]["validation_frequency"],
                experiment_dir=str(experiment_dir),
            )

            # Save model and results
            model_path = experiment_dir / "final_model.pth"
            save_model(model, str(model_path), config)

            # Update UI when training finishes
            self.root.after(0, self.on_training_finished)

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback

            traceback.print_exc()
            # Update UI in case of error
            self.root.after(0, self.on_training_finished)

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

    def on_training_finished(self):
        """Method called when training finishes"""
        self.training_running = False
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


if __name__ == "__main__":
    root = tk.Tk()
    app = InteractiveTrainer(root)
    root.mainloop()
