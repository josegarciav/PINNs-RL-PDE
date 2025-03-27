import os
import torch
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time

from src.neural_networks.neural_networks import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import (
    setup_logging,
    save_model,
    load_model,
    plot_solution,
    plot_architecture_comparison,
    create_interactive_report,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    # Load configuration
    config = load_config()

    # Setup device
    device_name = config.get("device", "cpu")
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create experiment directory with descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"heat_eq_{config['model']['architecture']}_{'rl' if config['rl']['enabled'] else 'uniform'}_a{config['pde']['parameters']['alpha']}_f{config['pde']['exact_solution']['frequency']}"
    experiment_dir = Path(f"{config['paths']['results_dir']}/{exp_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(experiment_dir / "training.log")
    logger.info("Starting Heat Equation training experiment")
    logger.info(f"Configuration: {config}")

    # PDE setup
    pde = HeatEquation(
        alpha=config["pde"]["parameters"]["alpha"],
        domain=config["pde"]["domain"],
        time_domain=config["pde"]["time_domain"],
        boundary_conditions=config["pde"]["boundary_conditions"],
        initial_condition=config["pde"]["initial_condition"],
        exact_solution=config["pde"]["exact_solution"],
        device=device,
    )

    # Model setup with architecture selection
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

    # RL Agent setup (if enabled)
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
        
        # Update PDE with RL agent for adaptive sampling
        pde.rl_agent = rl_agent

    # Trainer setup
    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config=config["training"]["optimizer_config"],
        device=device,
        rl_agent=rl_agent,
    )

    # Training loop with timing
    logger.info("Starting training...")
    start_time = time.time()
    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        batch_size=config["training"]["batch_size"],
        num_points=config["training"]["num_collocation_points"],
        validation_frequency=config["training"]["validation_frequency"],
        experiment_dir=str(experiment_dir)
    )
    training_time = time.time() - start_time

    # Save model and results
    model_path = experiment_dir / "model.pth"
    save_model(model, str(model_path), config)

    # Plot architecture comparison
    comparison_path = experiment_dir / "architecture_comparison.html"
    plot_architecture_comparison(
        model=model,
        pde=pde,
        num_points=config["evaluation"]["num_points"],
        save_path=str(comparison_path),
    )

    # Save training history
    history_path = experiment_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    # Prepare metrics for interactive report
    metrics = {
        'training_loss': history['train_loss'],
        'validation_loss': history['val_loss'],
        'computation_time': {
            f"{pde.__class__.__name__}_{config['model']['architecture']}": training_time
        }
    }

    # Create interactive report
    report_path = experiment_dir / "interactive_report.html"
    create_interactive_report(
        experiment_dir=str(experiment_dir),
        pdes=[pde],  # For now just the heat equation
        architectures=[{
            'name': config['model']['architecture'],
            'model': model
        }],
        metrics=metrics,
        config=config,
        save_path=str(report_path)
    )

    logger.info("Training completed successfully")
    logger.info(f"Results saved in: {experiment_dir}")
    logger.info(f"Training time: {training_time:.2f} seconds")


if __name__ == "__main__":
    main()
