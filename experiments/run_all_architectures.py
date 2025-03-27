import os
import torch
import yaml
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, List
import time

from src.neural_networks.neural_networks import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import (
    setup_logging,
    save_model,
    plot_architecture_comparison,
    create_interactive_report,
)


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def run_architecture(
    config: Dict, architecture: str, device: torch.device, base_dir: Path
) -> Dict:
    """Run training for a specific architecture.

    Args:
        config: Configuration dictionary
        architecture: Neural network architecture to use
        device: Computing device
        base_dir: Base directory for results

    Returns:
        Dictionary containing results
    """
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"heat_eq_{architecture}_{'rl' if config['rl']['enabled'] else 'uniform'}_a{config['pde']['parameters']['alpha']}_f{config['pde']['exact_solution']['frequency']}"
    experiment_dir = base_dir / f"{exp_name}_{timestamp}"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(experiment_dir / "training.log")
    logger.info(f"Starting training for {architecture} architecture")

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

    # Model setup
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
        architecture=architecture,
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

    logger.info(f"Training completed for {architecture} architecture")
    return {
        "architecture": architecture,
        "experiment_dir": str(experiment_dir),
        "history": history,
        "final_metrics": trainer.evaluate(),
        "training_time": training_time,
        "model": model,
    }


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

    # Create base results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path(
        f"{config['paths']['results_dir']}/architecture_comparison_{timestamp}"
    )
    base_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(base_dir / "comparison.log")
    logger.info("Starting architecture comparison experiment")

    # List of architectures to test
    architectures = ["fourier", "resnet", "siren", "autoencoder", "feedforward"]
    results = []
    metrics = {
        'training_loss': {},
        'validation_loss': {},
        'computation_time': {}
    }

    # Run each architecture
    for architecture in architectures:
        logger.info(f"\nRunning {architecture} architecture...")
        try:
            result = run_architecture(config, architecture, device, base_dir)
            results.append(result)
            
            # Update metrics for interactive report
            metrics['training_loss'][f"HeatEquation_{architecture}"] = result['history']['train_loss']
            metrics['validation_loss'][f"HeatEquation_{architecture}"] = result['history']['val_loss']
            metrics['computation_time'][f"HeatEquation_{architecture}"] = result['training_time']
            
            logger.info(f"Successfully completed {architecture} architecture")
        except Exception as e:
            logger.error(f"Error running {architecture} architecture: {str(e)}")
            continue

    # Save comparison results
    comparison_path = base_dir / "comparison_results.json"
    with open(comparison_path, "w") as f:
        json.dump(results, f, indent=4)

    # Create interactive report
    report_path = base_dir / "interactive_report.html"
    create_interactive_report(
        experiment_dir=str(base_dir),
        pdes=[HeatEquation(**config['pde']['parameters'])],
        architectures=[{'name': result['architecture'], 'model': result['model']} for result in results],
        metrics=metrics,
        config=config,
        save_path=str(report_path)
    )

    # Print summary
    print("\nArchitecture Comparison Summary:")
    print("-" * 80)
    for result in results:
        print(f"\n{result['architecture'].title()} Architecture:")
        print(f"  Experiment Directory: {result['experiment_dir']}")
        print(f"  Final L2 Error: {result['final_metrics']['l2_error']:.6f}")
        print(f"  Training Time: {result['training_time']:.2f} seconds")
        print(f"  Final Loss: {result['history']['train_loss'][-1]:.6f}")

    logger.info("Architecture comparison completed successfully")
    logger.info(f"Results saved in: {base_dir}")


if __name__ == "__main__":
    main()
