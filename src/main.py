import torch
import os
import json
from datetime import datetime
from typing import Dict, Optional, Tuple

from src.config import Config
from src.neural_networks.neural_networks import PINNModel
from src.pde_base import PDEBase
from src.rl_agent import RLAgent
from src.trainer import PDETrainer
from src.utils import (
    setup_logging,
    save_model,
    load_model,
    plot_pinn_solution,
    plot_pinn_3d_solution,
    plot_training_history,
)


def setup_experiment(config: Config) -> Tuple[str, str]:
    """
    Setup experiment directories and logging.

    :param config: Configuration object
    :return: Tuple of (experiment directory, log directory)
    """
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(config.paths.experiments_dir, f"pinn_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)

    # Create model and log directories
    model_dir = os.path.join(exp_dir, config.paths.model_dir)
    log_dir = os.path.join(exp_dir, config.paths.log_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(exp_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=4)

    # Setup logging
    logger = setup_logging(log_dir)
    logger.info(f"Starting experiment in directory: {exp_dir}")

    return exp_dir, log_dir


def create_model(config: Config) -> PINNModel:
    """
    Create and initialize the PINN model.

    :param config: Configuration object
    :return: Initialized PINN model
    """
    model = PINNModel(
        input_dim=config.model.input_dim,
        hidden_dim=config.model.hidden_dim,
        output_dim=config.model.output_dim,
        num_layers=config.model.num_layers,
        activation=config.model.activation,
        fourier_features=config.model.fourier_features,
        fourier_scale=config.model.fourier_scale,
        dropout=config.model.dropout,
        layer_norm=config.model.layer_norm,
        device=config.device,
    )
    return model


def create_rl_agent(config: Config) -> Optional[RLAgent]:
    """
    Create the RL agent if enabled in config.

    :param config: Configuration object
    :return: Optional RL agent instance
    """
    if config.rl.enabled:
        return RLAgent(
            state_dim=config.rl.state_dim,
            action_dim=config.rl.action_dim,
            hidden_dim=config.rl.hidden_dim,
            learning_rate=config.rl.learning_rate,
            gamma=config.rl.gamma,
            epsilon_start=config.rl.epsilon_start,
            epsilon_end=config.rl.epsilon_end,
            epsilon_decay=config.rl.epsilon_decay,
            memory_size=config.rl.memory_size,
            batch_size=config.rl.batch_size,
            target_update=config.rl.target_update,
            reward_weights=config.rl.reward_weights,
            device=config.device,
        )
    return None


def train_model(
    model: PINNModel,
    pde: PDEBase,
    rl_agent: Optional[RLAgent],
    config: Config,
    exp_dir: str,
) -> Dict:
    """
    Train the PINN model with the specified configuration.

    :param model: PINN model instance
    :param pde: PDE instance
    :param rl_agent: Optional RL agent
    :param config: Configuration object
    :param exp_dir: Experiment directory
    :return: Training history
    """
    # Create trainer
    trainer = PDETrainer(
        pde=pde,
        pinn=model,
        rl_agent=rl_agent,
        config=config,
    )

    # Train model with experiment directory for real-time monitoring
    history = trainer.train(
        num_epochs=config.training.num_epochs,
        batch_size=config.training.batch_size,
        num_points=config.training.num_collocation_points,
        validation_frequency=config.training.validation_frequency,
        experiment_dir=exp_dir,
    )

    # Save final model
    model_path = os.path.join(exp_dir, config.paths.model_dir, "final_model.pth")
    save_model(model, model_path, config.to_dict())

    # Plot training history
    history_path = os.path.join(exp_dir, "training_history.png")
    plot_training_history(history, save_path=history_path)

    return history


def evaluate_model(model: PINNModel, pde: PDEBase, exp_dir: str, config: Config):
    """
    Evaluate the trained model and generate visualizations.

    :param model: Trained PINN model
    :param pde: PDE instance
    :param exp_dir: Experiment directory
    :param config: Configuration object
    """
    # Generate 2D plots
    plot_path = os.path.join(exp_dir, "solution_2d.png")
    plot_pinn_solution(
        model=model,
        pde=pde,
        domain=pde.domain,
        resolution=config.evaluation.resolution,
        device=config.device,
        save_path=plot_path,
    )

    # Generate 3D plots
    plot_3d_path = os.path.join(exp_dir, "solution_3d.png")
    plot_pinn_3d_solution(
        model=model,
        pde=pde,
        domain=pde.domain,
        t_domain=pde.t_domain,
        resolution=config.evaluation.resolution,
        device=config.device,
        save_path=plot_3d_path,
    )


def main():
    """Main training script."""
    # Load configuration
    config = Config()

    # Setup experiment
    exp_dir, log_dir = setup_experiment(config)

    # Create model and RL agent
    model = create_model(config)
    rl_agent = create_rl_agent(config)

    # Create PDE instance
    pde = PDEBase(
        domain=config.pde.domain,
        t_domain=config.pde.t_domain,
        initial_condition=config.pde.initial_condition,
        boundary_conditions=config.pde.boundary_conditions,
        diffusion_coefficient=config.pde.diffusion_coefficient,
        source_term=config.pde.source_term,
        device=config.device,
    )

    # Add RL agent to PDE for adaptive sampling if available
    if rl_agent:
        pde.rl_agent = rl_agent

    # Train model
    history = train_model(model, pde, rl_agent, config, exp_dir)

    # Evaluate model
    evaluate_model(model, pde, exp_dir, config)

    print(f"✅ Experiment completed successfully. Results saved in: {exp_dir}")


if __name__ == "__main__":
    main()
