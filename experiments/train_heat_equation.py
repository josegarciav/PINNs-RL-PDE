
import os
import torch
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import setup_logging, save_model, load_model, plot_solution

def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    # Load configuration
    config = load_config()
    
    # Setup device
    device_name = config.get('device', 'cpu')
    if device_name == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif device_name == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory with descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"heat_eq_{'rl' if config['rl']['enabled'] else 'uniform'}_a{config['pde']['parameters']['alpha']}_f{config['pde']['exact_solution']['frequency']}"
    experiment_dir = Path(f"{config['paths']['results_dir']}/{exp_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir / "training.log")
    logger.info("Starting Heat Equation training experiment")
    logger.info(f"Configuration: {config}")
    
    # PDE setup
    pde = HeatEquation(
        alpha=config['pde']['parameters']['alpha'],
        domain=config['pde']['domain'],
        time_domain=config['pde']['time_domain'],
        boundary_conditions=config['pde']['boundary_conditions'],
        initial_condition=config['pde']['initial_condition'],
        exact_solution=config['pde']['exact_solution'],
        device=device
    )
    
    # Model setup
    model = PINNModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        output_dim=config['model']['output_dim'],
        num_layers=config['model']['num_layers'],
        activation=config['model']['activation'],
        fourier_features=config['model']['fourier_features'],
        fourier_scale=config['model']['fourier_scale'],
        dropout=config['model']['dropout'],
        layer_norm=config['model']['layer_norm'],
        device=device
    ).to(device)
    
    # RL Agent setup (if enabled)
    rl_agent = None
    if config['rl']['enabled']:
        rl_agent = RLAgent(
            state_dim=config['rl']['state_dim'],
            action_dim=config['rl']['action_dim'],
            hidden_dim=config['rl']['hidden_dim'],
            learning_rate=config['rl']['learning_rate'],
            gamma=config['rl']['gamma'],
            epsilon_start=config['rl']['epsilon_start'],
            epsilon_end=config['rl']['epsilon_end'],
            epsilon_decay=config['rl']['epsilon_decay'],
            memory_size=config['rl']['memory_size'],
            batch_size=config['rl']['batch_size'],
            target_update=config['rl']['target_update'],
            reward_weights=config['rl']['reward_weights'],
            device=device
        )
    
    # Print model summary
    logger.info(f"Model architecture:\n{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    if rl_agent:
        logger.info(f"RL Agent architecture:\n{rl_agent.policy_net}")
        logger.info(f"RL Agent parameters: {sum(p.numel() for p in rl_agent.policy_net.parameters())}")
        logger.info(f"RL Agent configuration:")
        logger.info(f"  - State dim: {rl_agent.state_dim}")
        logger.info(f"  - Action dim: {rl_agent.action_dim}")
        logger.info(f"  - Hidden dim: {rl_agent.hidden_dim}")
        logger.info(f"  - Learning rate: {rl_agent.learning_rate}")
        logger.info(f"  - Gamma: {rl_agent.gamma}")
        logger.info(f"  - Epsilon: {rl_agent.epsilon}")
        logger.info(f"  - Memory size: {rl_agent.memory_size}")
        logger.info(f"  - Batch size: {rl_agent.batch_size}")
        logger.info(f"  - Target update: {rl_agent.target_update}")
        logger.info(f"  - Reward weights: {rl_agent.reward_weights}")
    
    # Training setup
    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config=config['training']['optimizer'],
        device=device,
        checkpoint_dir=experiment_dir / config['paths']['checkpoint_dir']
    )
    
    try:
        # Start training
        trainer.train(
            num_epochs=config['training']['num_epochs'],
            batch_size=config['training']['batch_size'],
            num_points=config['training']['num_collocation_points'],
            validation_frequency=config['training']['validation_frequency'],
            save_frequency=config['logging']['checkpoint_frequency']
        )
        
        # Final evaluation
        logger.info("Training completed. Performing final evaluation...")
        final_metrics = pde.validate(model, num_points=config['evaluation']['num_points'])
        logger.info(f"Final metrics: {final_metrics}")
        
        # Save final model, RL agent, and metrics
        save_model(model, experiment_dir / "final_model.pth")
        if rl_agent:
            save_model(rl_agent, experiment_dir / "final_rl_agent.pth")
        np.save(experiment_dir / "final_metrics.npy", final_metrics)
        
        # Plot final solution
        plot_solution(
            model=model,
            pde=pde,
            num_points=config['evaluation']['num_points'],
            save_path=experiment_dir / "final_solution.html",
            use_rl=config['rl']['enabled'],
            rl_agent=rl_agent
        )
        
        logger.info("Experiment completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
