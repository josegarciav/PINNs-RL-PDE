import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import setup_logging, save_model, load_model, plot_solution

def create_exact_solution(alpha, freq=2.0):
    """Create the exact solution function for the heat equation."""
    def exact_solution(x, t):
        return torch.exp(-freq**2 * np.pi**2 * alpha * t) * torch.sin(freq * np.pi * x)
    return exact_solution

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train PINN for Heat Equation')
    parser.add_argument('--use_rl', action='store_true', help='Use RL agent for adaptive sampling')
    parser.add_argument('--num_epochs', type=int, default=1000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for training')
    parser.add_argument('--num_points', type=int, default=10000, help='Number of collocation points')
    parser.add_argument('--alpha', type=float, default=0.01, help='Thermal diffusivity')
    parser.add_argument('--freq', type=float, default=2.0, help='Frequency for initial condition')
    parser.add_argument('--amp', type=float, default=1.0, help='Amplitude for initial condition')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension for PINN')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers for PINN')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Create experiment directory with descriptive name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"heat_eq_{'rl' if args.use_rl else 'uniform'}_a{args.alpha}_f{args.freq}"
    experiment_dir = Path(f"results/{exp_name}_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir / "training.log")
    logger.info("Starting Heat Equation training experiment")
    logger.info(f"Configuration: {vars(args)}")
    
    # Create exact solution function
    exact_solution = create_exact_solution(args.alpha, args.freq)
    
    # PDE setup
    pde = HeatEquation(
        alpha=args.alpha,
        domain=(0, 1),
        time_domain=(0, 1),
        boundary_conditions={
            'left': {'type': 'dirichlet', 'value': 0.0},
            'right': {'type': 'dirichlet', 'value': 0.0}
        },
        initial_condition={'type': 'sine', 'amplitude': args.amp, 'frequency': args.freq},
        exact_solution=exact_solution,
        device=device
    )
    
    # Model setup with configurable architecture
    model = PINNModel(
        input_dim=2,
        hidden_dim=args.hidden_dim,
        output_dim=1,
        num_layers=args.num_layers,
        activation='tanh',
        fourier_features=True,
        fourier_scale=10.0,
        dropout=0.1,
        layer_norm=True
    ).to(device)
    
    # RL Agent setup (if requested)
    rl_agent = None
    if args.use_rl:
        rl_agent = RLAgent(
            state_dim=2,
            action_dim=1,
            hidden_dim=args.hidden_dim,
            learning_rate=args.lr,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            reward_weights={
                'residual': 1.0,
                'boundary': 1.0,
                'initial': 1.0,
                'exploration': 0.1
            },
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
    
    # Training setup with configurable optimizer
    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config={
            'name': 'adam',
            'lr': args.lr,
            'weight_decay': 1e-5
        },
        device=device,
        checkpoint_dir=experiment_dir / "checkpoints"
    )
    
    # Training parameters
    validation_frequency = 100
    early_stopping_patience = 10
    rl_update_frequency = 10 if rl_agent else None
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    try:
        # Start training
        trainer.train(
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            num_points=args.num_points,
            validation_frequency=validation_frequency,
            save_frequency=100
        )
        
        # Final evaluation
        logger.info("Training completed. Performing final evaluation...")
        final_metrics = pde.validate(model)
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
            num_points=1000,
            save_path=experiment_dir / "final_solution.html",
            use_rl=args.use_rl,
            rl_agent=rl_agent
        )
        
        logger.info("Experiment completed successfully")

    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    main()
