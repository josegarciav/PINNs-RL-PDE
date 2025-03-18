import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import setup_logging, save_model, load_model, plot_solution

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path(f"results/heat_equation_{timestamp}")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(experiment_dir / "training.log")
    logger.info("Starting Heat Equation training experiment")
    
    # PDE setup
    pde = HeatEquation(
        alpha=0.01,  # Thermal diffusivity
        domain=(0, 1),
        time_domain=(0, 1),
        device=device
    )
    
    # Model setup
    model = PINNModel(
        input_dim=2,  # (x, t)
        hidden_dim=64,
        output_dim=1,  # u(x, t)
        num_layers=4,
        activation='tanh',
        fourier_features=True,
        fourier_scale=10.0,
        dropout=0.1,
        layer_norm=True
    ).to(device)
    
    # RL Agent setup for adaptive collocation point sampling
    rl_agent = RLAgent(
        state_dim=2,  # (x, t)
        action_dim=1,  # Sampling probability
        hidden_dim=64,
        learning_rate=0.001,
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
    logger.info(f"RL Agent architecture:\n{rl_agent}")
    
    # Training setup
    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config={
            'name': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5
        },
        device=device,
        checkpoint_dir=experiment_dir / "checkpoints"
    )
    
    # Training parameters
    num_epochs = 10000
    batch_size = 1000
    num_collocation_points = 10000
    validation_frequency = 100
    early_stopping_patience = 10
    rl_update_frequency = 10  # Update RL agent every N epochs
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Generate collocation points using RL agent
        if epoch % rl_update_frequency == 0:
            # Update RL agent's policy
            rl_agent.update_epsilon(epoch)
            x, t = pde.generate_adaptive_collocation_points(num_collocation_points, rl_agent)
        else:
            # Use current policy for sampling
            x, t = pde.generate_adaptive_collocation_points(num_collocation_points, rl_agent)
        
        # Training step
        train_loss = trainer.train_step(x, t, batch_size)
        
        # Validation
        if (epoch + 1) % validation_frequency == 0:
            val_loss = trainer._compute_validation_loss()
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            logger.info(f"Training Loss: {train_loss:.6f}")
            logger.info(f"Validation Loss: {val_loss:.6f}")
            logger.info(f"RL Agent Epsilon: {rl_agent.epsilon:.6f}")
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model and RL agent
                save_model(model, experiment_dir / "best_model.pth")
                save_model(rl_agent, experiment_dir / "best_rl_agent.pth")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    logger.info("Early stopping triggered")
                    break
            
            # Plot current solution
            plot_solution(
                model=model,
                pde=pde,
                num_points=1000,
                save_path=experiment_dir / f"solution_epoch_{epoch+1}.png"
            )
    
    # Final evaluation
    logger.info("Training completed. Performing final evaluation...")
    final_metrics = trainer.evaluate()
    logger.info(f"Final metrics: {final_metrics}")
    
    # Save final model, RL agent, and metrics
    save_model(model, experiment_dir / "final_model.pth")
    save_model(rl_agent, experiment_dir / "final_rl_agent.pth")
    np.save(experiment_dir / "final_metrics.npy", final_metrics)
    
    # Plot final solution
    plot_solution(
        model=model,
        pde=pde,
        num_points=1000,
        save_path=experiment_dir / "final_solution.png"
    )
    
    logger.info("Experiment completed successfully")

if __name__ == "__main__":
    main()
