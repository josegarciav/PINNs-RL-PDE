
import torch
from src.pdes.heat_equation import HeatEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pinn import PINNModel
from src.trainer import PDETrainer
from src.rl_agent import CollocationRLAgent
from src.config import CONFIG


def run_experiment(pde_class, use_rl=False):
    """Run training for a given PDE and log results."""
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize PDE and PINN
    pde = pde_class(device=device)
    pinn = PINNModel(input_dim=2, hidden_dim=64, output_dim=1).to(device)

    # Initialize RL agent if needed
    rl_agent = CollocationRLAgent(state_dim=2, action_dim=1) if use_rl else None

    # Train model
    trainer = PDETrainer(pde, pinn, rl_agent, CONFIG)
    trainer.train()

if __name__ == "__main__":
    # Run experiments with and without RL-based collocation
    print("Training PINN on Heat Equation (Standard Collocation)")
    run_experiment(HeatEquation, use_rl=False)

    print("Training PINN on Heat Equation (RL-Optimized Collocation)")
    run_experiment(HeatEquation, use_rl=True)

    print("Training PINN on Black-Scholes Equation (Standard Collocation)")
    run_experiment(BlackScholesEquation, use_rl=False)

    print("Training PINN on Black-Scholes Equation (RL-Optimized Collocation)")
    run_experiment(BlackScholesEquation, use_rl=True)
