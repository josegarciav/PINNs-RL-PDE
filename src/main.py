
import torch
from config import CONFIG
from pinn import PINNModel
from rl_agent import CollocationRLAgent
from trainer import PDETrainer
from pdes.heat_equation import HeatEquation
from utils import plot_pinn_solution, save_model

def main():
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # PDE initialization
    pde = HeatEquation(device=device)

    # PINN model initialization
    pinn = PINNModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim']
    ).to(device)

    # Optional RL agent initialization
    rl_agent = None
    if CONFIG['use_rl']:
        rl_agent = CollocationRLAgent(
            state_dim=CONFIG['input_dim'],
            action_dim=CONFIG.get('num_points', 100),
            lr=CONFIG['rl_learning_rate'],
            gamma=CONFIG['rl_gamma'],
            device=device
        )

    # Training
    trainer = PDETrainer(pde, pinn, rl_agent, CONFIG)
    trainer.train()

    # Save trained model
    save_model(pinn, "pinn_trained.pth")

    # Visualize the results
    plot_pinn_solution(pinn, pde, domain=pde.domain)

if __name__ == "__main__":
    main()
