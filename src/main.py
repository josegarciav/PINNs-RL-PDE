
import torch
from config import CONFIG
from pinn import PINNModel
from rl_agent import CollocationRLAgent
from trainer import PDETrainer
from pdes.heat_equation import HeatEquation  # Example PDE
from utils import plot_solution, save_model

def main():
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize PDE
    pde = HeatEquation()

    # Initialize PINN
    pinn = PINNModel(
        input_dim=CONFIG['input_dim'], 
        hidden_dim=CONFIG['hidden_dim'], 
        output_dim=CONFIG['output_dim']
    ).to(device)

    # Initialize RL agent (optional)
    rl_agent = None
    if CONFIG['use_rl']:
        rl_agent = CollocationRLAgent(
            state_dim=2, action_dim=1, 
            lr=CONFIG['rl_learning_rate'], gamma=CONFIG['rl_gamma']
        )

    # Train the model
    trainer = PDETrainer(pde, pinn, rl_agent, CONFIG)
    trainer.train()

    save_model(pinn, "pinn_trained.pth")

    plot_solution(pinn, pde)

if __name__ == "__main__":
    main()
