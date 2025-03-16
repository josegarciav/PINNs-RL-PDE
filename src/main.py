
import torch
from src.config import CONFIG
from src.pinn import PINNModel
from src.rl_agent import CollocationRLAgent
from src.trainer import PDETrainer
from src.pdes.heat_equation import HeatEquation
from src.utils import plot_pinn_solution, save_model, plot_pinn_3d_solution

def main():
    # Device setup
    device = torch.device(CONFIG.get("device", "mps") if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # PDE initialization
    pde = HeatEquation(domain=(0, 1), device=device)

    # PINN model initialization with explicit parameters
    pinn = PINNModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        num_layers=CONFIG['num_layers'],
        activation=torch.tanh if CONFIG['activation'] == 'tanh' else torch.relu,
        device=device
    )

    # RL agent initialization (optional)
    rl_agent = None
    if CONFIG['use_rl']:
        rl_agent = CollocationRLAgent(
            state_dim=CONFIG['input_dim'],
            action_dim=CONFIG['num_points'],
            lr=CONFIG['rl_learning_rate'],
            gamma=CONFIG['rl_gamma'],
            hidden_dim=CONFIG['rl_hidden_dim'],
            device=device
        )

    # Start training
    trainer = PDETrainer(pde, pinn, rl_agent, CONFIG)
    trainer.train()

    # Save trained model
    # save_model(pinn, CONFIG['model_save_path'])

    # Plot results
    plot_pinn_solution(pinn, pde, domain=pde.domain, device=device)
    plot_pinn_3d_solution(pinn, pde, domain=(0, 1), t_domain=(0, 1), resolution=100)

if __name__ == "__main__":
    main()
