
import torch
import sys
from pathlib import Path
sys_path = Path(__file__).resolve().parent.parent

if str(sys_path) not in sys.path:
    sys.path.append(str(sys_path))

from src.pdes.heat_equation import HeatEquation
from src.pinn import PINNModel
from src.rl_agent import CollocationRLAgent
from src.trainer import PDETrainer
from src.config import CONFIG
from src.utils import plot_pinn_solution, plot_pinn_3d_solution


def main():
    device = torch.device(CONFIG.get('device', 'mps') if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # PDE initialization
    pde = HeatEquation(domain=(0, 1), device=device)

    # Initialize PINN
    pinn = PINNModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        device=device
    )

    # RL agent initialization (conditional)
    rl_agent = None
    if CONFIG['use_rl']:
        rl_agent = CollocationRLAgent(
            state_dim=CONFIG['input_dim'],
            action_dim=CONFIG['num_points'],
            lr=CONFIG['rl_learning_rate'],
            gamma=CONFIG['rl_gamma'],
            hidden_dim=CONFIG['hidden_dim'],
            device=device
        )

    # Trainer initialization
    trainer = PDETrainer(pde=pde, pinn=pinn, rl_agent=rl_agent, config=CONFIG)

    # Training
    trainer.train()

    # Visualization
    plot_pinn_solution(pinn, pde, domain=pde.domain, device=device)
    plot_pinn_3d_solution(pinn, pde, domain=pde.domain, t_domain=(0, 1), resolution=100, device=device)


if __name__ == "__main__":
    main()
