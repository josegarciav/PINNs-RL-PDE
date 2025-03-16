
import torch
import sys
import os

# Ensure project root is in path
import sys
from pathlib import Path
sys_path = Path(__file__).resolve().parents[1]
if str(sys_path) not in sys.path:
    sys.path.append(str(sys_path))

from src.pdes.heat_equation import HeatEquation
from src.pinn import PINNModel
from src.trainer import PDETrainer
from src.config import CONFIG
from src.utils import plot_pinn_solution

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize PDE
    pde = HeatEquation(domain=(0, 1), device=device)

    # Initialize PINN
    pinn = PINNModel(
        input_dim=CONFIG['input_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        output_dim=CONFIG['output_dim'],
        device=device
    )

    # Initialize trainer without RL
    trainer = PDETrainer(pde=pde, pinn=pinn, rl_agent=None, config=CONFIG)

    # Train model
    trainer.train()

    # Plot results
    plot_pinn_solution(pinn, pde)

if __name__ == "__main__":
    main()
