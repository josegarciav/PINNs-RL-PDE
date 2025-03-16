
import torch
from src.pdes.heat_equation import HeatEquation
from src.pinn import PINNModel
from src.trainer import PDETrainer
from src.config import CONFIG
from src.utils import plot_pinn_solution

def main():
    # Set up device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Initialize PDE, PINN, and trainer
    pde = HeatEquation(device=device)
    pinn = PINNModel(input_dim=2, hidden_dim=64, output_dim=1).to(device)
    trainer = PDETrainer(pde, pinn, None, CONFIG)  # No RL for first test

    # Train PINN
    trainer.train()
    plot_pinn_solution(pinn, pde)

if __name__ == "__main__":
    main()
