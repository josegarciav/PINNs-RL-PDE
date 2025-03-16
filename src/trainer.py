
import torch
import torch.optim as optim
from src.utils import generate_collocation_points


class PDETrainer:
    def __init__(self, pde, pinn, rl_agent, config):
        """
        Trainer for PINNs with optional RL-based adaptive collocation.
        
        :param pde: PDE instance.
        :param pinn: PINN model.
        :param rl_agent: Reinforcement Learning agent for collocation selection (can be None).
        :param config: Dictionary with training settings.
        """
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.pde = pde
        self.pinn = pinn.to(self.device)
        self.rl_agent = rl_agent
        self.config = config
        self.optimizer = optim.Adam(self.pinn.parameters(), lr=config['learning_rate'])
        self.lambda_bc = config.get('lambda_bc', 1.0)  # Weight for boundary loss

    def train(self):
        """
        Executes the training loop for the PINN model.
        """
        for epoch in range(self.config['num_epochs']):
            # Generate collocation points using RL or standard sampling
            if self.rl_agent:
                collocation_points = self.rl_agent.select_points()
            else:
                collocation_points = generate_collocation_points(self.config['num_points']).to(self.device)

            # Compute PDE residuals
            residual_loss = torch.mean(
                self.pde.compute_residual(
                    self.pinn, collocation_points[:, [0]], collocation_points[:, [1]]
                ) ** 2
            )

            # Compute Boundary Condition Loss
            boundary_points = generate_collocation_points(self.config['num_boundary_points']).to(self.device)
            bc_loss = self.pde.enforce_boundary_conditions(self.pinn, boundary_points)

            # Total loss: PDE loss + weighted boundary loss
            total_loss = residual_loss + self.lambda_bc * bc_loss

            # Optimize the PINN model
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update RL agent (if used)
            if self.rl_agent:
                reward = -total_loss.item()  # Reward function (minimizing loss)
                self.rl_agent.update_policy(reward)

            # Logging
            if epoch % self.config['log_interval'] == 0:
                print(f"Epoch {epoch}: Residual Loss = {residual_loss.item()}, BC Loss = {bc_loss.item()}, Total Loss = {total_loss.item()}")

        print("✅ Training completed.")
