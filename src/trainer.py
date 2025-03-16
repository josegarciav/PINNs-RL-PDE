
import torch
import torch.optim as optim
from src.utils import generate_collocation_points

class PDETrainer:
    def __init__(self, pde, pinn, rl_agent, config):
        """
        Trainer for Physics-Informed Neural Networks with optional RL-based adaptive collocation.

        :param pde: PDE class instance.
        :param pinn: PINN neural network model.
        :param rl_agent: RL agent for adaptive collocation (or None).
        :param config: Dictionary of configuration parameters.
        """
        self.device = torch.device(config.get('device', 'mps') if torch.backends.mps.is_available() else 'cpu')
        self.pde = pde
        self.pinn = pinn.to(self.device)
        self.rl_agent = rl_agent
        self.config = config
        self.optimizer = optim.Adam(self.pinn.parameters(), lr=config['learning_rate'])
        self.lambda_bc = config.get('lambda_bc', 1.0) # Weight for boundary conditions loss

    def train(self):
        epochs = self.config["num_epochs"]
        num_points = self.config["num_points"]
        num_boundary_points = self.config["num_boundary_points"]
        log_interval = self.config["log_interval"]

        for epoch in range(epochs):
            # Generate collocation points
            if self.rl_agent:
                collocation_points = self.rl_agent.select_action(num_points).to(self.device)
            else:
                collocation_points = generate_collocation_points(num_points, domain=self.pde.domain, device=self.device)
            
            x_colloc, t_collocation = collocation_points[:, [0]], collocation_points[:, [1]]

            # Compute PDE residuals
            residual = self.pde.compute_residual(self.pinn, x=collocation_points[:, [0]], t=collocation_points[:, [1]])
            residual_loss = torch.mean(residual**2)

            # Compute boundary conditions loss
            boundary_points = generate_collocation_points(num_boundary_points, domain=self.pde.domain, device=self.device)
            x_bc, t_bc = boundary_points[:, [0]], boundary_points[:, [1]]
            bc_loss = self.pde.enforce_boundary_conditions(self.pinn, x=boundary_points[:, [0]], t=boundary_points[:, [1]])

            # Total loss
            total_loss = residual_loss + self.config['lambda_bc'] * bc_loss

            # Optimization step
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # RL agent update
            if self.rl_agent:
                reward = -total_loss.item()
                self.rl_agent.update(reward)

            # Logging
            if epoch % log_interval == 0 or epoch == epochs - 1:
                print(
                    f"Epoch [{epoch}/{epochs}]: Residual Loss = {residual_loss.item():.6f}, "
                    f"BC Loss = {bc_loss.item():.4f}, Total Loss = {total_loss.item():.4f}"
                )

        print("✅ Training completed successfully.")
