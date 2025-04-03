import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import os
from datetime import datetime
from src.utils.utils import save_training_metrics


class PDETrainer:
    """Trainer for Physics-Informed Neural Networks."""

    def __init__(
        self,
        model: nn.Module,
        pde: "PDEBase",
        optimizer_config: Dict,
        device: Optional[torch.device] = None,
        rl_agent=None,
        viz_frequency=10,
        validation_frequency=10,
        early_stopping_config=None,
    ):
        """
        Initialize trainer.

        :param model: PINN model
        :param pde: PDE instance
        :param optimizer_config: Optimizer configuration
        :param device: Device to train on
        :param rl_agent: Reinforcement Learning agent
        :param viz_frequency: Frequency of visualization
        :param validation_frequency: Frequency of validation checks
        :param early_stopping_config: Early stopping configuration
        """
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.pde = pde
        self.validation_frequency = validation_frequency

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optimizer_config.get("lr", 0.001),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )

        # Setup learning rate scheduler - using only ReduceLROnPlateau
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        # Training history
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "residual_loss": [],
            "boundary_loss": [],
            "initial_loss": [],
            "learning_rate": [],
        }

        # Early stopping configuration
        if early_stopping_config is None:
            early_stopping_config = {"enabled": True, "patience": 10}

        self.early_stopping_enabled = early_stopping_config.get("enabled", True)
        self.patience = early_stopping_config.get("patience", 10)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # Setup logging
        self._setup_logging()

        self.rl_agent = rl_agent
        self.viz_frequency = viz_frequency

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _compute_validation_loss(self, num_points: int = 1000) -> Dict[str, float]:
        """
        Compute validation loss on a separate set of points.

        :param num_points: Number of validation points
        :return: Dictionary of validation losses
        """
        self.model.eval()

        # Generate validation points
        x_val, t_val = self.pde.generate_collocation_points(num_points)
        x_val = x_val.to(self.device).requires_grad_(True)
        t_val = t_val.to(self.device).requires_grad_(True)

        # Compute losses
        losses = self.pde.compute_loss(self.model, x_val, t_val)

        return {
            "total_loss": losses["total"].item(),
            "residual_loss": losses["residual"].item(),
            "boundary_loss": losses["boundary"].item(),
            "initial_loss": losses["initial"].item(),
        }

    def _update_learning_rate(self, val_loss: float):
        """
        Update learning rate based on validation loss.

        :param val_loss: Current validation loss
        """
        # Update ReduceLROnPlateau scheduler
        self.scheduler.step(val_loss)

        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["learning_rate"].append(current_lr)
        self.logger.info(f"Current learning rate: {current_lr:.6f}")

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        num_points: int,
        experiment_dir: str = None,
    ):
        """Train the model."""
        self.logger.info("Starting training...")

        # Record start time
        start_time = datetime.now()

        # Create directories for visualizations and experiment data
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)
            viz_dir = os.path.join(experiment_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            self.logger.info(f"Saving visualizations to: {viz_dir}")

        points_history = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []
            epoch_points = []  # Store all points for this epoch

            # Training loop with progress bar
            pbar = tqdm(
                range(num_points // batch_size), desc=f"Epoch {epoch+1}/{num_epochs}"
            )
            for _ in pbar:
                # Generate batch of collocation points
                # Use adaptive sampling if RL agent is available
                sampling_strategy = (
                    "adaptive" if self.rl_agent is not None else "uniform"
                )
                x_batch, t_batch = self.pde.generate_collocation_points(
                    batch_size, strategy=sampling_strategy
                )
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                # Store points for visualization
                epoch_points.append(x_batch.cpu().detach().numpy())

                # Forward pass
                self.optimizer.zero_grad()

                # Compute losses
                losses = self.pde.compute_loss(self.model, x_batch, t_batch)
                total_loss = losses["total"]

                # Backward pass
                total_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Optimize
                self.optimizer.step()

                # Update progress bar
                epoch_losses.append(total_loss.item())
                pbar.set_postfix(
                    {
                        "loss": f"{total_loss.item():.6f}",
                        "residual": f"{losses['residual'].item():.6f}",
                        "boundary": f"{losses['boundary'].item():.6f}",
                        "initial": f"{losses['initial'].item():.6f}",
                    }
                )

            # Store average points for this epoch
            points_history.append(np.concatenate(epoch_points, axis=0))

            # Compute average epoch loss
            avg_loss = np.mean(epoch_losses)
            self.history["train_loss"].append(avg_loss)

            # Validation
            if (epoch + 1) % self.validation_frequency == 0:
                val_losses = self._compute_validation_loss()
                self.history["val_loss"].append(val_losses["total_loss"])
                self.history["residual_loss"].append(val_losses["residual_loss"])
                self.history["boundary_loss"].append(val_losses["boundary_loss"])
                self.history["initial_loss"].append(val_losses["initial_loss"])

                # Update learning rate
                self._update_learning_rate(val_losses["total_loss"])

                # Log validation metrics
                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs} - "
                    f"Train Loss: {avg_loss:.6f} - "
                    f"Val Loss: {val_losses['total_loss']:.6f} - "
                    f"Residual: {val_losses['residual_loss']:.6f} - "
                    f"Boundary: {val_losses['boundary_loss']:.6f} - "
                    f"Initial: {val_losses['initial_loss']:.6f}"
                )

                # Early stopping check
                if val_losses["total_loss"] < self.best_val_loss:
                    self.best_val_loss = val_losses["total_loss"]
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if (
                        self.patience_counter >= self.patience
                        and self.early_stopping_enabled
                    ):
                        self.logger.info("Early stopping triggered!")
                        break

            # Save progress for real-time monitoring
            if experiment_dir:
                try:
                    # Calculate current training time
                    current_time = datetime.now()
                    training_time_minutes = (
                        current_time - start_time
                    ).total_seconds() / 60.0

                    # Update metrics
                    current_metrics = {
                        "current_epoch": epoch + 1,
                        "current_loss": avg_loss,
                        "best_val_loss": self.best_val_loss,
                        "early_stopping_counter": self.patience_counter,
                        "training_time_minutes": training_time_minutes,
                        "early_stopping_triggered": self.patience_counter
                        >= self.patience
                        and self.early_stopping_enabled,
                    }
                    save_training_metrics(self.history, experiment_dir, current_metrics)
                except Exception as e:
                    self.logger.warning(f"Error saving training metrics: {e}")

        # Calculate final training time
        end_time = datetime.now()
        training_time_minutes = (end_time - start_time).total_seconds() / 60.0

        # Save final plots and metrics
        if experiment_dir:
            try:
                # Save final metrics
                final_metrics = {
                    "total_epochs": epoch + 1,
                    "final_loss": avg_loss,
                    "best_val_loss": self.best_val_loss,
                    "early_stopping_triggered": self.patience_counter >= self.patience
                    and self.early_stopping_enabled,
                    "training_time_minutes": training_time_minutes,
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                save_training_metrics(self.history, experiment_dir, final_metrics)

                # Save training history plot
                self.plot_training_history(
                    os.path.join(
                        experiment_dir, "visualizations", "final_training_history.png"
                    )
                )

                # Save solution comparison plot
                self.plot_solution_comparison(
                    num_points=200,  # Increased resolution for better visualization
                    save_path=os.path.join(
                        experiment_dir,
                        "visualizations",
                        "final_solution_comparison.png",
                    ),
                )

                # Save collocation evolution plot if enough points
                if len(points_history) > 1:
                    if hasattr(self.pde, "visualize_collocation_evolution"):
                        self.pde.visualize_collocation_evolution(
                            points_history=points_history,
                            save_path=os.path.join(
                                experiment_dir,
                                "visualizations",
                                "final_collocation_evolution.png",
                            ),
                        )
            except Exception as e:
                self.logger.warning(f"Error saving plots: {e}")

        return self.history

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.

        :return: Dictionary of training metrics
        """
        return self.history

    def plot_training_history(self, save_path=None):
        """Plot training history."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 10))

        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        if self.history["val_loss"]:
            # Fill missing validation loss values with None
            val_loss_full = []
            val_idx = 0
            for i in range(len(self.history["train_loss"])):
                if (i + 1) % self.validation_frequency == 0 and val_idx < len(
                    self.history["val_loss"]
                ):
                    val_loss_full.append(self.history["val_loss"][val_idx])
                    val_idx += 1
                else:
                    val_loss_full.append(None)
            # Plot only non-None values
            epochs = range(1, len(self.history["train_loss"]) + 1)
            val_epochs = [i for i, v in enumerate(val_loss_full, 1) if v is not None]
            val_values = [v for v in val_loss_full if v is not None]
            plt.plot(val_epochs, val_values, label="Val Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")  # Use log scale for better visualization
        plt.legend()

        # Plot component losses
        plt.subplot(2, 2, 2)
        if self.history["residual_loss"]:
            plt.plot(val_epochs, self.history["residual_loss"], label="Residual Loss")
        if self.history["boundary_loss"]:
            plt.plot(val_epochs, self.history["boundary_loss"], label="Boundary Loss")
        if self.history["initial_loss"]:
            plt.plot(val_epochs, self.history["initial_loss"], label="Initial Loss")
        plt.title("Component Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.yscale("log")  # Use log scale for better visualization
        plt.legend()

        # Plot learning rate
        plt.subplot(2, 2, 3)
        if self.history["learning_rate"]:
            plt.plot(val_epochs, self.history["learning_rate"], label="Learning Rate")
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")  # Use log scale for better visualization
        plt.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        else:
            plt.savefig("training_history.png")
        plt.close()

    def plot_solution_comparison(self, num_points=100, save_path=None):
        """Plot comparison between exact and predicted solutions."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Generate grid points
        x_domain = self.pde.config.domain[0]
        t_domain = self.pde.config.time_domain
        x = np.linspace(x_domain[0], x_domain[1], num_points)
        t = np.linspace(t_domain[0], t_domain[1], num_points)
        X, T = np.meshgrid(x, t)

        # Convert to torch tensors
        X_torch = torch.from_numpy(X.flatten()).float().to(self.device)
        T_torch = torch.from_numpy(T.flatten()).float().to(self.device)

        # Get predictions
        self.model.eval()
        with torch.no_grad():
            inputs = torch.stack([X_torch, T_torch], dim=1)
            predicted = self.model(inputs).cpu().numpy()

        # Reshape predictions
        predicted = predicted.reshape(X.shape)

        # Get exact solution
        X_tensor = torch.from_numpy(X).float().to(self.device)
        T_tensor = torch.from_numpy(T).float().to(self.device)
        exact = self.pde.exact_solution(X_tensor, T_tensor).cpu().numpy()

        # Calculate error
        error = np.abs(exact - predicted)

        # Create interactive Plotly figure
        fig = make_subplots(
            rows=1,
            cols=3,
            specs=[[{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]],
            subplot_titles=("Exact Solution", "Predicted Solution", "Absolute Error"),
        )

        # Add surfaces
        fig.add_trace(
            go.Surface(x=X, y=T, z=exact, colorscale="viridis", name="Exact"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Surface(x=X, y=T, z=predicted, colorscale="viridis", name="Predicted"),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Surface(x=X, y=T, z=error, colorscale="viridis", name="Error"),
            row=1,
            col=3,
        )

        # Update layout
        fig.update_layout(
            title="Solution Comparison",
            scene=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)"),
            scene2=dict(xaxis_title="x", yaxis_title="t", zaxis_title="u(x,t)"),
            scene3=dict(
                xaxis_title="x", yaxis_title="t", zaxis_title="|u_exact - u_pred|"
            ),
            width=1800,
            height=600,
        )

        # Save both static and interactive plots
        if save_path:
            # Save static matplotlib plot
            plt_save_path = save_path
            # Save interactive HTML
            html_save_path = save_path.rsplit(".", 1)[0] + ".html"
            fig.write_html(html_save_path)
        else:
            plt_save_path = "solution_comparison.png"
            fig.write_html("solution_comparison.html")

        # Also create and save static plot for compatibility
        fig_static = plt.figure(figsize=(20, 6))

        # Plot exact solution
        ax1 = fig_static.add_subplot(131, projection="3d")
        surf1 = ax1.plot_surface(X, T, exact, cmap="viridis")
        ax1.set_title("Exact Solution")
        ax1.set_xlabel("x")
        ax1.set_ylabel("t")
        ax1.set_zlabel("u(x,t)")
        plt.colorbar(surf1, ax=ax1)

        # Plot predicted solution
        ax2 = fig_static.add_subplot(132, projection="3d")
        surf2 = ax2.plot_surface(X, T, predicted, cmap="viridis")
        ax2.set_title("Predicted Solution")
        ax2.set_xlabel("x")
        ax2.set_ylabel("t")
        ax2.set_zlabel("u(x,t)")
        plt.colorbar(surf2, ax=ax2)

        # Plot error
        ax3 = fig_static.add_subplot(133, projection="3d")
        surf3 = ax3.plot_surface(X, T, error, cmap="viridis")
        ax3.set_title("Absolute Error")
        ax3.set_xlabel("x")
        ax3.set_ylabel("t")
        ax3.set_zlabel("|u_exact - u_pred|")
        plt.colorbar(surf3, ax=ax3)

        plt.tight_layout()
        plt.savefig(plt_save_path)
        plt.close()
