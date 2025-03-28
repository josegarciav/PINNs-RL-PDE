import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, List, Optional, Tuple
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
    ):
        """
        Initialize trainer.

        :param model: PINN model
        :param pde: PDE instance
        :param optimizer_config: Optimizer configuration
        :param device: Device to train on
        :param rl_agent: Reinforcement Learning agent
        :param viz_frequency: Frequency of visualization
        """
        self.device = device or torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.pde = pde

        # Setup optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optimizer_config.get("learning_rate", 0.001),
            weight_decay=optimizer_config.get("weight_decay", 0.0),
        )

        # Setup learning rate schedulers
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
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

        # Early stopping
        self.best_val_loss = float("inf")
        self.patience = optimizer_config.get("patience", 10)
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

        # Update cosine scheduler
        self.cosine_scheduler.step()

        # Log current learning rate
        current_lr = self.optimizer.param_groups[0]["lr"]
        self.history["learning_rate"].append(current_lr)
        self.logger.info(f"Current learning rate: {current_lr:.6f}")

    def train(
        self,
        num_epochs: int,
        batch_size: int,
        num_points: int,
        validation_frequency: int = 10,
        experiment_dir: str = None,
    ):
        """
        Train the model.

        :param num_epochs: Number of training epochs
        :param batch_size: Batch size for training
        :param num_points: Number of collocation points
        :param validation_frequency: Frequency of validation
        :param experiment_dir: Directory to save real-time training data
        """
        self.logger.info("Starting training...")

        # Create directories for visualizations and experiment data
        os.makedirs("visualizations", exist_ok=True)
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)

            # Save initial metadata
            metadata = {
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_architecture": self.model.__class__.__name__,
                "pde_type": self.pde.__class__.__name__,
                "training_params": {
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "num_points": num_points,
                    "validation_frequency": validation_frequency,
                },
            }

            # Save initial metadata
            try:
                save_training_metrics({}, experiment_dir, metadata)
            except Exception as e:
                self.logger.warning(f"Error saving initial metadata: {e}")

        points_history = []
        for epoch in range(num_epochs):
            self.model.train()
            epoch_losses = []

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

            # Compute average epoch loss
            avg_loss = np.mean(epoch_losses)
            self.history["train_loss"].append(avg_loss)

            # Validation
            if (epoch + 1) % validation_frequency == 0:
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
                    if self.patience_counter >= self.patience:
                        self.logger.info("Early stopping triggered!")
                        break

            # Store collocation points for visualization
            if self.rl_agent:
                points_history.append(x_batch.cpu().numpy())

            # Visualize collocation points at regular intervals
            if epoch % self.viz_frequency == 0 and self.rl_agent:
                self.rl_agent.visualize_collocation_evolution(points_history, epoch)

            # Generate comprehensive visualization at the end of training
            if epoch == num_epochs - 1 or (self.patience_counter >= self.patience):
                if hasattr(self.pde, "visualize_collocation_evolution"):
                    self.pde.visualize_collocation_evolution(
                        save_path=f"visualizations/final_collocation_evolution_epoch_{epoch}.png"
                    )

            # Save progress for real-time monitoring
            if experiment_dir:
                try:
                    # Update metrics
                    current_metrics = {
                        "current_epoch": epoch + 1,
                        "current_loss": avg_loss,
                        "best_val_loss": self.best_val_loss,
                        "early_stopping_counter": self.patience_counter,
                    }
                    save_training_metrics(self.history, experiment_dir, current_metrics)
                except Exception as e:
                    self.logger.warning(f"Error saving training metrics: {e}")

        # Save final metrics
        if experiment_dir:
            try:
                # Calculate training time in minutes
                start_time_str = metadata.get("start_time")
                end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Calculate duration if start time is available
                training_time_minutes = None
                if start_time_str:
                    try:
                        start_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
                        end_time = datetime.strptime(end_time_str, "%Y-%m-%d %H:%M:%S")
                        duration = end_time - start_time
                        training_time_minutes = duration.total_seconds() / 60.0
                    except Exception as e:
                        self.logger.warning(f"Error calculating training time: {e}")
                
                final_metrics = {
                    "end_time": end_time_str,
                    "total_epochs": epoch + 1,
                    "final_loss": avg_loss,
                    "best_val_loss": self.best_val_loss,
                    "early_stopping_triggered": self.patience_counter >= self.patience,
                    "training_time_minutes": training_time_minutes,
                }
                save_training_metrics(self.history, experiment_dir, final_metrics)
            except Exception as e:
                self.logger.warning(f"Error saving final metrics: {e}")

        return self.history

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history.

        :return: Dictionary of training metrics
        """
        return self.history

    def plot_training_history(self):
        """Plot training history."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 10))

        # Plot losses
        plt.subplot(2, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot component losses
        plt.subplot(2, 2, 2)
        plt.plot(self.history["residual_loss"], label="Residual Loss")
        plt.plot(self.history["boundary_loss"], label="Boundary Loss")
        plt.plot(self.history["initial_loss"], label="Initial Loss")
        plt.title("Component Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()

        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(self.history["learning_rate"], label="Learning Rate")
        plt.title("Learning Rate")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()

        plt.tight_layout()
        plt.savefig("training_history.png")
        plt.close()
