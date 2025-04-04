import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import os
from datetime import datetime
from src.utils.utils import save_training_metrics
import plotly.graph_objects as go


class PDETrainer:
    """Trainer for Physics-Informed Neural Networks."""

    def __init__(
        self,
        model: nn.Module,
        pde: "PDEBase",
        optimizer_config: Dict,
        config: Dict,
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
        :param config: Full configuration dictionary
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
        self.config = config
        self.validation_frequency = validation_frequency

        # Setup optimizer
        self._initialize_optimizer_and_scheduler()

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

    def _update_scheduler(self, val_loss=None):
        """Update the learning rate scheduler."""
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss if val_loss is not None else self.train_loss)
        else:
            self.scheduler.step()

    def _initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler."""
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config["training"]["optimizer_config"]["learning_rate"],
            weight_decay=self.config["training"]["optimizer_config"]["weight_decay"]
        )

        # Initialize scheduler based on type
        scheduler_type = self.config["training"].get("scheduler_type", "reduce_lr")
        
        if scheduler_type == "reduce_lr":
            reduce_lr_params = self.config["training"].get("reduce_lr_params", {
                "factor": 0.5,
                "patience": 50,
                "min_lr": 1e-6
            })
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=reduce_lr_params["factor"],
                patience=reduce_lr_params["patience"],
                min_lr=reduce_lr_params["min_lr"],
                verbose=True
            )
        elif scheduler_type == "cosine":
            cosine_params = self.config["training"].get("cosine_params", {
                "T_max": 100,
                "eta_min": 1e-6
            })
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cosine_params["T_max"],
                eta_min=cosine_params["eta_min"]
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

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
                self._update_scheduler(val_losses["total_loss"])

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
        try:
            # Create parent directory if it doesn't exist
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            fig = go.Figure()

            # Add traces for each loss component
            for key in self.history.keys():
                if key != "lr":  # Skip learning rate
                    fig.add_trace(
                        go.Scatter(y=self.history[key], name=key, mode="lines")
                    )

            # Update layout
            fig.update_layout(
                title="Training History",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                yaxis_type="log",  # Use log scale for loss
            )

            if save_path:
                # Save both interactive HTML and static PNG
                html_path = save_path.replace(".png", ".html")
                fig.write_html(html_path)
                fig.write_image(save_path)

        except Exception as e:
            logging.warning(f"Error plotting training history: {e}")

    def plot_solution_comparison(self, num_points=50, save_path=None):
        """Plot comparison between exact and predicted solutions"""
        if self.pde.dimension == 2:
            # Generate grid points for x and y
            x = torch.linspace(
                self.pde.domain[0][0],
                self.pde.domain[0][1],
                num_points,
                device=self.device,
            )
            y = torch.linspace(
                self.pde.domain[1][0],
                self.pde.domain[1][1],
                num_points,
                device=self.device,
            )
            X, Y = torch.meshgrid(x, y, indexing="ij")

            # Generate frames for different time points
            frames = []
            times = torch.linspace(
                self.pde.time_domain[0], self.pde.time_domain[1], 10, device=self.device
            )

            for t in times:
                # Prepare input for the model
                t_repeated = t.repeat(num_points * num_points)
                points = torch.stack([X.flatten(), Y.flatten(), t_repeated], dim=1)

                # Get predictions and exact solutions
                with torch.no_grad():
                    pred = (
                        self.model(points).reshape(num_points, num_points).cpu().numpy()
                    )
                    exact = (
                        self.pde.exact_solution(points)
                        .reshape(num_points, num_points)
                        .cpu()
                        .numpy()
                    )
                    error = np.abs(pred - exact)

                # Create frame with both plots
                frame = go.Frame(
                    data=[
                        go.Surface(
                            x=X.cpu().numpy(),
                            y=Y.cpu().numpy(),
                            z=pred,
                            colorscale="viridis",
                            name="Predicted",
                        ),
                        go.Surface(
                            x=X.cpu().numpy(),
                            y=Y.cpu().numpy(),
                            z=exact,
                            colorscale="viridis",
                            name="Exact",
                        ),
                        go.Surface(
                            x=X.cpu().numpy(),
                            y=Y.cpu().numpy(),
                            z=error,
                            colorscale="viridis",
                            name="Error",
                        ),
                    ],
                    name=f"t={t.item():.2f}",
                )
                frames.append(frame)

            # Create figure with animation
            fig = go.Figure(
                data=[frames[0].data[0], frames[0].data[1], frames[0].data[2]],
                frames=frames,
            )

            # Add play button and slider
            fig.update_layout(
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            }
                        ],
                    }
                ],
                sliders=[
                    {
                        "currentvalue": {"prefix": "t="},
                        "steps": [
                            {
                                "args": [
                                    [f.name],
                                    {
                                        "frame": {"duration": 0, "redraw": True},
                                        "mode": "immediate",
                                    },
                                ],
                                "label": f.name,
                                "method": "animate",
                            }
                            for f in frames
                        ],
                    }
                ],
            )

            # Update layout
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5),
                    ),
                    xaxis_title="x",
                    yaxis_title="y",
                    zaxis_title="u(x,y,t)",
                ),
                title="Solution Comparison (2D Heat Equation)",
            )

            # Save both interactive and static plots
            if save_path:
                # Save interactive HTML
                html_path = save_path.replace(".png", ".html")
                fig.write_html(html_path)

                # Save static PNG for each time step
                for i, t in enumerate(times):
                    static_fig = go.Figure(data=frames[i].data)
                    static_fig.update_layout(
                        scene=dict(
                            camera=dict(
                                up=dict(x=0, y=0, z=1),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=1.5, y=1.5, z=1.5),
                            ),
                            xaxis_title="x",
                            yaxis_title="y",
                            zaxis_title="u(x,y,t)",
                        ),
                        title=f"Solution at t={t.item():.2f}",
                    )
                    png_path = save_path.replace(".png", f"_t{i}.png")
                    static_fig.write_image(png_path)

        else:  # 1D case
            # Generate points
            x = torch.linspace(
                self.pde.domain[0][0],
                self.pde.domain[0][1],
                num_points,
                device=self.device,
            )
            t = torch.linspace(
                self.pde.time_domain[0],
                self.pde.time_domain[1],
                num_points,
                device=self.device,
            )
            X, T = torch.meshgrid(x, t, indexing="ij")
            points = torch.stack([X.flatten(), T.flatten()], dim=1)

            # Get predictions and exact solutions
            with torch.no_grad():
                pred = self.model(points).reshape(num_points, num_points).cpu().numpy()
                exact = (
                    self.pde.exact_solution(points)
                    .reshape(num_points, num_points)
                    .cpu()
                    .numpy()
                )
                error = np.abs(pred - exact)

            # Create interactive plot
            fig = go.Figure()

            # Add surfaces for predicted, exact, and error
            fig.add_trace(
                go.Surface(
                    x=X.cpu().numpy(),
                    y=T.cpu().numpy(),
                    z=pred,
                    colorscale="viridis",
                    name="Predicted",
                )
            )
            fig.add_trace(
                go.Surface(
                    x=X.cpu().numpy(),
                    y=T.cpu().numpy(),
                    z=exact,
                    colorscale="viridis",
                    name="Exact",
                )
            )
            fig.add_trace(
                go.Surface(
                    x=X.cpu().numpy(),
                    y=T.cpu().numpy(),
                    z=error,
                    colorscale="viridis",
                    name="Error",
                )
            )

            # Update layout
            fig.update_layout(
                scene=dict(
                    camera=dict(
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=0),
                        eye=dict(x=1.5, y=1.5, z=1.5),
                    ),
                    xaxis_title="x",
                    yaxis_title="t",
                    zaxis_title="u(x,t)",
                ),
                title="Solution Comparison (1D)",
            )

            if save_path:
                # Save interactive HTML
                html_path = save_path.replace(".png", ".html")
                fig.write_html(html_path)

                # Save static PNG
                fig.write_image(save_path)

    def save_plots(self, save_dir):
        """Save training plots"""
        try:
            # Create visualization directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)

            # Save training history plot
            history_path = os.path.join(save_dir, "final_training_history.png")
            self.plot_training_history(save_path=history_path)

            # Save solution comparison plot
            solution_path = os.path.join(save_dir, "final_solution_comparison.png")
            self.plot_solution_comparison(save_path=solution_path)

            # Save collocation points evolution if available
            if (
                hasattr(self, "collocation_points_history")
                and self.collocation_points_history
            ):
                collocation_path = os.path.join(save_dir, "collocation_evolution.png")
                self.visualize_collocation_evolution(save_path=collocation_path)

            logging.info(f"Plots saved successfully to {save_dir}")
        except Exception as e:
            logging.warning(f"Error saving plots: {e}")

    def visualize_collocation_evolution(self, save_path=None):
        """Visualize the evolution of collocation points"""
        try:
            if (
                not hasattr(self, "collocation_points_history")
                or not self.collocation_points_history
            ):
                logging.warning("No collocation points history available")
                return

            # Create parent directory if it doesn't exist
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Create animation frames
            frames = []
            for i, points in enumerate(self.collocation_points_history):
                if self.pde.dimension == 2:
                    frame = go.Frame(
                        data=[
                            go.Scatter3d(
                                x=points[:, 0].cpu().numpy(),
                                y=points[:, 1].cpu().numpy(),
                                z=(
                                    points[:, 2].cpu().numpy()
                                    if points.shape[1] > 2
                                    else np.zeros_like(points[:, 0].cpu().numpy())
                                ),
                                mode="markers",
                                marker=dict(size=2),
                            )
                        ],
                        name=f"Epoch {i*10}",
                    )
                else:
                    frame = go.Frame(
                        data=[
                            go.Scatter(
                                x=points[:, 0].cpu().numpy(),
                                y=points[:, 1].cpu().numpy(),
                                mode="markers",
                                marker=dict(size=2),
                            )
                        ],
                        name=f"Epoch {i*10}",
                    )
                frames.append(frame)

            # Create figure with animation
            fig = go.Figure(data=frames[0].data, frames=frames)

            # Add play button
            fig.update_layout(
                updatemenus=[
                    {
                        "type": "buttons",
                        "showactive": False,
                        "buttons": [
                            {
                                "label": "Play",
                                "method": "animate",
                                "args": [
                                    None,
                                    {
                                        "frame": {"duration": 500, "redraw": True},
                                        "fromcurrent": True,
                                    },
                                ],
                            }
                        ],
                    }
                ],
                title="Collocation Points Evolution",
            )

            if save_path:
                # Save both interactive HTML and static PNG
                html_path = save_path.replace(".png", ".html")
                fig.write_html(html_path)

                # Save static PNG for first and last frame
                static_fig = go.Figure(data=frames[0].data)
                static_fig.write_image(save_path.replace(".png", "_initial.png"))
                static_fig = go.Figure(data=frames[-1].data)
                static_fig.write_image(save_path.replace(".png", "_final.png"))

        except Exception as e:
            logging.warning(f"Error visualizing collocation evolution: {e}")
