import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
import logging
from tqdm import tqdm
import os
from datetime import datetime
from src.utils.utils import save_training_metrics
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
from src.components.adaptive_weights import AdaptiveLossWeights

matplotlib.use("agg")
import matplotlib.pyplot as plt


class PDETrainer:
    """Trainer for Physics-Informed Neural Networks."""

    def __init__(
        self,
        model: nn.Module,
        pde: "PDEBase",
        optimizer_config: Dict,
        config: "Config",
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
        :param config: Full configuration object
        :param device: Device to train on
        :param rl_agent: Reinforcement Learning agent
        :param viz_frequency: Frequency of visualization
        :param validation_frequency: Frequency of validation checks
        :param early_stopping_config: Early stopping configuration
        """
        # Use device from config by default, fall back to provided device or cpu
        self.device = device or (
            config.device
            if hasattr(config, "device")
            else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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
            "loss_weights": [],  # Track weight evolution
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

        # Initialize adaptive weights handler if enabled
        self.use_adaptive_weights = config.training.adaptive_weights.enabled
        if self.use_adaptive_weights:
            self.adaptive_weights = AdaptiveLossWeights(
                strategy=config.training.adaptive_weights.strategy,
                alpha=config.training.adaptive_weights.alpha,
                eps=config.training.adaptive_weights.eps,
            )
        else:
            self.logger.info("Adaptive weights are disabled")
            self.adaptive_weights = None

    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def setup_experiment_logging(self, experiment_dir=None):
        """
        Setup experiment-specific logging if experiment_dir is provided.

        :param experiment_dir: Directory for the experiment
        """
        if experiment_dir:
            # Remove all FileHandlers from the logger
            for handler in self.logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    self.logger.removeHandler(handler)

            # Add experiment-specific log file handler
            log_file = os.path.join(experiment_dir, "experiment.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
            )
            self.logger.addHandler(file_handler)
            self.logger.info(f"Experiment logs will be saved to {log_file}")

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
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

        # Initialize scheduler based on type
        scheduler_type = self.config.training.learning_rate_scheduler.type

        if scheduler_type == "reduce_lr":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.training.learning_rate_scheduler.factor,
                patience=self.config.training.learning_rate_scheduler.patience,
                min_lr=self.config.training.learning_rate_scheduler.min_lr,
                verbose=True,
            )
        elif scheduler_type == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.learning_rate_scheduler.min_lr,
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
        self.model.train()
        if hasattr(self.model, "architecture_name"):
            architecture_name = self.model.architecture_name
        else:
            architecture_name = self.model.__class__.__name__

        # Log PDE and architecture information
        self.logger.info("=" * 50)
        self.logger.info(f"STARTING TRAINING FOR PDE: {self.pde.__class__.__name__}")
        self.logger.info(f"NEURAL NETWORK ARCHITECTURE: {architecture_name}")
        if hasattr(self.model, "model"):
            self.logger.info(f"MODEL STRUCTURE: {self.model.model}")
        self.logger.info(f"DEVICE: {self.device}")

        # Log adaptive weights configuration if enabled
        if self.use_adaptive_weights:
            self.logger.info(f"ADAPTIVE WEIGHTS ENABLED")
            self.logger.info(
                f"  Strategy: {self.config.training.adaptive_weights.strategy}"
            )
            self.logger.info(f"  Alpha: {self.config.training.adaptive_weights.alpha}")
            self.logger.info(f"  Epsilon: {self.config.training.adaptive_weights.eps}")
        else:
            self.logger.info("ADAPTIVE WEIGHTS DISABLED")
            # Convert loss_weights to something printable
            weights_dict = {
                k: float(v) if isinstance(v, torch.Tensor) else v
                for k, v in self.config.training.loss_weights.items()
            }
            self.logger.info(f"  Fixed weights: {weights_dict}")
        self.logger.info("=" * 50)

        # Initialize adaptive weights history
        if self.use_adaptive_weights and "loss_weights" not in self.history:
            self.history["loss_weights"] = []

        # Initialize the adaptive weights object if it's enabled but not initialized
        if self.use_adaptive_weights and self.adaptive_weights is None:
            self.logger.info("Initializing adaptive weights")
            self.adaptive_weights = AdaptiveLossWeights(
                strategy=self.config.training.adaptive_weights.strategy,
                alpha=self.config.training.adaptive_weights.alpha,
                eps=self.config.training.adaptive_weights.eps,
            )

        # Record start time
        start_time = datetime.now()

        # Create directories for visualizations and experiment data
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)

            # Setup experiment-specific logging
            self.setup_experiment_logging(experiment_dir)

            # Create visualization directory
            viz_dir = os.path.join(experiment_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

        self.logger.info("Starting training...")

        # Initialize points history
        self.points_history = []

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
                    "adaptive"
                    if self.rl_agent is not None
                    else self.config.training.collocation_distribution
                    # "uniform", "latin_hypercube", "sobol"
                )
                x_batch, t_batch = self.pde.generate_collocation_points(
                    batch_size, strategy=sampling_strategy
                )
                # Ensure the tensors are on the right device
                x_batch = x_batch.to(self.device)
                t_batch = t_batch.to(self.device)

                # Store points for visualization
                points = torch.cat(
                    [x_batch, t_batch], dim=1
                )  # t_batch is already [N, 1]
                epoch_points.append(points.cpu().detach().numpy())

                # Forward pass
                self.optimizer.zero_grad()
                losses = self.pde.compute_loss(self.model, x_batch, t_batch)

                # Apply adaptive weights if enabled
                if self.use_adaptive_weights:
                    # Prepare loss components tensor - include smoothness if present
                    loss_components = []
                    component_names = ["residual", "boundary", "initial"]

                    # Add each component to the list
                    for component in component_names:
                        loss_components.append(losses[component])

                    # Add smoothness component if it exists and has a non-zero weight
                    smoothness_weight = self.config.training.loss_weights.get(
                        "smoothness", 0.0
                    )
                    if "smoothness" in losses and smoothness_weight > 0:
                        loss_components.append(losses["smoothness"])
                        component_names.append("smoothness")

                    # Convert to tensor
                    loss_components = torch.tensor(loss_components, device=self.device)

                    if self.config.training.adaptive_weights.strategy == "lrw":
                        # For Learning Rate Weighting, we need gradients
                        # First, get individual gradients for each component
                        grad_norms = []
                        for i, loss_name in enumerate(component_names):
                            component_loss = losses[loss_name]
                            self.optimizer.zero_grad()
                            component_loss.backward(retain_graph=True)

                            # Calculate gradient norm for this component
                            grad_norm = 0.0
                            for param in self.model.parameters():
                                if param.grad is not None:
                                    grad_norm += param.grad.norm().item() ** 2
                            grad_norm = torch.tensor(grad_norm**0.5, device=self.device)
                            grad_norms.append(grad_norm)

                        # Get adaptive weights based on gradients
                        grad_norms = torch.stack(grad_norms)
                        weights = self.adaptive_weights.update(gradients=grad_norms)
                    else:
                        # For Relative Error Weighting
                        weights = self.adaptive_weights.update(losses=loss_components)

                    # Apply weights to compute the weighted loss
                    total_loss = 0
                    for i, component in enumerate(component_names):
                        if i < len(
                            weights
                        ):  # Ensure we have a weight for this component
                            total_loss += weights[i] * losses[component]

                    losses["total"] = total_loss

                    # Print detailed debugging info
                    print(f"\nAdaptive weights calculation:")
                    print(
                        f"- Strategy: {self.config.training.adaptive_weights.strategy}"
                    )
                    for i, component in enumerate(component_names):
                        if i < len(weights):
                            # Ensure values are Python floats, not tensors
                            loss_val = (
                                losses[component].item()
                                if isinstance(losses[component], torch.Tensor)
                                else losses[component]
                            )
                            weight_val = (
                                weights[i].item()
                                if isinstance(weights[i], torch.Tensor)
                                else weights[i]
                            )
                            weighted_val = weight_val * loss_val
                            print(
                                f"- {component}: loss={loss_val:.6f}, weight={weight_val:.6f}, weighted={weighted_val:.6f}"
                            )

                    # Ensure total_loss is a Python float
                    total_loss_val = (
                        total_loss.item()
                        if isinstance(total_loss, torch.Tensor)
                        else total_loss
                    )
                    print(f"- Total loss: {total_loss_val:.6f}\n")

                    # Store weights for visualization/tracking
                    if isinstance(weights, torch.Tensor):
                        weights_np = weights.detach().cpu().numpy()
                    else:
                        weights_np = np.array(weights)

                    # Ensure we have 4 weights (including smoothness) for consistency in visualization
                    if len(weights_np) < 4:
                        padding = np.zeros(4 - len(weights_np))
                        weights_np = np.concatenate([weights_np, padding])

                    self.history["loss_weights"].append(weights_np)

                loss = losses["total"]

                # Backward pass
                loss.backward()
                if self.config.training.gradient_clipping > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.training.gradient_clipping
                    )
                self.optimizer.step()

                # Update progress bar
                pbar.set_postfix(
                    {"loss": loss.item(), "res": losses["residual"].item()}
                )
                epoch_losses.append(loss.item())

            # End of epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            points_array = np.concatenate(epoch_points, axis=0)
            self.points_history.append(points_array)

            # Update learning rate
            self._update_scheduler(avg_epoch_loss)

            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Ensure all values are Python floats, not tensors
            def ensure_float(val):
                if isinstance(val, torch.Tensor):
                    return val.item()
                return val

            epoch_data = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "residual_loss": ensure_float(losses["residual"]),
                "boundary_loss": ensure_float(losses["boundary"]),
                "initial_loss": ensure_float(losses["initial"]),
                "learning_rate": current_lr,
            }

            # Add weights info if using adaptive weights
            if self.use_adaptive_weights and len(self.history["loss_weights"]) > 0:
                current_weights = self.history["loss_weights"][-1]
                weights_str = ""
                component_names = ["residual", "boundary", "initial"]
                if len(current_weights) >= 4:
                    component_names.append("smoothness")

                for i, name in enumerate(component_names):
                    if i < len(current_weights):
                        weight_value = current_weights[i]
                        # Ensure weight_value is a Python float, not a tensor
                        if isinstance(weight_value, torch.Tensor):
                            weight_value = weight_value.item()
                        weights_str += f", {name}_weight={weight_value:.4f}"

                # Ensure weights_str is a string, not a tensor
                if isinstance(weights_str, torch.Tensor):
                    weights_str = (
                        weights_str.item()
                        if weights_str.numel() == 1
                        else str(weights_str.tolist())
                    )

                self.logger.info(f"Adaptive weights:{weights_str}")
            else:
                weights_str = ""

            # Update history
            for key, value in epoch_data.items():
                if key in self.history:
                    self.history[key].append(value)

            # Validation (if applicable)
            if epoch % self.validation_frequency == 0:
                val_losses = self._compute_validation_loss()
                self.history["val_loss"].append(val_losses["total_loss"])
                # Ensure weights_str is a string, not a tensor
                if isinstance(weights_str, torch.Tensor):
                    weights_str = (
                        weights_str.item()
                        if weights_str.numel() == 1
                        else str(weights_str.tolist())
                    )

                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Train Loss: {avg_epoch_loss:.6f}, "
                    f"Val Loss: {val_losses['total_loss']:.6f}, "
                    f"LR: {current_lr:.6f}{weights_str}"
                )

                # Early stopping check
                if self.early_stopping_enabled:
                    val_loss = val_losses["total_loss"]
                    if val_loss < self.best_val_loss - 1e-6:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1

                    if self.patience_counter >= self.patience:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch+1}, "
                            f"best val loss: {self.best_val_loss:.6f}"
                        )
                        break
            else:
                # Ensure weights_str is a string, not a tensor
                if isinstance(weights_str, torch.Tensor):
                    weights_str = (
                        weights_str.item()
                        if weights_str.numel() == 1
                        else str(weights_str.tolist())
                    )

                self.logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, "
                    f"Train Loss: {avg_epoch_loss:.6f}, "
                    f"LR: {current_lr:.6f}{weights_str}"
                )

        # End of training
        train_time = (datetime.now() - start_time).total_seconds() / 60.0
        self.logger.info(f"Training completed in {train_time} minutes")

        # Save final visualizations and metrics
        if experiment_dir:
            # Create metadata with training information
            metadata = {
                "training_time_minutes": train_time,
                "total_epochs": num_epochs,
                "final_loss": (
                    float(self.history["train_loss"][-1])
                    if self.history["train_loss"]
                    else None
                ),
                "best_val_loss": (
                    float(self.best_val_loss)
                    if self.best_val_loss != float("inf")
                    else None
                ),
                "early_stopping_triggered": (
                    self.patience_counter >= self.patience
                    if self.early_stopping_enabled
                    else False
                ),
                "current_epoch": len(self.history["train_loss"]),
                "training_params": {
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "num_points": num_points,
                    "validation_frequency": self.validation_frequency,
                },
                "rl_enabled": self.rl_agent is not None,
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                # PDE information
                "pde_type": getattr(
                    self.pde,
                    "pde_type",
                    getattr(self.pde, "name", type(self.pde).__name__),
                ),
                "pde_name": getattr(
                    self.pde.config, "name", getattr(self.pde, "name", "")
                ),
                # PDE parameters and conditions
                "domain": getattr(self.pde, "domain", []),
                "time_domain": getattr(self.pde, "time_domain", []),
                "boundary_conditions": getattr(
                    self.pde.config, "boundary_conditions", {}
                ),
                "initial_condition": getattr(self.pde.config, "initial_condition", {}),
                "pde_parameters": getattr(self.pde.config, "parameters", {}),
                # Architecture information
                "architecture": getattr(
                    self.model,
                    "architecture_name",
                    getattr(
                        self.config,
                        "architecture",
                        (
                            getattr(self.config.model, "architecture", "unknown")
                            if hasattr(self.config, "model")
                            else "unknown"
                        ),
                    ),
                ),
            }

            # Save training history as metrics.json and history.json
            self.logger.info("Saving training metrics...")
            save_training_metrics(self.history, experiment_dir, metadata=metadata)

            # Save the model directly in the experiment directory
            self.logger.info("Saving model...")
            model_path = os.path.join(experiment_dir, "final_model.pt")
            torch.save(self.model.state_dict(), model_path)

            # Save all plots using the save_plots method
            self.logger.info("Generating and saving all plots...")
            self.save_plots(viz_dir)

            # Generate FDM comparison plots explicitly
            self.logger.info("Generating FDM comparison plots...")
            try:
                self.generate_fdm_comparison(experiment_dir)
            except Exception as e:
                self.logger.error(f"Error generating FDM comparison: {str(e)}")
                import traceback

                traceback.print_exc()

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

            # Create two subplots: one for losses and one for weights if using adaptive weights
            if self.use_adaptive_weights and len(self.history["loss_weights"]) > 0:
                fig = make_subplots(
                    rows=2, cols=1, subplot_titles=("Losses", "Loss Weights")
                )

                # Add traces for each loss component in first subplot
                for key in [
                    "train_loss",
                    "val_loss",
                    "residual_loss",
                    "boundary_loss",
                    "initial_loss",
                ]:
                    if key in self.history and self.history[key]:
                        fig.add_trace(
                            go.Scatter(y=self.history[key], name=key, mode="lines"),
                            row=1,
                            col=1,
                        )

                # Add traces for loss weights in second subplot
                weights = np.array(self.history["loss_weights"])
                if len(weights) > 0:
                    components = ["residual", "boundary", "initial", "smoothness"]
                    for i, component in enumerate(components):
                        if (
                            i < weights.shape[1]
                        ):  # Ensure we have data for this component
                            fig.add_trace(
                                go.Scatter(
                                    y=weights[:, i],
                                    name=f"{component}_weight",
                                    mode="lines",
                                ),
                                row=2,
                                col=1,
                            )

                # Update layout
                fig.update_layout(
                    title="Training History",
                    height=800,  # Increase height for two subplots
                )

                # Update y-axis for losses to be log scale
                fig.update_yaxes(type="log", title="Loss", row=1, col=1)

                # Update y-axis for weights
                fig.update_yaxes(title="Weight Value", row=2, col=1)

                # Update x-axis
                fig.update_xaxes(title="Epoch", row=2, col=1)
            else:
                # Original single plot for losses
                fig = go.Figure()

                # Add traces for each loss component
                for key in self.history.keys():
                    if (
                        key not in ["lr", "loss_weights"] and self.history[key]
                    ):  # Skip learning rate and weights
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
        try:
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
                    self.pde.config.time_domain[0],
                    self.pde.config.time_domain[1],
                    10,
                    device=self.device,
                )

                for t in times:
                    # Prepare input for the model
                    t_repeated = t.repeat(num_points * num_points)
                    points = torch.stack([X.flatten(), Y.flatten(), t_repeated], dim=1)

                    # Get predictions and exact solutions
                    with torch.no_grad():
                        pred = (
                            self.model(points)
                            .reshape(num_points, num_points)
                            .cpu()
                            .numpy()
                        )

                        # Add logging and error checking for exact solution
                        logging.info(f"Computing exact solution for t={t.item()}")
                        exact = self.pde.exact_solution(X.flatten(), Y.flatten(), t)
                        if exact is None:
                            raise ValueError("exact_solution returned None")

                        logging.info(
                            f"Exact solution shape before reshape: {exact.shape}"
                        )
                        exact = exact.reshape(num_points, num_points).cpu().numpy()
                        abs_error = np.abs(pred - exact)
                        rel_error = np.abs(
                            (pred - exact) / (exact + 1e-10)
                        )  # Add small constant to avoid division by zero

                        # Take the minimum between absolute and relative errors
                        error = np.minimum(abs_error, rel_error)

                        # Apply log scale to error
                        error = np.log10(
                            error + 1e-10
                        )  # Add small constant to avoid log(0)

                    # Create frame with both plots side by side and error
                    frame = go.Frame(
                        data=[
                            go.Surface(
                                x=X.cpu().numpy(),
                                y=Y.cpu().numpy(),
                                z=exact,
                                colorscale="viridis",
                                name="Exact",
                                showscale=True,
                                subplot="scene1",
                            ),
                            go.Surface(
                                x=X.cpu().numpy(),
                                y=Y.cpu().numpy(),
                                z=pred,
                                colorscale="viridis",
                                name="PINN Predicted",
                                showscale=True,
                                subplot="scene2",
                            ),
                            go.Surface(
                                x=X.cpu().numpy(),
                                y=Y.cpu().numpy(),
                                z=error,
                                colorscale="viridis",
                                name="Log Min Error",
                                showscale=True,
                                subplot="scene3",
                            ),
                        ],
                        name=f"t={t.item():.2f}",
                    )
                    frames.append(frame)

                # Create figure with animation and subplots
                fig = go.Figure(
                    data=[frames[0].data[0], frames[0].data[1], frames[0].data[2]],
                    frames=frames,
                )

                # Update layout with three subplots side by side
                fig.update_layout(
                    scene=dict(domain=dict(x=[0, 0.33], y=[0, 1])),
                    scene2=dict(domain=dict(x=[0.33, 0.66], y=[0, 1])),
                    scene3=dict(domain=dict(x=[0.66, 1], y=[0, 1])),
                    title=f"Solution Comparison (2D)",
                    width=1800,  # Increased width for better visibility
                    height=600,
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

                if save_path:
                    try:
                        # First save the requested format (png, jpg, etc.)
                        fig.write_image(save_path)

                        # Then try to save HTML format as well
                        base_path = os.path.splitext(save_path)[0]
                        html_path = base_path + ".html"
                        fig.write_html(html_path)

                        # Remove individual time step saving to simplify output
                        logging.info(
                            f"Saved visualizations to {save_path} and {html_path}"
                        )
                    except Exception as e:
                        logging.warning(f"Error saving visualization: {e}")

            else:  # 1D case
                # Generate points
                x = torch.linspace(
                    self.pde.domain[0][0],
                    self.pde.domain[0][1],
                    num_points,
                    device=self.device,
                )
                t = torch.linspace(
                    self.pde.config.time_domain[0],
                    self.pde.config.time_domain[1],
                    num_points,
                    device=self.device,
                )
                X, T = torch.meshgrid(x, t, indexing="ij")
                points = torch.stack([X.flatten(), T.flatten()], dim=1)

                # Get predictions and exact solutions
                with torch.no_grad():
                    pred = (
                        self.model(points).reshape(num_points, num_points).cpu().numpy()
                    )

                    # Add logging and error checking for exact solution
                    logging.info("Computing exact solution for 1D case")
                    exact = self.pde.exact_solution(X.flatten(), t=T.flatten())
                    if exact is None:
                        raise ValueError("exact_solution returned None")

                    logging.info(f"Exact solution shape before reshape: {exact.shape}")
                    exact = exact.reshape(num_points, num_points).cpu().numpy()

                    # Calculate both absolute and relative errors
                    abs_error = np.abs(pred - exact)
                    rel_error = np.abs(
                        (pred - exact) / (exact + 1e-10)
                    )  # Add small constant to avoid division by zero

                    # Take the minimum between absolute and relative errors
                    error = np.minimum(abs_error, rel_error)

                    # Apply log scale to error
                    error = np.log10(
                        error + 1e-10
                    )  # Add small constant to avoid log(0)

                # Create figure with subplots
                fig = make_subplots(
                    rows=1,
                    cols=3,
                    subplot_titles=("Exact", "PINN Predicted", "Log Min Error"),
                    specs=[
                        [{"type": "surface"}, {"type": "surface"}, {"type": "surface"}]
                    ],
                )

                # Add surfaces for exact, predicted, and error
                fig.add_trace(
                    go.Surface(
                        x=X.cpu().numpy(),
                        y=T.cpu().numpy(),
                        z=exact,
                        colorscale="viridis",
                        name="Exact",
                        showscale=False,  # Hide colorscale for exact solution
                    ),
                    row=1,
                    col=1,
                )
                fig.add_trace(
                    go.Surface(
                        x=X.cpu().numpy(),
                        y=T.cpu().numpy(),
                        z=pred,
                        colorscale="viridis",
                        name="PINN Predicted",
                        showscale=False,  # Hide colorscale for predicted solution
                    ),
                    row=1,
                    col=2,
                )
                fig.add_trace(
                    go.Surface(
                        x=X.cpu().numpy(),
                        y=T.cpu().numpy(),
                        z=error,
                        colorscale="viridis",
                        name="Log Min Error",
                        showscale=True,  # Only show colorscale for error
                        colorbar=dict(
                            title="Log Error",
                            x=0.98,  # Position colorbar at the right edge
                            y=0.5,  # Center colorbar vertically
                        ),
                    ),
                    row=1,
                    col=3,
                )

                # Update layout
                fig.update_layout(
                    title=f"Solution Comparison (1D)",
                    width=1800,  # Increased width for better visibility
                    height=600,
                    scene=dict(
                        xaxis_title="x",
                        yaxis_title="t",
                        zaxis_title="u(x,t)",
                    ),
                    scene2=dict(
                        xaxis_title="x",
                        yaxis_title="t",
                        zaxis_title="u(x,t)",
                    ),
                    scene3=dict(
                        xaxis_title="x",
                        yaxis_title="t",
                        zaxis_title="u(x,t)",
                    ),
                )

                if save_path:
                    try:
                        # First save the requested format (png, jpg, etc.)
                        fig.write_image(save_path)

                        # Then try to save HTML format as well
                        base_path = os.path.splitext(save_path)[0]
                        html_path = base_path + ".html"
                        fig.write_html(html_path)

                        # Remove individual time step saving to simplify output
                        logging.info(
                            f"Saved visualizations to {save_path} and {html_path}"
                        )
                    except Exception as e:
                        logging.warning(f"Error saving visualization: {e}")

        except Exception as e:
            logging.error(f"Error in plot_solution_comparison: {str(e)}")
            raise  # Re-raise the exception after logging

    def save_plots(self, save_dir):
        """Save training plots"""
        try:
            # Create visualization directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            logging.info(f"Saving plots to directory: {save_dir}")

            # Save training history plot
            history_path = os.path.join(save_dir, "final_training_history.png")
            self.plot_training_history(save_path=history_path)
            logging.info("Training history plot saved successfully")

            # Save solution comparison plot
            solution_path = os.path.join(save_dir, "final_solution_comparison.png")
            self.plot_solution_comparison(num_points=100, save_path=solution_path)
            logging.info("Solution comparison plots saved successfully")

            # Save collocation points evolution if available
            if hasattr(self, "points_history") and self.points_history:
                collocation_path = os.path.join(save_dir, "collocation_evolution.png")
                self.visualize_collocation_evolution(save_path=collocation_path)
                logging.info("Collocation evolution plot saved successfully")

            logging.info(f"All plots saved successfully to {save_dir}")
        except Exception as e:
            logging.error(f"Error saving plots: {str(e)}")

    def visualize_collocation_evolution(self, save_path=None):
        """Visualize the evolution of collocation points in a unified 2x2 grid"""
        try:
            # Check if we have points history from training
            if not hasattr(self, "points_history") or not self.points_history:
                logging.warning("No collocation points history available")
                return

            # Create parent directory if it doesn't exist
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # We'll use matplotlib for this visualization to create a 2x2 grid
            import matplotlib.pyplot as plt
            from matplotlib.colors import LinearSegmentedColormap
            import numpy as np

            # Create a figure with 2x2 subplots
            fig, axs = plt.subplots(2, 2, figsize=(16, 14))
            fig.suptitle("Evolution of Collocation Points", fontsize=20)

            # Define custom colormaps for each subplot
            colors = {
                "initial": plt.cm.Blues,  # Blue for initial
                "mid": plt.cm.Greens,  # Green for mid-training
                "final": plt.cm.Reds,  # Red for final
            }

            # Extract domain information
            if self.pde.dimension == 1:
                x_domain = self.pde.domain[0]
                t_domain = self.pde.config.time_domain

                # Determine indices for snapshots
                num_snapshots = len(self.points_history)
                initial_idx = 0
                mid_idx = num_snapshots // 2
                final_idx = num_snapshots - 1

                # Top-left: Evolution of points over time (superimpose snapshots)
                ax = axs[0, 0]
                ax.set_title("Progression of Points", fontsize=16)

                # Plot multiple snapshots with different colors
                snapshot_indices = []
                if num_snapshots >= 5:
                    # Choose 5 evenly spaced snapshots
                    step = num_snapshots // 5
                    snapshot_indices = [i * step for i in range(5)]
                    if snapshot_indices[-1] != num_snapshots - 1:
                        snapshot_indices[-1] = num_snapshots - 1
                else:
                    # Use all available snapshots
                    snapshot_indices = list(range(num_snapshots))

                for i, idx in enumerate(snapshot_indices):
                    points = self.points_history[idx]
                    label = f"Snapshot {idx}"
                    # Use different markers and colors for different snapshots
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=3,
                        alpha=0.5,
                        label=label,
                        marker=".",
                        edgecolors="none",
                    )

                # Add legend and labels
                ax.legend(fontsize=10)
                ax.set_xlabel("x", fontsize=14)
                ax.set_ylabel("t", fontsize=14)
                ax.set_xlim(x_domain)
                ax.set_ylim(t_domain)
                ax.grid(alpha=0.3)

                # Top-right: Initial distribution (heatmap)
                ax = axs[0, 1]
                initial_points = self.points_history[initial_idx]
                self._plot_density_heatmap(
                    ax,
                    initial_points,
                    colors["initial"],
                    "Initial Distribution",
                    x_domain,
                    t_domain,
                )

                # Bottom-left: Intermediate distribution (heatmap)
                ax = axs[1, 0]
                mid_points = self.points_history[mid_idx]
                self._plot_density_heatmap(
                    ax,
                    mid_points,
                    colors["mid"],
                    f"Intermediate Distribution (Snapshot {mid_idx})",
                    x_domain,
                    t_domain,
                )

                # Bottom-right: Final distribution (heatmap)
                ax = axs[1, 1]
                final_points = self.points_history[final_idx]
                self._plot_density_heatmap(
                    ax,
                    final_points,
                    colors["final"],
                    "Final Distribution",
                    x_domain,
                    t_domain,
                )

            else:  # 2D case
                # Similar structure but with 3D plots for spatial+time dimensions
                logging.warning(
                    "2D visualization not fully implemented, showing basic scatter plots"
                )

                # Simplified version for 2D: Just show scatter plots
                x_domain = self.pde.domain[0]
                y_domain = self.pde.domain[1]
                t_domain = self.pde.config.time_domain

                # Determine indices for snapshots
                num_snapshots = len(self.points_history)
                initial_idx = 0
                mid_idx = num_snapshots // 2
                final_idx = num_snapshots - 1

                # Top-left: Multiple snapshots
                ax = axs[0, 0]
                ax.set_title("Progression of Points (x-y projection)", fontsize=16)

                snapshot_indices = []
                if num_snapshots >= 5:
                    step = num_snapshots // 5
                    snapshot_indices = [i * step for i in range(5)]
                    if snapshot_indices[-1] != num_snapshots - 1:
                        snapshot_indices[-1] = num_snapshots - 1
                else:
                    snapshot_indices = list(range(num_snapshots))

                for i, idx in enumerate(snapshot_indices):
                    points = self.points_history[idx]
                    label = f"Snapshot {idx}"
                    ax.scatter(
                        points[:, 0],
                        points[:, 1],
                        s=3,
                        alpha=0.5,
                        label=label,
                        marker=".",
                        edgecolors="none",
                    )

                ax.legend(fontsize=10)
                ax.set_xlabel("x", fontsize=14)
                ax.set_ylabel("y", fontsize=14)
                ax.set_xlim(x_domain)
                ax.set_ylim(y_domain)
                ax.grid(alpha=0.3)

                # Other three plots: show density plots
                # Top-right: Initial
                ax = axs[0, 1]
                initial_points = self.points_history[initial_idx]
                self._plot_density_heatmap_2d(
                    ax,
                    initial_points,
                    colors["initial"],
                    "Initial Distribution",
                    x_domain,
                    y_domain,
                )

                # Bottom-left: Mid-training
                ax = axs[1, 0]
                mid_points = self.points_history[mid_idx]
                self._plot_density_heatmap_2d(
                    ax,
                    mid_points,
                    colors["mid"],
                    f"Intermediate Distribution (Snapshot {mid_idx})",
                    x_domain,
                    y_domain,
                )

                # Bottom-right: Final
                ax = axs[1, 1]
                final_points = self.points_history[final_idx]
                self._plot_density_heatmap_2d(
                    ax,
                    final_points,
                    colors["final"],
                    "Final Distribution",
                    x_domain,
                    y_domain,
                )

            # Adjust layout and save
            plt.tight_layout()

            # Save the figure
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches="tight")
                plt.close()
                logging.info(f"Collocation evolution saved to {save_path}")
            else:
                plt.show()

        except Exception as e:
            logging.warning(f"Error visualizing collocation evolution: {e}")
            import traceback

            traceback.print_exc()

    def _plot_density_heatmap(
        self, ax, points, colormap, title, x_domain, t_domain, bins=50
    ):
        """Helper method to plot density heatmap for 1D PDE collocation points"""
        # Create 2D histogram
        counts, xedges, yedges = np.histogram2d(
            points[:, 0], points[:, 1], bins=bins, range=[x_domain, t_domain]
        )

        # Display heatmap
        im = ax.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            extent=[x_domain[0], x_domain[1], t_domain[0], t_domain[1]],
            cmap=colormap,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of Points", fontsize=12)

        # Set labels and title
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("t", fontsize=14)

    def _plot_density_heatmap_2d(
        self, ax, points, colormap, title, x_domain, y_domain, bins=50
    ):
        """Helper method to plot density heatmap for 2D PDE collocation points"""
        # Create 2D histogram (using x and y coordinates)
        counts, xedges, yedges = np.histogram2d(
            points[:, 0], points[:, 1], bins=bins, range=[x_domain, y_domain]
        )

        # Display heatmap
        im = ax.imshow(
            counts.T,
            origin="lower",
            aspect="auto",
            extent=[x_domain[0], x_domain[1], y_domain[0], y_domain[1]],
            cmap=colormap,
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Number of Points", fontsize=12)

        # Set labels and title
        ax.set_title(title, fontsize=16)
        ax.set_xlabel("x", fontsize=14)
        ax.set_ylabel("y", fontsize=14)

    def generate_fdm_comparison(self, experiment_dir):
        """
        Generate comparison plots with finite difference method solutions.

        Args:
            experiment_dir: Directory to save visualizations
        """
        try:
            # Import required components
            from src.numerical_solvers.heat_equation_fdm import HeatEquationFDM

            # Check if the PDE type is supported
            pde_type = getattr(self.pde, "pde_type", None)
            if pde_type is None and hasattr(self.pde, "config"):
                pde_type = getattr(self.pde.config, "type", None)

            if pde_type == "heat":
                # Create visualization directory
                viz_dir = os.path.join(experiment_dir, "visualizations")
                os.makedirs(viz_dir, exist_ok=True)

                # Generate FDM comparison plots
                metrics = HeatEquationFDM.generate_fdm_comparison_plots(
                    pde=self.pde,
                    model=self.model,
                    device=self.device,
                    viz_dir=viz_dir,
                    logger=self.logger
                )

                if metrics:
                    self.logger.info("FDM comparison plots generated successfully")
                    self.logger.info(f"Error metrics: {metrics}")
                else:
                    self.logger.warning("Failed to generate FDM comparison plots")
            else:
                self.logger.info(f"FDM comparison not supported for PDE type: {pde_type}")

        except ImportError as e:
            self.logger.warning(f"Could not import required packages for FDM comparison: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error generating FDM comparison: {str(e)}")
