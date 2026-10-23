"""``PDETrainer``: the single training loop (epoch-based ``train`` and step-based ``train_steps``)."""

import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from pinnrl import plots
from pinnrl.training.adaptive_weights import AdaptiveLossWeights

if TYPE_CHECKING:
    from pinnrl.config import Config
    from pinnrl.pdes.pde_base import PDEBase


def save_training_metrics(
    history: dict[str, Any],
    experiment_dir: str,
    metadata: dict[str, Any] | None = None,
):
    """Save training metrics to JSON files.

    Writes the same history to ``metrics.json`` and ``history.json``. When
    ``metadata`` is given it is merged into an existing ``metadata.json`` and
    stamped with ``last_updated``.

    Args:
        history: Dictionary containing training metrics
        experiment_dir: Directory to save the metrics
        metadata: Optional metadata about the experiment

    Returns:
        Path of ``history.json``.
    """
    # Create experiment directory if it doesn't exist
    os.makedirs(experiment_dir, exist_ok=True)

    # Convert NumPy arrays to Python lists in history
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_history = convert_to_serializable(history)

    # Save metrics.json directly in the experiment directory
    metrics_file = os.path.join(experiment_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(serializable_history, f, indent=2)

    # Save history.json directly in the experiment directory
    history_file = os.path.join(experiment_dir, "history.json")
    with open(history_file, "w") as f:
        json.dump(serializable_history, f, indent=2)

    # Save metadata if provided
    if metadata:
        metadata_file = os.path.join(experiment_dir, "metadata.json")

        # Update existing metadata if it exists
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file) as f:
                    existing_metadata = json.load(f)
                # Update with new metadata
                existing_metadata.update(metadata)
                metadata = existing_metadata
            except Exception:
                pass

        # Add timestamp
        metadata["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Convert metadata to serializable format
        serializable_metadata = convert_to_serializable(metadata)

        with open(metadata_file, "w") as f:
            json.dump(serializable_metadata, f, indent=2)

    return history_file


def _as_float(value):
    return value.item() if isinstance(value, torch.Tensor) else value


class PDETrainer:
    """Trainer for Physics-Informed Neural Networks."""

    def __init__(
        self,
        model: nn.Module,
        pde: "PDEBase",
        config: "Config",
        device: torch.device | None = None,
        bandit_sampler=None,
    ):
        """Initialize trainer.

        Args:
            model: PINN model.
            pde: PDE instance.
            config: Full configuration object. Optimizer, scheduler, early
                stopping and loss settings all come from ``config.training``.
            device: Device to train on.
            bandit_sampler: Optional ``ResidualBanditSampler``. When given (or when
                ``collocation_distribution`` is ``"adaptive"``) collocation points
                come from the experimental RL bandit sampler.
        """
        # An explicit device wins, then config.device, then MPS if available, else CPU
        self.device = device or (
            config.device
            if hasattr(config, "device")
            else torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        )

        self.model = model.to(self.device)
        self.pde = pde
        self.config = config
        self.validation_frequency = config.training.validation_frequency

        # Setup logging up-front so optimizer init can warn through self.logger.
        self._setup_logging()

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

        es_cfg = config.training.early_stopping
        self.early_stopping_enabled = es_cfg.enabled
        self.patience = es_cfg.patience
        self.min_delta = float(es_cfg.min_delta)
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        self.use_rl = (
            bandit_sampler is not None
            or getattr(config.training, "collocation_distribution", None) == "adaptive"
        )
        if bandit_sampler is not None:
            pde.bandit_sampler = bandit_sampler

        # Initialize adaptive weights handler if enabled
        self.use_adaptive_weights = config.training.adaptive_weights.enabled
        if self.use_adaptive_weights:
            self.adaptive_weights = AdaptiveLossWeights(
                strategy=config.training.adaptive_weights.strategy,
                alpha=config.training.adaptive_weights.alpha,
                eps=config.training.adaptive_weights.eps,
                initial_weights=config.training.adaptive_weights.initial_weights,
            )
        else:
            self.logger.info("Adaptive weights are disabled")
            self.adaptive_weights = None

    def _setup_logging(self):
        """Use the module logger; console output is configured by the application.

        A library must not configure the root logger: ``pinnrl-train`` sets up
        console logging in its entry point, and notebooks stay quiet unless the
        user enables ``logging.getLogger("pinnrl")``.
        """
        self.logger = logging.getLogger(__name__)

    def setup_experiment_logging(self, experiment_dir=None):
        """Log to ``<experiment_dir>/experiment.log``, replacing earlier file handlers.

        Args:
            experiment_dir: Directory for the experiment; ``None`` is a no-op.
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
            file_handler.setLevel(logging.INFO)
            self.logger.addHandler(file_handler)
            # The run log records INFO even when the application logs at WARNING.
            if self.logger.getEffectiveLevel() > logging.INFO:
                self.logger.setLevel(logging.INFO)
            self.logger.info(f"Experiment logs will be saved to {log_file}")

    def _compute_validation_loss(self, num_points: int = 1000) -> dict[str, float]:
        """Compute the loss on freshly drawn uniform collocation points.

        ``total_loss`` is ``PDEBase.compute_loss``'s total: static
        ``loss_weights``, or unit physics weights when adaptive weighting is on.
        Leaves the model in eval mode.

        Args:
            num_points: Number of validation points.

        Returns:
            ``{"total_loss", "residual_loss", "boundary_loss", "initial_loss"}`` as floats.
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
        """Step the LR scheduler.

        ReduceLROnPlateau gets ``val_loss`` (``train`` passes the epoch-mean
        training loss), or the last recorded train loss when it is ``None``.
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            if val_loss is None:
                val_loss = self.history["train_loss"][-1]
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _warmup_epochs(self) -> int:
        sched_cfg = self.config.training.learning_rate_scheduler
        return 0 if sched_cfg is None else int(sched_cfg.warmup_epochs or 0)

    def _apply_warmup(self, epoch: int) -> None:
        """Linear LR warm-up over the first ``warmup_epochs`` epochs (first-order phase only)."""
        warmup = self._warmup_epochs()
        if warmup > 0 and epoch < warmup and not getattr(self, "_is_lbfgs", False):
            lr = self.config.training.learning_rate * (epoch + 1) / warmup
            for group in self.optimizer.param_groups:
                group["lr"] = lr

    def _save_live_snapshot(
        self,
        experiment_dir: str,
        epoch: int,
        grid_size: int = 60,
    ) -> None:
        """Persist a small grid of predictions + residuals for the live dashboard.

        Writes ``live_snapshot.npz`` with predicted ``u`` (output channel 0) and
        residual fields on a ``grid_size``² grid: ``(x, t)`` for 1-D PDEs,
        ``(x1, x2)`` at the mid time for 2-D. The dashboard's Monitor sub-tab polls this
        file via the existing ``dcc.Interval`` and renders ``go.Surface``
        plots, giving the user a live view of the solution as it converges.

        Wrapped in a broad ``except`` because viz failures must never crash
        the training loop.
        """
        if not experiment_dir:
            return
        try:
            dim = int(getattr(self.pde, "dimension", 1))
            time_lo, time_hi = float(self.pde.time_domain[0]), float(self.pde.time_domain[1])
            self.model.eval()

            if dim <= 1:
                x_lo, x_hi = float(self.pde.domain[0][0]), float(self.pde.domain[0][1])
                xs = np.linspace(x_lo, x_hi, grid_size, dtype=np.float32)
                ts = np.linspace(time_lo, time_hi, grid_size, dtype=np.float32)
                xx, tt = np.meshgrid(xs, ts, indexing="xy")
                x_flat = torch.tensor(xx.reshape(-1, 1), device=self.device)
                t_flat = torch.tensor(tt.reshape(-1, 1), device=self.device)
                with torch.no_grad():
                    u_pred = self.model(torch.cat([x_flat, t_flat], dim=1))
                # Multi-channel outputs (dataset modes) collapse to channel 0
                # for the dashboard surface plot.
                u_pred_np = u_pred.detach().cpu().numpy()
                if u_pred_np.ndim == 2 and u_pred_np.shape[-1] > 1:
                    u_pred_np = u_pred_np[..., 0]
                u_pred_np = u_pred_np.reshape(grid_size, grid_size)

                # Residual needs grad → enable then detach.
                x_g = x_flat.detach().clone().requires_grad_(True)
                t_g = t_flat.detach().clone().requires_grad_(True)
                try:
                    residual = self.pde.compute_residual(self.model, x_g, t_g)
                    residual_np = residual.detach().cpu().numpy().reshape(grid_size, grid_size)
                except Exception:
                    residual_np = np.zeros_like(u_pred_np)

                np.savez(
                    os.path.join(experiment_dir, "live_snapshot.npz"),
                    axis_x=xs,
                    axis_y=ts,
                    u_pred=u_pred_np,
                    residual=residual_np,
                    epoch=int(epoch),
                    dimension=1,
                    x_label="x",
                    y_label="t",
                    fixed_t=float("nan"),
                )
            else:
                x1_lo, x1_hi = float(self.pde.domain[0][0]), float(self.pde.domain[0][1])
                x2_lo, x2_hi = float(self.pde.domain[1][0]), float(self.pde.domain[1][1])
                fixed_t = 0.5 * (time_lo + time_hi)
                xs1 = np.linspace(x1_lo, x1_hi, grid_size, dtype=np.float32)
                xs2 = np.linspace(x2_lo, x2_hi, grid_size, dtype=np.float32)
                xx1, xx2 = np.meshgrid(xs1, xs2, indexing="xy")
                x_flat = torch.tensor(
                    np.stack([xx1.reshape(-1), xx2.reshape(-1)], axis=1),
                    device=self.device,
                    dtype=torch.float32,
                )
                t_flat = torch.full(
                    (x_flat.shape[0], 1), fixed_t, dtype=torch.float32, device=self.device
                )
                with torch.no_grad():
                    u_pred = self.model(torch.cat([x_flat, t_flat], dim=1))
                u_pred_np = u_pred.detach().cpu().numpy()
                if u_pred_np.ndim == 2 and u_pred_np.shape[-1] > 1:
                    u_pred_np = u_pred_np[..., 0]
                u_pred_np = u_pred_np.reshape(grid_size, grid_size)

                x_g = x_flat.detach().clone().requires_grad_(True)
                t_g = t_flat.detach().clone().requires_grad_(True)
                try:
                    residual = self.pde.compute_residual(self.model, x_g, t_g)
                    residual_np = residual.detach().cpu().numpy()
                    if residual_np.ndim == 2 and residual_np.shape[-1] > 1:
                        residual_np = residual_np[..., 0]
                    residual_np = residual_np.reshape(grid_size, grid_size)
                except Exception:
                    residual_np = np.zeros_like(u_pred_np)

                np.savez(
                    os.path.join(experiment_dir, "live_snapshot.npz"),
                    axis_x=xs1,
                    axis_y=xs2,
                    u_pred=u_pred_np,
                    residual=residual_np,
                    epoch=int(epoch),
                    dimension=2,
                    x_label="x1",
                    y_label="x2",
                    fixed_t=float(fixed_t),
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.debug(f"Live snapshot skipped: {exc}")
        finally:
            self.model.train()

    def _collect_optimizable_params(self):
        """Combine model weights with any trainable PDE parameters (inverse mode).

        In forward mode the PDE contributes nothing and the result is identical
        to ``self.model.parameters()``.
        """
        params = list(self.model.parameters())
        if hasattr(self.pde, "trainable_parameters_iter"):
            params += list(self.pde.trainable_parameters_iter())
        return params

    def _build_adam(self, params):
        return optim.Adam(
            params,
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )

    def _build_lbfgs(self, params):
        cfg = self.config.training.lbfgs
        return optim.LBFGS(
            params,
            lr=self.config.training.learning_rate,
            history_size=cfg.history_size,
            max_iter=cfg.max_iter,
            line_search_fn=cfg.line_search_fn,
            tolerance_grad=cfg.tolerance_grad,
            tolerance_change=cfg.tolerance_change,
        )

    def _build_scheduler(self, force_reduce_lr: bool = False):
        """Build the LR scheduler; ``None`` means a constant learning rate.

        ``cosine`` anneals over ``num_epochs - warmup_epochs`` down to ``min_lr``;
        ``reduce_lr`` is ReduceLROnPlateau. ``force_reduce_lr`` (L-BFGS) turns
        any non-constant type into ReduceLROnPlateau.
        """
        sched_cfg = self.config.training.learning_rate_scheduler
        scheduler_type = "constant" if sched_cfg is None else sched_cfg.type
        if scheduler_type == "constant":
            return None
        if force_reduce_lr or scheduler_type == "reduce_lr":
            if force_reduce_lr and scheduler_type != "reduce_lr":
                self.logger.warning(
                    "L-BFGS uses an internal line search; overriding scheduler "
                    f"'{scheduler_type}' with ReduceLROnPlateau."
                )
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.config.training.learning_rate_scheduler.factor,
                patience=self.config.training.learning_rate_scheduler.patience,
                min_lr=self.config.training.learning_rate_scheduler.min_lr,
            )
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.config.training.num_epochs - self._warmup_epochs()),
                eta_min=self.config.training.learning_rate_scheduler.min_lr,
            )
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    def _initialize_optimizer_and_scheduler(self):
        """Initialize optimizer and learning rate scheduler.

        Branches on ``training.optimizer``:
          - ``adam``       : standard first-order training.
          - ``lbfgs``      : closure-based quasi-Newton training (full batch).
          - ``adam_lbfgs`` : Adam first, switch to L-BFGS after epoch
            ``int(adam_lbfgs_switch_ratio * config.training.num_epochs)``.

        Trainable PDE parameters (inverse mode) are added to the optimizer
        alongside the model weights. While L-BFGS runs, a non-constant
        scheduler is replaced by ReduceLROnPlateau.
        """
        optimizer_type = getattr(self.config.training, "optimizer", "adam")
        params = self._collect_optimizable_params()

        if optimizer_type == "lbfgs":
            self.optimizer = self._build_lbfgs(params)
            self._is_lbfgs = True
        else:
            # Both "adam" and "adam_lbfgs" start with Adam.
            self.optimizer = self._build_adam(params)
            self._is_lbfgs = False

        if optimizer_type == "adam_lbfgs":
            ratio = getattr(self.config.training, "adam_lbfgs_switch_ratio", 0.7)
            self._switch_epoch = max(1, int(self.config.training.num_epochs * ratio))
        else:
            self._switch_epoch = None

        self.scheduler = self._build_scheduler(force_reduce_lr=self._is_lbfgs)
        self._optimizer_type = optimizer_type

    def _switch_to_lbfgs(self):
        """Hot-swap the optimizer to L-BFGS for the second phase of adam_lbfgs."""
        params = self._collect_optimizable_params()
        self.optimizer = self._build_lbfgs(params)
        self._is_lbfgs = True
        self.scheduler = self._build_scheduler(force_reduce_lr=True)

    def _lbfgs_step(self, x_batch, t_batch):
        """One L-BFGS optimizer step using a recompute-loss closure."""
        captured: dict[str, torch.Tensor] = {}

        def closure():
            self.optimizer.zero_grad()
            losses = self.pde.compute_loss(self.model, x_batch, t_batch)
            total = losses["total"]
            total.backward()
            captured["losses"] = losses
            return total

        self.optimizer.step(closure)
        if "losses" not in captured:
            # Defensive: torch's LBFGS.step always evaluates the closure at least once.
            captured["losses"] = self.pde.compute_loss(self.model, x_batch, t_batch)
        return captured["losses"]

    def _sample_batch(self, batch_size: int, progress: float):
        """Draw one batch of collocation points with the configured strategy.

        ``progress`` in [0, 1] is passed to the adaptive (bandit) sampler.
        """
        # Generate batch of collocation points; adaptive samplers need the model
        sampling_strategy = (
            "adaptive" if self.use_rl else self.config.training.collocation_distribution
        )
        sampling_kwargs = {}
        if sampling_strategy in ("residual_based", "adaptive"):
            sampling_kwargs["model"] = self.model
        if sampling_strategy == "adaptive":
            sampling_kwargs["progress"] = progress
        x_batch, t_batch = self.pde.generate_collocation_points(
            batch_size, strategy=sampling_strategy, **sampling_kwargs
        )
        return x_batch.to(self.device), t_batch.to(self.device)

    def _adaptive_total(self, losses):
        """Replace ``losses["total"]`` with the adaptively weighted objective."""
        # Adaptive weighting reweights the physics components only
        # (residual / boundary / initial); the data term keeps its static
        # ``loss_weights["data"]``. In ``data_only`` mode those
        # are zero contributions to the objective, so adaptive
        # weighting is a no-op and we trust the total assembled by
        # ``PDEBase.compute_loss`` (already ``data`` only).
        training_mode = self.config.training.mode
        if training_mode == "data_only":
            return
        component_names = ["residual", "boundary", "initial"]
        if self.config.training.adaptive_weights.strategy == "lrw":
            # Learning-rate weighting balances the gradient norm of each component
            # (over the model parameters, not trainable PDE parameters).
            # torch.autograd.grad leaves .grad untouched, so the total backward in
            # _step() sees only its own gradient; components without a graph count as 0.
            params = [p for p in self.model.parameters() if p.requires_grad]
            grad_norms = []
            for name in component_names:
                loss = losses[name]
                grad_norm = 0.0
                if loss.requires_grad:
                    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
                    grad_norm = sum(g.norm().item() ** 2 for g in grads if g is not None)
                grad_norms.append(torch.tensor(grad_norm**0.5, device=self.device))
            weights = self.adaptive_weights.update(gradients=torch.stack(grad_norms))
        else:
            # Relative error weighting balances the component values themselves.
            components = torch.stack([losses[name].detach() for name in component_names])
            weights = self.adaptive_weights.update(losses=components)

        losses["total"] = self.pde.weighted_total(losses, *weights[:3])

        self.logger.debug(
            "Adaptive weights (%s): %s",
            self.config.training.adaptive_weights.strategy,
            {name: float(weights[i]) for i, name in enumerate(component_names) if i < len(weights)},
        )

        # Store weights for visualization/tracking
        if isinstance(weights, torch.Tensor):
            weights_np = weights.detach().cpu().numpy()
        else:
            weights_np = np.array(weights)

        self.history["loss_weights"].append(weights_np)

    def _step(self, x_batch, t_batch):
        """One optimizer step on a batch; returns the loss components.

        Shared by :meth:`train` and :meth:`train_steps`. Applies adaptive loss
        weighting (if enabled) and gradient clipping of the model parameters, or
        an L-BFGS closure step (no adaptive weighting, no clipping).
        """
        if getattr(self, "_is_lbfgs", False):
            return self._lbfgs_step(x_batch, t_batch)
        self.optimizer.zero_grad()
        losses = self.pde.compute_loss(self.model, x_batch, t_batch)
        if self.use_adaptive_weights:
            self._adaptive_total(losses)
        losses["total"].backward()
        if self.config.training.gradient_clipping > 0:
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.training.gradient_clipping
            )
        self.optimizer.step()
        return losses

    def train_steps(
        self,
        num_steps: int | None = None,
        time_budget_s: float | None = None,
        sample_fn: Callable | None = None,
        resample_every: int = 1,
        checkpoints: Sequence[float] = (),
        on_checkpoint: Callable | None = None,
    ) -> dict[str, Any]:
        """Step-based training loop for benchmarks and studies.

        Runs :meth:`_step` (the same update as :meth:`train`) until
        ``num_steps`` steps or until training time reaches ``time_budget_s``.
        Training time is the summed wall-clock of sampling plus optimizer steps;
        checkpoint callbacks are excluded. There is no validation, early
        stopping, LR warm-up or scheduling, progress bar or disk I/O.

        Args:
            num_steps: Stop after this many steps.
            time_budget_s: Stop once training time reaches this many seconds.
            sample_fn: ``sample_fn(model, progress) -> (x, t)``, where ``progress``
                in [0, 1) is the elapsed fraction of the time budget (or of
                ``num_steps`` without one); defaults to the configured
                collocation strategy with ``config.training.batch_size``.
            resample_every: Draw new points every N steps and reuse them between.
            checkpoints: Increasing training-time marks (seconds).
            on_checkpoint: Called as ``on_checkpoint(step, train_time_s, model)``
                when training time first passes each mark.

        Returns:
            ``{"steps", "train_time_s", "loss"}`` with the per-step total loss.

        Raises:
            ValueError: If neither ``num_steps`` nor ``time_budget_s`` is given.
        """
        if num_steps is None and time_budget_s is None:
            raise ValueError("Pass num_steps, time_budget_s, or both.")
        if sample_fn is None:
            batch_size = self.config.training.batch_size

            def sample_fn(_model, progress):
                return self._sample_batch(batch_size, progress)

        pending = list(checkpoints)
        loss_curve = []
        clock = 0.0
        step = 0
        x_batch = t_batch = None
        self.model.train()
        while (num_steps is None or step < num_steps) and (
            time_budget_s is None or clock < time_budget_s
        ):
            if time_budget_s is not None:
                progress = clock / time_budget_s
            else:
                progress = step / num_steps
            start = time.perf_counter()
            if step % resample_every == 0:
                x_batch, t_batch = sample_fn(self.model, progress)
            losses = self._step(x_batch, t_batch)
            loss_value = float(losses["total"].item())  # syncs MPS/CUDA before the clock stops
            clock += time.perf_counter() - start
            step += 1
            loss_curve.append(loss_value)
            while pending and clock >= pending[0]:
                pending.pop(0)
                if on_checkpoint is not None:
                    on_checkpoint(step, clock, self.model)
                    self.model.train()
        return {"steps": step, "train_time_s": clock, "loss": loss_curve}

    def train(
        self,
        num_epochs: int | None = None,
        batch_size: int | None = None,
        num_points: int | None = None,
        experiment_dir: str | None = None,
        progress: bool = True,
    ):
        """Train the model.

        Each epoch takes ``max(1, num_points // batch_size)`` optimizer steps on
        freshly sampled batches (one full batch when ``optimizer`` is ``lbfgs``). Every
        ``validation_frequency`` epochs (starting at epoch 0) it validates,
        logs, checks early stopping (patience counted in epochs) and, with
        ``experiment_dir``, saves history, metadata and the live snapshot.

        Args:
            num_epochs: Defaults to ``config.training.num_epochs``.
            batch_size: Defaults to ``config.training.batch_size``; capped at
                ``num_points``.
            num_points: Collocation points per epoch; defaults to
                ``config.training.num_collocation_points``.
            experiment_dir: Run folder for ``experiment.log``, ``metadata.json``,
                ``metrics.json``/``history.json``, ``live_snapshot.npz``,
                ``final_model.pt`` and ``visualizations/``; ``None`` writes nothing.
            progress: Show one tqdm progress bar over epochs.

        Returns:
            The training history dict (``self.history``).
        """
        training_cfg = self.config.training
        num_epochs = training_cfg.num_epochs if num_epochs is None else num_epochs
        batch_size = training_cfg.batch_size if batch_size is None else batch_size
        num_points = training_cfg.num_collocation_points if num_points is None else num_points
        if num_points < batch_size:
            batch_size = num_points
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
            self.logger.info("ADAPTIVE WEIGHTS ENABLED")
            self.logger.info(f"  Strategy: {self.config.training.adaptive_weights.strategy}")
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

        # Inverse-problem warm-up: announce trainable parameters and seed history
        # buckets so the dashboard can render trajectories from epoch 0.
        trainable_pde_params = {}
        if hasattr(self.pde, "_trainable_params"):
            trainable_pde_params = dict(self.pde._trainable_params)
        if trainable_pde_params:
            self.logger.info(
                "INVERSE MODE — identifying %d PDE parameter(s): %s",
                len(trainable_pde_params),
                list(trainable_pde_params.keys()),
            )
            for name in trainable_pde_params:
                self.history.setdefault(f"param_{name}", [])

        # L-BFGS requires deterministic full-batch loss; override batch_size.
        # Only checked here, so the L-BFGS phase of adam_lbfgs keeps batch_size.
        if getattr(self, "_is_lbfgs", False) and batch_size != num_points:
            self.logger.warning(
                "L-BFGS optimizer requires full-batch updates; "
                f"overriding batch_size {batch_size} -> {num_points}."
            )
            batch_size = num_points

        # Adaptive weights are incompatible with the L-BFGS closure (per-component
        # gradient passes are too expensive); _step skips them during L-BFGS.
        if getattr(self, "_is_lbfgs", False) and self.use_adaptive_weights:
            self.logger.warning("Adaptive loss weighting is disabled while running L-BFGS.")

        # Record start time
        start_time = datetime.now()

        # Fields shared by the initial and the final metadata.json.
        run_info = {
            "pde_type": getattr(self.pde, "pde_type", type(self.pde).__name__),
            "pde_name": getattr(self.pde.config, "name", ""),
            "architecture": getattr(self.model, "architecture_name", type(self.model).__name__),
            "training_params": {
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "num_points": num_points,
                "validation_frequency": self.validation_frequency,
            },
            "collocation_distribution": (
                "adaptive" if self.use_rl else self.config.training.collocation_distribution
            ),
            "rl_enabled": self.use_rl,
            "optimizer": self._optimizer_type,
            "mode": self.config.training.mode,
            "trainable_parameters": list(trainable_pde_params.keys()),
            "true_parameters": dict(getattr(self.pde, "_true_parameters", {})),
        }

        # Create directories for visualizations and experiment data
        if experiment_dir:
            os.makedirs(experiment_dir, exist_ok=True)

            # Setup experiment-specific logging
            self.setup_experiment_logging(experiment_dir)

            # Create visualization directory
            viz_dir = os.path.join(experiment_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # Create .running marker file
            running_file = os.path.join(experiment_dir, ".running")
            open(running_file, "w").close()

            initial_metadata = {
                "status": "running",
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_epochs": num_epochs,
                "current_epoch": 0,
                **run_info,
            }
            metadata_path = os.path.join(experiment_dir, "metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(initial_metadata, f, indent=2)

        self.logger.info("Starting training...")

        # Initialize points history
        self.points_history = []

        # Emit an initial live snapshot so the Monitor tab has something to show
        # before the first epoch finishes.
        if experiment_dir:
            self._save_live_snapshot(experiment_dir, epoch=0)

        # tqdm.auto renders a widget in notebooks; refresh at most twice a second.
        pbar = tqdm(total=num_epochs, desc="Training", mininterval=0.5) if progress else None
        for epoch in range(num_epochs):
            self.model.train()
            self._apply_warmup(epoch)
            epoch_losses = []
            epoch_points = []  # Store all points for this epoch

            for _ in range(max(1, num_points // batch_size)):
                x_batch, t_batch = self._sample_batch(batch_size, epoch / max(1, num_epochs))
                epoch_points.append(torch.cat([x_batch, t_batch], dim=1).detach().cpu().numpy())
                losses = self._step(x_batch, t_batch)
                epoch_losses.append(float(losses["total"].item()))
            if pbar is not None:
                pbar.set_postfix(
                    loss=f"{epoch_losses[-1]:.3e}",
                    res=f"{losses['residual'].item():.3e}",
                    refresh=False,
                )
                pbar.update()

            # End of epoch
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            points_array = np.concatenate(epoch_points, axis=0)
            self.points_history.append(points_array)

            # Update learning rate (the scheduler takes over after warm-up)
            if epoch >= self._warmup_epochs():
                self._update_scheduler(avg_epoch_loss)

            # Log metrics
            current_lr = self.optimizer.param_groups[0]["lr"]

            epoch_data = {
                "epoch": epoch,
                "train_loss": avg_epoch_loss,
                "residual_loss": _as_float(losses["residual"]),
                "boundary_loss": _as_float(losses["boundary"]),
                "initial_loss": _as_float(losses["initial"]),
                "learning_rate": current_lr,
            }
            if "data" in losses:
                epoch_data["data_loss"] = _as_float(losses["data"])
                self.history.setdefault("data_loss", [])

            # Capture each trainable PDE parameter's current value so the
            # dashboard can plot its trajectory in real time.
            if trainable_pde_params:
                for name, p in trainable_pde_params.items():
                    epoch_data[f"param_{name}"] = float(p.detach().cpu().item())

            # Add weights info if using adaptive weights
            if self.use_adaptive_weights and len(self.history["loss_weights"]) > 0:
                current_weights = self.history["loss_weights"][-1]
                weights_str = ""
                component_names = ["residual", "boundary", "initial"]
                for i, name in enumerate(component_names):
                    if i < len(current_weights):
                        weights_str += f", {name}_weight={float(current_weights[i]):.4f}"

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
                self.logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, "
                    f"Train Loss: {avg_epoch_loss:.6f}, "
                    f"Val Loss: {val_losses['total_loss']:.6f}, "
                    f"LR: {current_lr:.6f}{weights_str}"
                )

                # Early stopping check
                if self.early_stopping_enabled:
                    val_loss = val_losses["total_loss"]
                    if val_loss < self.best_val_loss - self.min_delta:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                    else:
                        # Patience is in epochs; validation runs every validation_frequency.
                        self.patience_counter += self.validation_frequency

                    if self.patience_counter >= self.patience:
                        self.logger.info(
                            f"Early stopping triggered at epoch {epoch + 1}, "
                            f"best val loss: {self.best_val_loss:.6f}"
                        )
                        break

            # Periodic save of history and metadata
            if experiment_dir and epoch % self.validation_frequency == 0:
                save_training_metrics(self.history, experiment_dir)
                metadata_path = os.path.join(experiment_dir, "metadata.json")
                try:
                    with open(metadata_path) as f:
                        partial_metadata = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    partial_metadata = {}
                partial_metadata.update(
                    {
                        "current_epoch": epoch + 1,
                        "status": "running",
                        "final_loss": float(avg_epoch_loss),
                        "best_val_loss": (
                            float(self.best_val_loss)
                            if self.best_val_loss != float("inf")
                            else None
                        ),
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
                if trainable_pde_params:
                    partial_metadata["current_parameters"] = {
                        name: float(p.detach().cpu().item())
                        for name, p in trainable_pde_params.items()
                    }
                with open(metadata_path, "w") as f:
                    json.dump(partial_metadata, f, indent=2)
                # Refresh the live 3D snapshot for the dashboard.
                self._save_live_snapshot(experiment_dir, epoch=epoch + 1)

            # adam_lbfgs: hot-swap the optimizer once we cross the switch boundary.
            if (
                getattr(self, "_optimizer_type", "adam") == "adam_lbfgs"
                and not self._is_lbfgs
                and self._switch_epoch is not None
                and (epoch + 1) >= self._switch_epoch
            ):
                self.logger.info(f"Switching optimizer to L-BFGS at epoch {epoch + 1}")
                self._switch_to_lbfgs()
                if self.use_adaptive_weights:
                    self.logger.warning("Adaptive loss weighting is disabled while running L-BFGS.")

        if pbar is not None:
            pbar.close()

        # End of training
        train_time = (datetime.now() - start_time).total_seconds() / 60.0
        self.logger.info(f"Training completed in {train_time} minutes")

        # Save final visualizations and metrics
        if experiment_dir:
            # Create metadata with training information
            metadata = {
                **run_info,
                "status": "completed",
                "training_time_minutes": train_time,
                "total_epochs": num_epochs,
                "final_loss": (
                    float(self.history["train_loss"][-1]) if self.history["train_loss"] else None
                ),
                "best_val_loss": (
                    float(self.best_val_loss) if self.best_val_loss != float("inf") else None
                ),
                "early_stopping_triggered": (
                    self.patience_counter >= self.patience if self.early_stopping_enabled else False
                ),
                "current_epoch": len(self.history["train_loss"]),
                "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "domain": self.pde.domain,
                "time_domain": self.pde.time_domain,
                "boundary_conditions": self.pde.config.boundary_conditions,
                "initial_condition": self.pde.config.initial_condition,
                "pde_parameters": self.pde.config.parameters,
                "identified_parameters": (
                    self.pde.get_trainable_parameter_values() if trainable_pde_params else {}
                ),
            }

            # Save training history as metrics.json and history.json
            self.logger.info("Saving training metrics...")
            save_training_metrics(self.history, experiment_dir, metadata=metadata)

            # Final live snapshot reflects the trained model.
            self._save_live_snapshot(experiment_dir, epoch=len(self.history.get("train_loss", [])))

            # Remove .running marker
            running_file = os.path.join(experiment_dir, ".running")
            if os.path.exists(running_file):
                try:
                    os.remove(running_file)
                except OSError:
                    pass

            # Save the model directly in the experiment directory
            self.logger.info("Saving model...")
            model_path = os.path.join(experiment_dir, "final_model.pt")
            torch.save(self.model.state_dict(), model_path)

            # Save all plots using the save_plots method
            self.logger.info("Generating and saving all plots...")
            self.save_plots(viz_dir)

        return self.history

    def save_plots(self, save_dir):  # pragma: no cover
        """Write the post-training figures to ``save_dir``."""
        plots.save_training_plots(self, save_dir)
