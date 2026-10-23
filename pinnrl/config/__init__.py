import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

DEFAULT_CONFIG_PATH = str(Path(__file__).parent / "config.yaml")


def resolve_device(name: str) -> torch.device:
    """Return ``torch.device(name)`` if that backend is available, else CPU.

    Only ``"cuda"`` and ``"mps"`` are probed; any other name resolves to CPU,
    so a config written on one machine still runs on another.
    """
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass
class LearningRateSchedulerConfig:
    """Learning rate scheduler settings.

    Attributes:
        type: Scheduler type (``"cosine"`` or ``"reduce_lr"``).
        warmup_epochs: Epochs of linear LR warm-up before the scheduler starts.
        min_lr: Minimum learning rate.
        factor: Multiplicative factor for ``reduce_lr`` scheduler.
        patience: Epochs to wait before reducing LR.
    """

    type: str
    warmup_epochs: int
    min_lr: float
    factor: float
    patience: int


@dataclass
class EarlyStoppingConfig:
    """Early-stopping criteria for training.

    Attributes:
        enabled: Whether early stopping is active.
        patience: Epochs without improvement before stopping.
        min_delta: Minimum loss decrease to count as improvement.
    """

    enabled: bool
    patience: int
    min_delta: float


@dataclass
class LBFGSConfig:
    """L-BFGS optimizer settings.

    Attributes:
        history_size: Number of past gradients/updates retained for the
            inverse-Hessian approximation.
        max_iter: Max iterations of the inner line search per ``step()`` call.
        line_search_fn: Line-search strategy passed to ``torch.optim.LBFGS``
            (``"strong_wolfe"`` or ``None``).
        tolerance_grad: First-order optimality tolerance.
        tolerance_change: Loss-change tolerance for early termination.
    """

    history_size: int = 50
    max_iter: int = 20
    line_search_fn: Optional[str] = "strong_wolfe"
    tolerance_grad: float = 1e-7
    tolerance_change: float = 1e-9


_LBFGS_FIELDS = tuple(LBFGSConfig.__dataclass_fields__)


def _lbfgs_value(key: str, value: Any) -> Any:
    """Coerce YAML L-BFGS settings (``1e-7`` parses as a string) to their field types."""
    if key in ("tolerance_grad", "tolerance_change"):
        return float(value)
    if key in ("history_size", "max_iter"):
        return int(value)
    return value


@dataclass
class AdaptiveWeightsConfig:
    """Configuration for adaptive loss weighting.

    Attributes:
        enabled: Whether adaptive weighting is active.
        strategy: Weight adaptation strategy (``"lrw"`` or ``"rbw"``).
        alpha: Moving average factor (0–1).
        eps: Small constant for numerical stability.
        initial_weights: Initial weights for [residual, boundary, initial] components.
    """

    enabled: bool = False
    strategy: str = "rbw"  # Options: "lrw" or "rbw"
    alpha: float = 0.9  # Moving average factor (0-1)
    eps: float = 1e-5  # Small constant for numerical stability
    initial_weights: List[float] = None  # Initial weights for [pde, boundary, initial] components

    def __post_init__(self):
        if self.initial_weights is None:
            self.initial_weights = [0.5, 0.3, 0.2]  # Default initial weights


@dataclass
class TrainingConfig:
    """Training hyper-parameters and sub-configurations.

    Attributes:
        num_epochs: Total training epochs.
        batch_size: Mini-batch size.
        num_collocation_points: Number of interior collocation points.
        num_boundary_points: Number of boundary condition points.
        num_initial_points: Number of initial condition points.
        learning_rate: Optimizer learning rate.
        weight_decay: L2 regularisation coefficient.
        gradient_clipping: Max gradient norm for clipping.
        early_stopping: Early-stopping configuration.
        learning_rate_scheduler: LR scheduler configuration.
        collocation_distribution: Sampling distribution for collocation points.
        adaptive_weights: Adaptive loss weighting configuration.
        loss_weights: Static loss component weights.
        validation_frequency: Epochs between validation, logging and run checkpoints.
    """

    num_epochs: int
    batch_size: int
    num_collocation_points: int
    num_boundary_points: int
    num_initial_points: int
    learning_rate: float
    weight_decay: float
    gradient_clipping: float
    early_stopping: EarlyStoppingConfig
    learning_rate_scheduler: LearningRateSchedulerConfig
    collocation_distribution: str = "uniform"
    adaptive_weights: AdaptiveWeightsConfig = None
    loss_weights: Dict[str, float] = None
    optimizer: str = "adam"  # "adam" | "lbfgs" | "adam_lbfgs"
    adam_lbfgs_switch_ratio: float = 0.7
    lbfgs: Optional[LBFGSConfig] = None
    mode: str = "forward"  # "forward" | "inverse" | "data_only" | "data_augmented"
    loss_function: str = "mse"  # "mse" | "mae" | "huber"
    huber_delta: float = 1.0
    validation_frequency: int = 10  # validate, log and checkpoint every N epochs

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {"residual": 1.0, "boundary": 1.0, "initial": 1.0}
        if "data" not in self.loss_weights:
            self.loss_weights["data"] = 1.0
        if self.adaptive_weights is None:
            self.adaptive_weights = AdaptiveWeightsConfig()
        if self.lbfgs is None:
            self.lbfgs = LBFGSConfig()
        if self.optimizer not in ("adam", "lbfgs", "adam_lbfgs"):
            raise ValueError(
                f"Invalid optimizer '{self.optimizer}'. "
                "Choose from 'adam', 'lbfgs', or 'adam_lbfgs'."
            )
        if self.mode not in ("forward", "inverse", "data_only", "data_augmented"):
            raise ValueError(
                f"Invalid mode '{self.mode}'. Choose 'forward', 'inverse', "
                "'data_only', or 'data_augmented'."
            )
        if self.loss_function not in ("mse", "mae", "huber"):
            raise ValueError(
                f"Invalid loss_function '{self.loss_function}'. " "Choose 'mse', 'mae', or 'huber'."
            )

    @property
    def optimizer_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        return {"learning_rate": self.learning_rate, "weight_decay": self.weight_decay}

    def __getitem__(self, key: str) -> Any:
        """Get the value for a given key, also to make it subscriptible."""
        if key == "optimizer_config":
            return self.optimizer_config
        return getattr(self, key)

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value for a given key or return the default value if the key doesn't exist."""
        if key == "optimizer_config":
            return self.optimizer_config
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "TrainingConfig":
        """Build from the ``training:`` block of ``config.yaml``.

        This is the one place the YAML layout is translated into a
        ``TrainingConfig``; both :class:`Config` and ``pinnrl-train`` use it.
        Missing keys fall back to defaults. Numeric fields are coerced with
        ``float()`` because YAML 1.1 reads ``1e-7`` (no decimal point) as a string.
        """
        optimizer_cfg = cfg.get("optimizer_config", {})
        early_stopping = cfg.get("early_stopping", {})
        reduce_lr = cfg.get("reduce_lr_params", {})
        adaptive = cfg.get("adaptive_weights", {})
        lbfgs = cfg.get("lbfgs", {})

        loss_weights = cfg.get("loss_weights")
        if loss_weights is not None and "pde" in loss_weights:
            loss_weights = dict(loss_weights)
            loss_weights["residual"] = loss_weights.pop("pde")

        return cls(
            num_epochs=cfg.get("num_epochs", 10000),
            batch_size=cfg.get("batch_size", 128),
            num_collocation_points=cfg.get("num_collocation_points", 1000),
            num_boundary_points=cfg.get("num_boundary_points", 100),
            num_initial_points=cfg.get("num_initial_points", 100),
            learning_rate=float(
                optimizer_cfg.get("learning_rate", cfg.get("learning_rate", 0.001))
            ),
            weight_decay=float(optimizer_cfg.get("weight_decay", cfg.get("weight_decay", 0.0001))),
            gradient_clipping=float(cfg.get("gradient_clipping", 1.0)),
            early_stopping=EarlyStoppingConfig(
                enabled=early_stopping.get("enabled", True),
                patience=early_stopping.get("patience", 100),
                min_delta=float(early_stopping.get("min_delta", 1e-4)),
            ),
            learning_rate_scheduler=LearningRateSchedulerConfig(
                type=cfg.get("scheduler_type", "cosine"),
                warmup_epochs=cfg.get("warmup_epochs", 0),
                min_lr=float(reduce_lr.get("min_lr", 1e-6)),
                factor=float(reduce_lr.get("factor", 0.5)),
                patience=reduce_lr.get("patience", 50),
            ),
            collocation_distribution=cfg.get("collocation_distribution", "uniform"),
            adaptive_weights=AdaptiveWeightsConfig(
                enabled=adaptive.get("enabled", False),
                strategy=adaptive.get("strategy", "rbw"),
                alpha=float(adaptive.get("alpha", 0.9)),
                eps=float(adaptive.get("eps", 1e-5)),
                initial_weights=adaptive.get("initial_weights"),
            ),
            loss_weights=loss_weights,
            optimizer=cfg.get("optimizer", "adam"),
            adam_lbfgs_switch_ratio=float(cfg.get("adam_lbfgs_switch_ratio", 0.7)),
            lbfgs=LBFGSConfig(
                **{k: _lbfgs_value(k, v) for k, v in lbfgs.items() if k in _LBFGS_FIELDS}
            ),
            mode=cfg.get("mode", "forward"),
            loss_function=cfg.get("loss_function", "mse"),
            huber_delta=float(cfg.get("huber_delta", 1.0)),
            validation_frequency=int(cfg.get("validation_frequency", 10)),
        )


@dataclass
class ModelConfig:
    """Configuration for neural network models.

    Attributes:
        input_dim: Number of input features (e.g. 2 for (x, t)).
        hidden_dim: Width of hidden layers.
        output_dim: Number of output features.
        num_layers: Depth of the network.
        activation: Activation function name.
        fourier_features: Whether to use Fourier feature mapping.
        fourier_scale: Scale parameter for Fourier features.
        dropout: Dropout rate (0.0 disables).
        layer_norm: Whether to apply layer normalisation.
        architecture: Architecture backend name.
        hidden_dims: Per-layer widths (overrides ``hidden_dim``).
        omega_0: Frequency factor for SIREN networks.
        num_blocks: Number of residual / FNO blocks.
        num_heads: Number of attention heads.
        latent_dim: Latent dimension for the AutoEncoder.
        mapping_size: Size of the Fourier feature mapping.
        scale: Scale for Fourier feature mapping.
        modes: Number of Fourier modes for FNO.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_layers: int
    activation: str
    fourier_features: bool
    fourier_scale: Optional[float]
    dropout: float
    layer_norm: bool
    architecture: str
    hidden_dims: Optional[List[int]] = None
    omega_0: Optional[float] = None  # For SIREN networks
    num_blocks: Optional[int] = None  # For ResNet
    num_heads: Optional[int] = None  # For attention networks
    latent_dim: Optional[int] = None  # For autoencoder
    mapping_size: int = 32  # Default mapping size for Fourier features
    scale: float = 10.0  # Default scale for Fourier features
    modes: Optional[int] = None  # For FNO (number of Fourier modes)

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: str,
        fourier_features: int = 0,
        fourier_scale: float = 1.0,
        dropout: float = 0.0,
        layer_norm: bool = False,
        architecture: str = "feedforward",
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.activation = activation
        self.fourier_features = fourier_features
        self.fourier_scale = fourier_scale
        self.dropout = dropout
        self.layer_norm = layer_norm
        self.architecture = architecture

        # Configure hidden_dims as a list based on hidden_dim when not explicitly set
        self.hidden_dims = [hidden_dim] * num_layers

        # For ResNet and FNO, set num_blocks to be the same as num_layers
        if architecture in ("resnet", "fno"):
            self.num_blocks = num_layers

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value for a given key or return the default value if the key doesn't exist."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Get the value for a given key, also to make it subscriptable."""
        return getattr(self, key)

    @classmethod
    def from_architecture(
        cls,
        architecture: str,
        arch_cfg: Dict[str, Any],
        input_dim: int,
        output_dim: int,
    ) -> "ModelConfig":
        """Build from one entry of the ``architectures:`` block of ``config.yaml``.

        ``hidden_dim`` and ``num_layers`` default to the first width and the
        length of ``hidden_dims`` when only the list is given. Architecture
        extras (``omega_0``, ``modes``, ...) are copied when present.
        """
        hidden_dims = arch_cfg.get("hidden_dims")
        model = cls(
            input_dim=input_dim,
            hidden_dim=arch_cfg.get("hidden_dim", hidden_dims[0] if hidden_dims else 128),
            output_dim=output_dim,
            num_layers=arch_cfg.get("num_layers", len(hidden_dims) if hidden_dims else 4),
            activation=arch_cfg.get("activation", "tanh"),
            fourier_features=architecture == "fourier",
            fourier_scale=arch_cfg.get("scale", 1.0),
            dropout=arch_cfg.get("dropout", 0.0),
            layer_norm=arch_cfg.get("layer_norm", False),
            architecture=architecture,
        )
        for key in _ARCHITECTURE_EXTRAS:
            if key in arch_cfg:
                setattr(model, key, arch_cfg[key])
        return model


_ARCHITECTURE_EXTRAS = (
    "hidden_dims",
    "mapping_size",
    "scale",
    "omega_0",
    "num_heads",
    "num_blocks",
    "latent_dim",
    "modes",
)


@dataclass
class RLConfig:
    """The ``rl:`` block: the experimental residual-bandit sampler (``--rl``).

    Sampler fields left as ``None`` fall back to the defaults of
    :class:`~pinnrl.rl.bandit_sampler.ResidualBanditSampler`, so those
    defaults live in one place.
    """

    enabled: bool = False
    n_x_cells: Optional[int] = None
    n_t_cells: Optional[int] = None
    policy_fraction: Optional[float] = None
    temperature: Optional[float] = None
    probe_cells: Optional[int] = None
    probe_points: Optional[int] = None
    probe_every: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> "RLConfig":
        """Build from the ``rl:`` block of ``config.yaml``; unknown keys are ignored."""
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})

    def build_sampler(self, pde, device: Optional[torch.device] = None):
        """Create the :class:`ResidualBanditSampler` these settings describe for ``pde``."""
        from pinnrl.rl.bandit_sampler import ResidualBanditSampler

        overrides = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name != "enabled" and getattr(self, name) is not None
        }
        return ResidualBanditSampler(
            tuple(pde.domain[0]),
            tuple(pde.time_domain),
            device=device or pde.device,
            **overrides,
        )


class Config:
    """Configuration for the PINNs-RL-PDE framework.

    Loads ``device``, ``model``, ``training`` and ``rl`` from ``config.yaml``.
    The model block is taken from the architecture configured for
    ``pde_type`` (falling back to the top-level ``model:`` block).
    """

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = DEFAULT_CONFIG_PATH
        self.config_path = config_path
        self.model = None
        self.training = None
        self.rl = None
        self.device = torch.device("cpu")  # Default to CPU, will be overridden by config

        # Load configuration if file exists
        if os.path.exists(config_path):
            self._load_config()
            self._validate_config()

    def get(self, key: str, default: Any = None) -> Any:
        """Get the value for a given key or return the default value if the key doesn't exist."""
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        """Get the value for a given key, also to make it subscriptable."""
        return getattr(self, key)

    def _load_config(self):
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        self.pde_type = config_dict.get("pde_type", "heat")
        pde_config = config_dict.get("pde_configs", {}).get(self.pde_type, {})
        model_config = config_dict.get("model", {})

        self.device = resolve_device(config_dict.get("device", "mps"))

        architecture = pde_config.get("architecture", model_config.get("architecture", "fourier"))
        arch_config = {**model_config, **config_dict.get("architectures", {}).get(architecture, {})}
        self.model = ModelConfig.from_architecture(
            architecture,
            arch_config,
            input_dim=pde_config.get("input_dim", model_config.get("input_dim", 2)),
            output_dim=pde_config.get("output_dim", model_config.get("output_dim", 1)),
        )

        self.training = TrainingConfig.from_dict(config_dict.get("training", {}))

        self.rl = RLConfig.from_dict(config_dict.get("rl", {}))

    def _validate_config(self):
        """Validate configuration parameters."""
        if self.model.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.model.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.model.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.model.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.model.activation not in ["tanh", "relu", "gelu", "leaky_relu", "sigmoid"]:
            raise ValueError(f"Invalid activation: {self.model.activation}")

        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
