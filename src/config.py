import os
import torch
import yaml
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LearningRateSchedulerConfig:
    type: str
    warmup_epochs: int
    min_lr: float
    factor: float
    patience: int


@dataclass
class EarlyStoppingConfig:
    enabled: bool
    patience: int
    min_delta: float


@dataclass
class TrainingConfig:
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


@dataclass
class ModelConfig:
    """Configuration for PINN model."""

    input_dim: int = 2
    hidden_dim: int = 128
    output_dim: int = 1
    num_layers: int = 4
    activation: str = "tanh"
    fourier_features: bool = True
    fourier_scale: float = 2.0
    dropout: float = 0.1
    layer_norm: bool = True
    architecture: str = "fourier"  # New parameter for architecture selection


@dataclass
class PDEConfig:
    domain: List[float]
    t_domain: List[float]
    initial_condition: str
    boundary_conditions: Dict[str, str]
    diffusion_coefficient: float
    source_term: str


@dataclass
class RLConfig:
    enabled: bool
    state_dim: int
    action_dim: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    epsilon_start: float
    epsilon_end: float
    epsilon_decay: float
    memory_size: int
    batch_size: int
    target_update: int
    reward_weights: Dict[str, float]


@dataclass
class EvaluationConfig:
    resolution: int
    num_test_points: int
    metrics: List[str]
    save_plots: bool
    plot_frequency: int


@dataclass
class LoggingConfig:
    level: str
    save_tensorboard: bool
    log_frequency: int


@dataclass
class PathsConfig:
    experiments_dir: str
    model_dir: str
    log_dir: str
    tensorboard_dir: str


class Config:
    """Configuration class for PINN training."""

    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize configuration from YAML file.

        :param config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self._load_config()
        self._validate_config()

    def _load_config(self):
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        # Device configuration
        self.device = self._get_device(config_dict.get("device", "mps"))

        # Model configuration
        model_config = config_dict.get("model", {})
        self.model = ModelConfig(
            input_dim=model_config.get("input_dim", 2),
            hidden_dim=model_config.get("hidden_dim", 128),
            output_dim=model_config.get("output_dim", 1),
            num_layers=model_config.get("num_layers", 4),
            activation=model_config.get("activation", "tanh"),
            fourier_features=model_config.get("fourier_features", True),
            fourier_scale=model_config.get("fourier_scale", 2.0),
            dropout=model_config.get("dropout", 0.1),
            layer_norm=model_config.get("layer_norm", True),
            architecture=model_config.get("architecture", "fourier"),  # New parameter
        )

        # PDE configuration
        pde_config = config_dict.get("pde", {})
        self.pde = PDEConfig(
            domain=pde_config.get("domain", [0.0, 1.0]),
            t_domain=pde_config.get("t_domain", [0.0, 1.0]),
            initial_condition=pde_config.get("initial_condition", "sin(pi*x)"),
            boundary_conditions=pde_config.get(
                "boundary_conditions", {"left": "0.0", "right": "0.0"}
            ),
            diffusion_coefficient=pde_config.get("diffusion_coefficient", 0.01),
            source_term=pde_config.get("source_term", "0.0"),
        )

        # Training configuration
        training_config = config_dict.get("training", {})
        early_stopping_config = training_config.get("early_stopping", {})
        lr_scheduler_config = training_config.get("learning_rate_scheduler", {})

        self.training = TrainingConfig(
            num_epochs=training_config.get("num_epochs", 10000),
            batch_size=training_config.get("batch_size", 128),
            num_collocation_points=training_config.get("num_collocation_points", 1000),
            num_boundary_points=training_config.get("num_boundary_points", 100),
            num_initial_points=training_config.get("num_initial_points", 100),
            learning_rate=training_config.get("learning_rate", 0.001),
            weight_decay=training_config.get("weight_decay", 0.0001),
            gradient_clipping=training_config.get("gradient_clipping", 1.0),
            early_stopping=EarlyStoppingConfig(
                enabled=early_stopping_config.get("enabled", True),
                patience=early_stopping_config.get("patience", 100),
                min_delta=early_stopping_config.get("min_delta", 1e-4),
            ),
            learning_rate_scheduler=LearningRateSchedulerConfig(
                type=lr_scheduler_config.get("type", "cosine"),
                warmup_epochs=lr_scheduler_config.get("warmup_epochs", 100),
                min_lr=lr_scheduler_config.get("min_lr", 1e-6),
                factor=lr_scheduler_config.get("factor", 0.5),
                patience=lr_scheduler_config.get("patience", 50),
            ),
        )

        # RL configuration
        rl_config = config_dict.get("rl", {})
        self.rl = RLConfig(
            enabled=rl_config.get("enabled", False),
            state_dim=rl_config.get("state_dim", 128),
            action_dim=rl_config.get("action_dim", 100),
            hidden_dim=rl_config.get("hidden_dim", 64),
            learning_rate=rl_config.get("learning_rate", 0.0001),
            gamma=rl_config.get("gamma", 0.99),
            epsilon_start=rl_config.get("epsilon_start", 1.0),
            epsilon_end=rl_config.get("epsilon_end", 0.01),
            epsilon_decay=rl_config.get("epsilon_decay", 0.995),
            memory_size=rl_config.get("memory_size", 10000),
            batch_size=rl_config.get("batch_size", 64),
            target_update=rl_config.get("target_update", 100),
            reward_weights=rl_config.get(
                "reward_weights",
                {"residual": 1.0, "boundary": 1.0, "initial": 1.0, "exploration": 0.1},
            ),
        )

        # Evaluation configuration
        eval_config = config_dict.get("evaluation", {})
        self.evaluation = EvaluationConfig(
            resolution=eval_config.get("resolution", 100),
            num_test_points=eval_config.get("num_test_points", 1000),
            metrics=eval_config.get("metrics", ["l2_error", "h1_error", "max_error"]),
            save_plots=eval_config.get("save_plots", True),
            plot_frequency=eval_config.get("plot_frequency", 100),
        )

        # Logging configuration
        logging_config = config_dict.get("logging", {})
        self.logging = LoggingConfig(
            level=logging_config.get("level", "INFO"),
            save_tensorboard=logging_config.get("save_tensorboard", True),
            log_frequency=logging_config.get("log_frequency", 100),
        )

        # Paths configuration
        paths_config = config_dict.get("paths", {})
        self.paths = PathsConfig(
            experiments_dir=paths_config.get("experiments_dir", "experiments"),
            model_dir=paths_config.get("model_dir", "models"),
            log_dir=paths_config.get("log_dir", "logs"),
            tensorboard_dir=paths_config.get("tensorboard_dir", "runs"),
        )

    def _validate_config(self):
        """Validate configuration parameters."""
        # Validate device
        if self.device not in ["cuda", "mps", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")

        # Validate model configuration
        if self.model.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.model.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.model.output_dim <= 0:
            raise ValueError("output_dim must be positive")
        if self.model.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.model.activation not in ["tanh", "relu", "gelu"]:
            raise ValueError(f"Invalid activation: {self.model.activation}")

        # Validate PDE configuration
        if len(self.pde.domain) != 2:
            raise ValueError("domain must be a list of two values")
        if len(self.pde.t_domain) != 2:
            raise ValueError("t_domain must be a list of two values")
        if self.pde.diffusion_coefficient <= 0:
            raise ValueError("diffusion_coefficient must be positive")

        # Validate training configuration
        if self.training.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.training.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.training.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        # Validate RL configuration if enabled
        if self.rl.enabled:
            if self.rl.state_dim <= 0:
                raise ValueError("state_dim must be positive")
            if self.rl.action_dim <= 0:
                raise ValueError("action_dim must be positive")
            if not 0 <= self.rl.gamma <= 1:
                raise ValueError("gamma must be between 0 and 1")

    def _get_device(self, device_str: str) -> str:
        """
        Get the appropriate device based on availability.

        :param device_str: Desired device string
        :return: Available device string
        """
        if device_str == "cuda" and torch.cuda.is_available():
            return "cuda"
        elif device_str == "mps" and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        :return: Dictionary representation of configuration
        """
        return {
            "device": self.device,
            "model": {
                "input_dim": self.model.input_dim,
                "hidden_dim": self.model.hidden_dim,
                "output_dim": self.model.output_dim,
                "num_layers": self.model.num_layers,
                "activation": self.model.activation,
                "fourier_features": self.model.fourier_features,
                "fourier_scale": self.model.fourier_scale,
                "dropout": self.model.dropout,
                "layer_norm": self.model.layer_norm,
                "architecture": self.model.architecture,
            },
            "pde": {
                "domain": self.pde.domain,
                "t_domain": self.pde.t_domain,
                "initial_condition": self.pde.initial_condition,
                "boundary_conditions": self.pde.boundary_conditions,
                "diffusion_coefficient": self.pde.diffusion_coefficient,
                "source_term": self.pde.source_term,
            },
            "training": {
                "num_epochs": self.training.num_epochs,
                "batch_size": self.training.batch_size,
                "num_collocation_points": self.training.num_collocation_points,
                "num_boundary_points": self.training.num_boundary_points,
                "num_initial_points": self.training.num_initial_points,
                "learning_rate": self.training.learning_rate,
                "weight_decay": self.training.weight_decay,
                "gradient_clipping": self.training.gradient_clipping,
                "early_stopping": {
                    "enabled": self.training.early_stopping.enabled,
                    "patience": self.training.early_stopping.patience,
                    "min_delta": self.training.early_stopping.min_delta,
                },
                "learning_rate_scheduler": {
                    "type": self.training.learning_rate_scheduler.type,
                    "warmup_epochs": self.training.learning_rate_scheduler.warmup_epochs,
                    "min_lr": self.training.learning_rate_scheduler.min_lr,
                    "factor": self.training.learning_rate_scheduler.factor,
                    "patience": self.training.learning_rate_scheduler.patience,
                },
            },
            "rl": {
                "enabled": self.rl.enabled,
                "state_dim": self.rl.state_dim,
                "action_dim": self.rl.action_dim,
                "hidden_dim": self.rl.hidden_dim,
                "learning_rate": self.rl.learning_rate,
                "gamma": self.rl.gamma,
                "epsilon_start": self.rl.epsilon_start,
                "epsilon_end": self.rl.epsilon_end,
                "epsilon_decay": self.rl.epsilon_decay,
                "memory_size": self.rl.memory_size,
                "batch_size": self.rl.batch_size,
                "target_update": self.rl.target_update,
                "reward_weights": self.rl.reward_weights,
            },
            "evaluation": {
                "resolution": self.evaluation.resolution,
                "num_test_points": self.evaluation.num_test_points,
                "metrics": self.evaluation.metrics,
                "save_plots": self.evaluation.save_plots,
                "plot_frequency": self.evaluation.plot_frequency,
            },
            "logging": {
                "level": self.logging.level,
                "save_tensorboard": self.logging.save_tensorboard,
                "log_frequency": self.logging.log_frequency,
            },
            "paths": {
                "experiments_dir": self.paths.experiments_dir,
                "model_dir": self.paths.model_dir,
                "log_dir": self.paths.log_dir,
                "tensorboard_dir": self.paths.tensorboard_dir,
            },
        }
