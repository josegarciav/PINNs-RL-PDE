"""Comprehensive tests for the src.config module.

Covers: Config, ModelConfig, TrainingConfig, AdaptiveWeightsConfig,
LearningRateSchedulerConfig, EarlyStoppingConfig, PDEConfig, RLConfig,
EvaluationConfig, LoggingConfig, PathsConfig, and bug-fix validations.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch
import yaml

from src.config import (
    DEFAULT_CONFIG_PATH,
    AdaptiveWeightsConfig,
    Config,
    EarlyStoppingConfig,
    EvaluationConfig,
    LearningRateSchedulerConfig,
    LoggingConfig,
    ModelConfig,
    PathsConfig,
    PDEConfig,
    RLConfig,
    TrainingConfig,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_yaml(path: str, data: dict) -> str:
    """Write a dict as YAML to *path* and return the path."""
    with open(path, "w") as f:
        yaml.dump(data, f)
    return path


def _minimal_config_dict() -> dict:
    """Return a minimal valid config dict that will pass validation."""
    return {
        "device": "cpu",
        "pde_type": "heat",
        "architectures": {
            "fourier": {
                "hidden_dims": [64, 64],
                "mapping_size": 32,
                "scale": 2.0,
                "activation": "tanh",
                "dropout": 0.0,
                "layer_norm": True,
            },
        },
        "pde_configs": {
            "heat": {
                "name": "Heat Equation",
                "architecture": "fourier",
                "input_dim": 2,
                "output_dim": 1,
                "domain": [[0, 1]],
                "time_domain": [0, 1],
                "initial_condition": {"type": "sin"},
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "diffusion_coefficient": 0.01,
            },
        },
        "model": {
            "input_dim": 2,
            "hidden_dim": 64,
            "output_dim": 1,
            "num_layers": 4,
            "activation": "tanh",
        },
        "training": {
            "num_epochs": 100,
            "batch_size": 32,
            "num_collocation_points": 100,
            "num_boundary_points": 50,
            "num_initial_points": 50,
            "optimizer_config": {
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
            },
            "gradient_clipping": 1.0,
            "early_stopping": {"enabled": True, "patience": 10, "min_delta": 1e-4},
            "loss_weights": {"residual": 1.0, "boundary": 1.0, "initial": 1.0},
        },
        "rl": {
            "enabled": False,
            "state_dim": 2,
            "action_dim": 1,
            "hidden_dim": 64,
            "learning_rate": 0.001,
            "gamma": 0.99,
            "epsilon_start": 1.0,
            "epsilon_end": 0.01,
            "epsilon_decay": 0.995,
            "memory_size": 1000,
            "batch_size": 32,
            "target_update": 10,
            "reward_weights": {
                "residual": 1.0,
                "boundary": 1.0,
                "initial": 1.0,
                "exploration": 0.1,
            },
        },
        "evaluation": {
            "resolution": 50,
            "num_test_points": 100,
            "metrics": ["l2_error"],
            "save_plots": False,
            "plot_frequency": 10,
        },
        "logging": {"level": "INFO", "save_tensorboard": False, "log_frequency": 50},
        "paths": {
            "experiments_dir": "experiments",
            "model_dir": "models",
            "log_dir": "logs",
            "tensorboard_dir": "runs",
        },
    }


def _config_from_dict(data: dict) -> Config:
    """Write *data* to a temp YAML file and load a Config from it."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(data, f)
        path = f.name
    try:
        return Config(config_path=path)
    finally:
        os.unlink(path)


# ===========================================================================
# 1. Config — default loading from config.yaml
# ===========================================================================


class TestConfigDefaultLoading:
    """Loading from the project's config.yaml should populate every attribute."""

    def test_default_config_loads(self):
        cfg = Config()
        assert cfg.model is not None
        assert cfg.pde is not None
        assert cfg.training is not None
        assert cfg.rl is not None
        assert cfg.paths is not None

    def test_default_config_path(self):
        cfg = Config()
        assert cfg.config_path == DEFAULT_CONFIG_PATH

    def test_device_is_torch_device(self):
        cfg = Config()
        assert isinstance(cfg.device, torch.device)

    def test_model_is_model_config(self):
        cfg = Config()
        assert isinstance(cfg.model, ModelConfig)

    def test_pde_is_pde_config(self):
        cfg = Config()
        assert isinstance(cfg.pde, PDEConfig)

    def test_training_is_training_config(self):
        cfg = Config()
        assert isinstance(cfg.training, TrainingConfig)

    def test_rl_is_rl_config(self):
        cfg = Config()
        assert isinstance(cfg.rl, RLConfig)

    def test_paths_is_paths_config(self):
        cfg = Config()
        assert isinstance(cfg.paths, PathsConfig)

    def test_evaluation_is_evaluation_config(self):
        cfg = Config()
        assert isinstance(cfg.evaluation, EvaluationConfig)

    def test_logging_is_logging_config(self):
        cfg = Config()
        assert isinstance(cfg.logging, LoggingConfig)

    def test_pde_type_stored(self):
        cfg = Config()
        assert cfg.pde_type == "heat"

    def test_pde_full_config_stored(self):
        cfg = Config()
        assert isinstance(cfg.pde_full_config, dict)


class TestConfigNonexistentPath:
    """When config path does not exist, attributes stay None."""

    def test_no_file_gives_none_attributes(self):
        cfg = Config(config_path="/nonexistent/path/to/config.yaml")
        assert cfg.model is None
        assert cfg.pde is None
        assert cfg.training is None
        assert cfg.rl is None
        assert cfg.paths is None
        assert cfg.device == torch.device("cpu")


# ===========================================================================
# 2. Config._validate_config()
# ===========================================================================


class TestValidateConfig:

    def test_valid_config_passes(self):
        """A minimal valid config should not raise."""
        _config_from_dict(_minimal_config_dict())

    # -- Model validation --

    def test_invalid_input_dim(self):
        d = _minimal_config_dict()
        d["model"]["input_dim"] = -1
        d["pde_configs"]["heat"]["input_dim"] = -1
        with pytest.raises(ValueError, match="input_dim must be positive"):
            _config_from_dict(d)

    def test_invalid_hidden_dim(self):
        d = _minimal_config_dict()
        d["model"]["hidden_dim"] = 0
        # Remove arch-specific hidden_dim so model falls back to model config
        d["architectures"]["fourier"].pop("hidden_dims", None)
        d["architectures"]["fourier"]["hidden_dim"] = 0
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            _config_from_dict(d)

    def test_invalid_output_dim(self):
        d = _minimal_config_dict()
        d["model"]["output_dim"] = -1
        d["pde_configs"]["heat"]["output_dim"] = -1
        with pytest.raises(ValueError, match="output_dim must be positive"):
            _config_from_dict(d)

    def test_invalid_num_layers(self):
        d = _minimal_config_dict()
        d["model"]["num_layers"] = 0
        # Ensure arch-specific config also has 0 layers
        d["architectures"]["fourier"]["num_layers"] = 0
        with pytest.raises(ValueError, match="num_layers must be positive"):
            _config_from_dict(d)

    def test_invalid_activation(self):
        d = _minimal_config_dict()
        d["model"]["activation"] = "swish"
        d["architectures"]["fourier"]["activation"] = "swish"
        with pytest.raises(ValueError, match="Invalid activation"):
            _config_from_dict(d)

    # -- PDE validation --

    def test_invalid_domain_format(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["domain"] = [1, 2, 3]
        with pytest.raises(ValueError, match="domain"):
            _config_from_dict(d)

    def test_invalid_t_domain(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["time_domain"] = [0, 1, 2]
        with pytest.raises(ValueError, match="t_domain"):
            _config_from_dict(d)

    def test_invalid_diffusion_coefficient(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["diffusion_coefficient"] = -0.5
        with pytest.raises(ValueError, match="diffusion_coefficient must be positive"):
            _config_from_dict(d)

    # -- Training validation --

    def test_invalid_num_epochs(self):
        d = _minimal_config_dict()
        d["training"]["num_epochs"] = 0
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            _config_from_dict(d)

    def test_invalid_batch_size(self):
        d = _minimal_config_dict()
        d["training"]["batch_size"] = -1
        with pytest.raises(ValueError, match="batch_size must be positive"):
            _config_from_dict(d)

    def test_invalid_learning_rate(self):
        d = _minimal_config_dict()
        d["training"]["optimizer_config"]["learning_rate"] = 0
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            _config_from_dict(d)

    # -- RL validation (only when enabled) --

    def test_rl_invalid_state_dim_when_enabled(self):
        d = _minimal_config_dict()
        d["rl"]["enabled"] = True
        d["rl"]["state_dim"] = 0
        with pytest.raises(ValueError, match="state_dim must be positive"):
            _config_from_dict(d)

    def test_rl_invalid_action_dim_when_enabled(self):
        d = _minimal_config_dict()
        d["rl"]["enabled"] = True
        d["rl"]["action_dim"] = -1
        with pytest.raises(ValueError, match="action_dim must be positive"):
            _config_from_dict(d)

    def test_rl_invalid_gamma_when_enabled(self):
        d = _minimal_config_dict()
        d["rl"]["enabled"] = True
        d["rl"]["gamma"] = 1.5
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            _config_from_dict(d)

    def test_rl_not_validated_when_disabled(self):
        """RL fields are not validated when rl.enabled is False."""
        d = _minimal_config_dict()
        d["rl"]["enabled"] = False
        d["rl"]["state_dim"] = -999
        # Should NOT raise
        _config_from_dict(d)

    # -- Domain format variants --

    def test_valid_flat_domain(self):
        """[0.0, 1.0] is a valid domain (simple 1D)."""
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["domain"] = [0.0, 1.0]
        cfg = _config_from_dict(d)
        assert cfg.pde.domain == [0.0, 1.0]

    def test_valid_nested_domain_2d(self):
        """[[0,1],[0,1]] is a valid 2D domain."""
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["domain"] = [[0, 1], [0, 1]]
        cfg = _config_from_dict(d)
        assert cfg.pde.domain == [[0, 1], [0, 1]]


# ===========================================================================
# 3. Config._get_device()
# ===========================================================================


class TestGetDevice:

    def test_cpu_always_returns_cpu(self):
        d = _minimal_config_dict()
        d["device"] = "cpu"
        cfg = _config_from_dict(d)
        assert cfg.device == torch.device("cpu")

    def test_cuda_falls_back_to_cpu_when_unavailable(self):
        with patch("torch.cuda.is_available", return_value=False):
            d = _minimal_config_dict()
            d["device"] = "cuda"
            cfg = _config_from_dict(d)
            assert cfg.device == torch.device("cpu")

    def test_cuda_returns_cuda_when_available(self):
        with patch("torch.cuda.is_available", return_value=True):
            d = _minimal_config_dict()
            d["device"] = "cuda"
            cfg = _config_from_dict(d)
            assert cfg.device == torch.device("cuda")

    def test_mps_falls_back_to_cpu_when_unavailable(self):
        with patch("torch.backends.mps.is_available", return_value=False):
            d = _minimal_config_dict()
            d["device"] = "mps"
            cfg = _config_from_dict(d)
            assert cfg.device == torch.device("cpu")

    def test_mps_returns_mps_when_available(self):
        with patch("torch.backends.mps.is_available", return_value=True):
            d = _minimal_config_dict()
            d["device"] = "mps"
            cfg = _config_from_dict(d)
            assert cfg.device == torch.device("mps")

    def test_unknown_device_falls_back_to_cpu(self):
        d = _minimal_config_dict()
        d["device"] = "tpu"
        # _get_device returns cpu for unknown, but _validate_config may reject
        # Actually _get_device returns cpu for unknown string, then validate
        # checks against valid_devices list -- cpu is valid so it passes.
        cfg = _config_from_dict(d)
        assert cfg.device == torch.device("cpu")


# ===========================================================================
# 4. Config.to_dict() — round-trip
# ===========================================================================


class TestToDict:

    def test_to_dict_returns_dict(self):
        cfg = _config_from_dict(_minimal_config_dict())
        result = cfg.to_dict()
        assert isinstance(result, dict)

    def test_to_dict_has_all_top_level_keys(self):
        cfg = _config_from_dict(_minimal_config_dict())
        result = cfg.to_dict()
        expected_keys = {
            "device",
            "model",
            "pde",
            "training",
            "rl",
            "evaluation",
            "logging",
            "paths",
        }
        assert expected_keys == set(result.keys())

    def test_to_dict_model_section(self):
        cfg = _config_from_dict(_minimal_config_dict())
        model = cfg.to_dict()["model"]
        assert model["input_dim"] == cfg.model.input_dim
        assert model["hidden_dim"] == cfg.model.hidden_dim
        assert model["output_dim"] == cfg.model.output_dim
        assert model["architecture"] == cfg.model.architecture

    def test_to_dict_training_section(self):
        cfg = _config_from_dict(_minimal_config_dict())
        training = cfg.to_dict()["training"]
        assert training["num_epochs"] == cfg.training.num_epochs
        assert training["learning_rate"] == cfg.training.learning_rate
        assert "early_stopping" in training
        assert "learning_rate_scheduler" in training
        assert "loss_weights" in training

    def test_to_dict_pde_section(self):
        cfg = _config_from_dict(_minimal_config_dict())
        pde = cfg.to_dict()["pde"]
        assert pde["domain"] == cfg.pde.domain
        assert pde["t_domain"] == cfg.pde.t_domain

    def test_to_dict_rl_section(self):
        cfg = _config_from_dict(_minimal_config_dict())
        rl = cfg.to_dict()["rl"]
        assert rl["enabled"] == cfg.rl.enabled
        assert rl["gamma"] == cfg.rl.gamma
        assert "reward_weights" in rl

    def test_to_dict_device_is_torch_device(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert isinstance(cfg.to_dict()["device"], torch.device)

    def test_to_dict_preserves_loss_weights(self):
        d = _minimal_config_dict()
        d["training"]["loss_weights"] = {"residual": 5.0, "boundary": 3.0, "initial": 2.0}
        cfg = _config_from_dict(d)
        assert cfg.to_dict()["training"]["loss_weights"] == {
            "residual": 5.0,
            "boundary": 3.0,
            "initial": 2.0,
        }


# ===========================================================================
# 5. Config.get() and Config.__getitem__()
# ===========================================================================


class TestConfigDictAccess:

    def test_getitem_model(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert cfg["model"] is cfg.model

    def test_getitem_training(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert cfg["training"] is cfg.training

    def test_getitem_pde(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert cfg["pde"] is cfg.pde

    def test_getitem_device(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert cfg["device"] == cfg.device

    def test_get_existing_key(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert cfg.get("model") is cfg.model

    def test_get_missing_key_returns_default(self):
        cfg = _config_from_dict(_minimal_config_dict())
        assert cfg.get("nonexistent") is None
        assert cfg.get("nonexistent", 42) == 42

    def test_getitem_missing_key_raises(self):
        cfg = _config_from_dict(_minimal_config_dict())
        with pytest.raises(AttributeError):
            _ = cfg["nonexistent_key"]


# ===========================================================================
# 6. ModelConfig
# ===========================================================================


class TestModelConfig:

    def test_basic_construction(self):
        mc = ModelConfig(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            num_layers=3,
            activation="tanh",
        )
        assert mc.input_dim == 2
        assert mc.hidden_dim == 64
        assert mc.output_dim == 1
        assert mc.num_layers == 3
        assert mc.activation == "tanh"
        assert mc.architecture == "feedforward"  # default

    def test_hidden_dims_auto_created(self):
        mc = ModelConfig(input_dim=2, hidden_dim=32, output_dim=1, num_layers=5, activation="relu")
        assert mc.hidden_dims == [32, 32, 32, 32, 32]

    def test_resnet_sets_num_blocks(self):
        mc = ModelConfig(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            num_layers=6,
            activation="tanh",
            architecture="resnet",
        )
        assert mc.num_blocks == 6

    def test_fno_sets_num_blocks(self):
        mc = ModelConfig(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            num_layers=4,
            activation="gelu",
            architecture="fno",
        )
        assert mc.num_blocks == 4

    def test_feedforward_no_num_blocks(self):
        mc = ModelConfig(
            input_dim=2,
            hidden_dim=64,
            output_dim=1,
            num_layers=4,
            activation="tanh",
            architecture="feedforward",
        )
        assert not hasattr(mc, "num_blocks") or mc.num_blocks is None

    def test_default_optional_kwargs(self):
        mc = ModelConfig(input_dim=2, hidden_dim=64, output_dim=1, num_layers=3, activation="tanh")
        assert mc.fourier_features == 0
        assert mc.fourier_scale == 1.0
        assert mc.dropout == 0.0
        assert mc.layer_norm is False

    def test_get_existing(self):
        mc = ModelConfig(input_dim=2, hidden_dim=64, output_dim=1, num_layers=3, activation="tanh")
        assert mc.get("input_dim") == 2

    def test_get_missing_returns_default(self):
        mc = ModelConfig(input_dim=2, hidden_dim=64, output_dim=1, num_layers=3, activation="tanh")
        assert mc.get("nonexistent") is None
        assert mc.get("nonexistent", "fallback") == "fallback"

    def test_getitem(self):
        mc = ModelConfig(input_dim=2, hidden_dim=64, output_dim=1, num_layers=3, activation="tanh")
        assert mc["hidden_dim"] == 64

    def test_getitem_missing_raises(self):
        mc = ModelConfig(input_dim=2, hidden_dim=64, output_dim=1, num_layers=3, activation="tanh")
        with pytest.raises(AttributeError):
            _ = mc["nonexistent_key"]


# ===========================================================================
# 7. TrainingConfig
# ===========================================================================


class TestTrainingConfig:

    def _make_training_config(self, **overrides):
        defaults = dict(
            num_epochs=100,
            batch_size=32,
            num_collocation_points=100,
            num_boundary_points=50,
            num_initial_points=50,
            learning_rate=0.001,
            weight_decay=0.0001,
            gradient_clipping=1.0,
            early_stopping=EarlyStoppingConfig(enabled=True, patience=10, min_delta=1e-4),
            learning_rate_scheduler=LearningRateSchedulerConfig(
                type="cosine", warmup_epochs=10, min_lr=1e-6, factor=0.5, patience=50
            ),
        )
        defaults.update(overrides)
        return TrainingConfig(**defaults)

    def test_post_init_defaults_loss_weights(self):
        tc = self._make_training_config()
        assert tc.loss_weights == {"residual": 1.0, "boundary": 1.0, "initial": 1.0}

    def test_post_init_defaults_adaptive_weights(self):
        tc = self._make_training_config()
        assert isinstance(tc.adaptive_weights, AdaptiveWeightsConfig)
        assert tc.adaptive_weights.enabled is False

    def test_explicit_loss_weights_preserved(self):
        weights = {"residual": 5.0, "boundary": 3.0, "initial": 2.0}
        tc = self._make_training_config(loss_weights=weights)
        assert tc.loss_weights == weights

    def test_explicit_adaptive_weights_preserved(self):
        aw = AdaptiveWeightsConfig(enabled=True, strategy="lrw", alpha=0.5, eps=1e-4)
        tc = self._make_training_config(adaptive_weights=aw)
        assert tc.adaptive_weights.enabled is True
        assert tc.adaptive_weights.strategy == "lrw"

    def test_optimizer_config_property(self):
        tc = self._make_training_config(learning_rate=0.005, weight_decay=0.001)
        oc = tc.optimizer_config
        assert oc == {"learning_rate": 0.005, "weight_decay": 0.001}

    def test_getitem_regular_attr(self):
        tc = self._make_training_config()
        assert tc["num_epochs"] == 100

    def test_getitem_optimizer_config(self):
        tc = self._make_training_config(learning_rate=0.01)
        assert tc["optimizer_config"]["learning_rate"] == 0.01

    def test_get_regular_attr(self):
        tc = self._make_training_config()
        assert tc.get("batch_size") == 32

    def test_get_optimizer_config(self):
        tc = self._make_training_config()
        assert tc.get("optimizer_config") is not None
        assert "learning_rate" in tc.get("optimizer_config")

    def test_get_missing_returns_default(self):
        tc = self._make_training_config()
        assert tc.get("nonexistent") is None
        assert tc.get("nonexistent", 99) == 99

    def test_collocation_distribution_default(self):
        tc = self._make_training_config()
        assert tc.collocation_distribution == "uniform"


# ===========================================================================
# 8. AdaptiveWeightsConfig
# ===========================================================================


class TestAdaptiveWeightsConfig:

    def test_defaults(self):
        aw = AdaptiveWeightsConfig()
        assert aw.enabled is False
        assert aw.strategy == "rbw"
        assert aw.alpha == 0.9
        assert aw.eps == 1e-5
        assert aw.initial_weights == [0.5, 0.3, 0.2]

    def test_post_init_none_initial_weights(self):
        aw = AdaptiveWeightsConfig(enabled=True, initial_weights=None)
        assert aw.initial_weights == [0.5, 0.3, 0.2]

    def test_explicit_initial_weights(self):
        aw = AdaptiveWeightsConfig(initial_weights=[0.4, 0.4, 0.2])
        assert aw.initial_weights == [0.4, 0.4, 0.2]

    def test_custom_strategy(self):
        aw = AdaptiveWeightsConfig(strategy="lrw")
        assert aw.strategy == "lrw"


# ===========================================================================
# 9. All dataclasses instantiation
# ===========================================================================


class TestDataclassInstantiation:

    def test_learning_rate_scheduler_config(self):
        lrs = LearningRateSchedulerConfig(
            type="cosine", warmup_epochs=100, min_lr=1e-6, factor=0.5, patience=50
        )
        assert lrs.type == "cosine"
        assert lrs.warmup_epochs == 100
        assert lrs.min_lr == 1e-6
        assert lrs.factor == 0.5
        assert lrs.patience == 50

    def test_early_stopping_config(self):
        es = EarlyStoppingConfig(enabled=True, patience=20, min_delta=1e-5)
        assert es.enabled is True
        assert es.patience == 20
        assert es.min_delta == 1e-5

    def test_pde_config(self):
        pc = PDEConfig(
            domain=[0.0, 1.0],
            t_domain=[0.0, 1.0],
            initial_condition="sin(pi*x)",
            boundary_conditions={"left": "0.0", "right": "0.0"},
            diffusion_coefficient=0.01,
            source_term="0.0",
        )
        assert pc.domain == [0.0, 1.0]
        assert pc.diffusion_coefficient == 0.01

    def test_rl_config(self):
        rc = RLConfig(
            enabled=True,
            state_dim=4,
            action_dim=2,
            hidden_dim=64,
            learning_rate=0.001,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            target_update=100,
            reward_weights={"residual": 1.0},
        )
        assert rc.enabled is True
        assert rc.gamma == 0.99

    def test_evaluation_config(self):
        ec = EvaluationConfig(
            resolution=100,
            num_test_points=500,
            metrics=["l2_error", "max_error"],
            save_plots=True,
            plot_frequency=50,
        )
        assert ec.resolution == 100
        assert "l2_error" in ec.metrics

    def test_logging_config(self):
        lc = LoggingConfig(level="DEBUG", save_tensorboard=True, log_frequency=10)
        assert lc.level == "DEBUG"
        assert lc.save_tensorboard is True

    def test_paths_config(self):
        pc = PathsConfig(
            experiments_dir="exp",
            model_dir="mod",
            log_dir="log",
            tensorboard_dir="tb",
        )
        assert pc.experiments_dir == "exp"
        assert pc.tensorboard_dir == "tb"


# ===========================================================================
# 10. Bug-fix validations
# ===========================================================================


class TestBugFixLossWeightsNormalization:
    """Bug #12: loss_weights key 'pde' should be normalized to 'residual'."""

    def test_pde_key_normalized_to_residual(self):
        d = _minimal_config_dict()
        d["training"]["loss_weights"] = {"pde": 10.0, "boundary": 5.0, "initial": 3.0}
        cfg = _config_from_dict(d)
        assert "residual" in cfg.training.loss_weights
        assert "pde" not in cfg.training.loss_weights
        assert cfg.training.loss_weights["residual"] == 10.0

    def test_residual_key_untouched(self):
        d = _minimal_config_dict()
        d["training"]["loss_weights"] = {"residual": 7.0, "boundary": 5.0, "initial": 3.0}
        cfg = _config_from_dict(d)
        assert cfg.training.loss_weights["residual"] == 7.0

    def test_no_loss_weights_uses_default(self):
        d = _minimal_config_dict()
        d["training"].pop("loss_weights", None)
        cfg = _config_from_dict(d)
        # __post_init__ default
        assert cfg.training.loss_weights == {"residual": 1.0, "boundary": 1.0, "initial": 1.0}


class TestBugFixOptimizerConfigLR:
    """Bug #2: LR should be read from training.optimizer_config.learning_rate."""

    def test_lr_from_optimizer_config(self):
        d = _minimal_config_dict()
        d["training"]["optimizer_config"] = {"learning_rate": 0.005, "weight_decay": 0.0005}
        cfg = _config_from_dict(d)
        assert cfg.training.learning_rate == 0.005
        assert cfg.training.weight_decay == 0.0005

    def test_lr_fallback_to_top_level(self):
        d = _minimal_config_dict()
        d["training"].pop("optimizer_config", None)
        d["training"]["learning_rate"] = 0.01
        d["training"]["weight_decay"] = 0.001
        cfg = _config_from_dict(d)
        assert cfg.training.learning_rate == 0.01
        assert cfg.training.weight_decay == 0.001

    def test_optimizer_config_property_matches(self):
        d = _minimal_config_dict()
        d["training"]["optimizer_config"] = {"learning_rate": 0.005, "weight_decay": 0.0005}
        cfg = _config_from_dict(d)
        oc = cfg.training.optimizer_config
        assert oc["learning_rate"] == 0.005
        assert oc["weight_decay"] == 0.0005


# ===========================================================================
# Architecture-specific param injection (Bug #1)
# ===========================================================================


class TestArchitectureParamInjection:
    """Bug #1: Architecture-specific params should be injected into ModelConfig."""

    def test_fourier_mapping_size_injected(self):
        d = _minimal_config_dict()
        d["architectures"]["fourier"]["mapping_size"] = 256
        cfg = _config_from_dict(d)
        assert cfg.model.mapping_size == 256

    def test_fourier_scale_injected(self):
        d = _minimal_config_dict()
        d["architectures"]["fourier"]["scale"] = 8.0
        cfg = _config_from_dict(d)
        assert cfg.model.scale == 8.0

    def test_hidden_dims_injected(self):
        d = _minimal_config_dict()
        d["architectures"]["fourier"]["hidden_dims"] = [128, 256, 128]
        cfg = _config_from_dict(d)
        assert cfg.model.hidden_dims == [128, 256, 128]

    def test_siren_omega_0_injected(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["architecture"] = "siren"
        d["architectures"]["siren"] = {
            "hidden_dims": [64, 64],
            "omega_0": 30.0,
            "activation": "tanh",
        }
        cfg = _config_from_dict(d)
        assert cfg.model.omega_0 == 30.0

    def test_attention_num_heads_injected(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["architecture"] = "attention"
        d["architectures"]["attention"] = {
            "hidden_dim": 64,
            "num_heads": 8,
            "num_layers": 4,
            "activation": "gelu",
        }
        cfg = _config_from_dict(d)
        assert cfg.model.num_heads == 8

    def test_autoencoder_latent_dim_injected(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["architecture"] = "autoencoder"
        d["architectures"]["autoencoder"] = {
            "hidden_dims": [64, 128, 64],
            "latent_dim": 32,
            "activation": "relu",
        }
        cfg = _config_from_dict(d)
        assert cfg.model.latent_dim == 32

    def test_fno_modes_injected(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["architecture"] = "fno"
        d["architectures"]["fno"] = {
            "hidden_dim": 128,
            "num_blocks": 4,
            "modes": 16,
            "activation": "gelu",
        }
        cfg = _config_from_dict(d)
        assert cfg.model.modes == 16
        assert cfg.model.num_blocks == 4


# ===========================================================================
# PDE-specific config selection
# ===========================================================================


class TestPDEConfigSelection:

    def test_heat_pde_selected_by_default(self):
        cfg = Config()
        assert cfg.pde_type == "heat"

    def test_custom_pde_type(self):
        d = _minimal_config_dict()
        d["pde_type"] = "heat"
        cfg = _config_from_dict(d)
        assert cfg.pde_type == "heat"

    def test_unknown_pde_type_uses_fallback(self):
        d = _minimal_config_dict()
        d["pde_type"] = "nonexistent_pde"
        # Should not crash -- falls back to model defaults
        cfg = _config_from_dict(d)
        assert cfg.pde_type == "nonexistent_pde"

    def test_input_dim_from_pde_config(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["input_dim"] = 3
        cfg = _config_from_dict(d)
        assert cfg.model.input_dim == 3

    def test_output_dim_from_pde_config(self):
        d = _minimal_config_dict()
        d["pde_configs"]["heat"]["output_dim"] = 2
        cfg = _config_from_dict(d)
        assert cfg.model.output_dim == 2


# ===========================================================================
# Scheduler config loading
# ===========================================================================


class TestSchedulerConfigLoading:

    def test_cosine_scheduler_params(self):
        d = _minimal_config_dict()
        d["training"]["scheduler_type"] = "cosine"
        d["training"]["cosine_params"] = {
            "T_max": 200,
            "eta_min": 1e-7,
            "warmup_epochs": 50,
            "min_lr": 1e-7,
        }
        cfg = _config_from_dict(d)
        assert cfg.training.learning_rate_scheduler.warmup_epochs == 50

    def test_reduce_lr_scheduler_params(self):
        d = _minimal_config_dict()
        d["training"]["scheduler_type"] = "reduce_lr"
        d["training"]["reduce_lr_params"] = {
            "factor": 0.3,
            "patience": 25,
            "min_lr": 1e-5,
        }
        cfg = _config_from_dict(d)
        assert cfg.training.learning_rate_scheduler.factor == 0.3
        assert cfg.training.learning_rate_scheduler.patience == 25

    def test_scheduler_type_as_dict(self):
        d = _minimal_config_dict()
        d["training"]["scheduler_type"] = {"type": "cosine"}
        d["training"]["cosine_params"] = {"warmup_epochs": 20, "min_lr": 1e-6}
        cfg = _config_from_dict(d)
        assert cfg.training.learning_rate_scheduler.warmup_epochs == 20


# ===========================================================================
# Default config.yaml values from real project config
# ===========================================================================


class TestDefaultConfigYAMLValues:
    """Verify specific values from the project's config.yaml are loaded correctly."""

    def test_default_lr_is_0_005(self):
        cfg = Config()
        assert cfg.training.learning_rate == 0.005

    def test_default_weight_decay(self):
        cfg = Config()
        assert cfg.training.weight_decay == 0.0005

    def test_default_num_epochs(self):
        cfg = Config()
        assert cfg.training.num_epochs == 3000

    def test_default_batch_size(self):
        cfg = Config()
        assert cfg.training.batch_size == 2048

    def test_loss_weights_from_yaml(self):
        cfg = Config()
        assert cfg.training.loss_weights["residual"] == 15.0
        assert cfg.training.loss_weights["boundary"] == 20.0
        assert cfg.training.loss_weights["initial"] == 10.0

    def test_rl_disabled_by_default(self):
        cfg = Config()
        assert cfg.rl.enabled is False

    def test_heat_architecture_is_fourier(self):
        cfg = Config()
        assert cfg.model.architecture == "fourier"

    def test_fourier_mapping_size_from_yaml(self):
        cfg = Config()
        assert cfg.model.mapping_size == 512

    def test_fourier_scale_from_yaml(self):
        cfg = Config()
        assert cfg.model.scale == 4.0

    def test_early_stopping_enabled(self):
        cfg = Config()
        assert cfg.training.early_stopping.enabled is True

    def test_collocation_distribution_uniform(self):
        cfg = Config()
        assert cfg.training.collocation_distribution == "uniform"

    def test_adaptive_weights_disabled(self):
        cfg = Config()
        assert cfg.training.adaptive_weights.enabled is False
