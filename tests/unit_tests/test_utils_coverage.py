"""Comprehensive tests for src/utils/utils.py to maximize coverage."""

import json
import logging
import os

import numpy as np
import pytest
import torch

from src.utils.utils import (
    generate_collocation_points,
    load_model,
    save_model,
    save_training_metrics,
    setup_logging,
)


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_creates_log_directory(self, tmp_path):
        log_dir = str(tmp_path / "my_logs")
        logger = setup_logging(log_dir)
        assert os.path.isdir(log_dir)
        assert isinstance(logger, logging.Logger)

    def test_creates_log_file(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        setup_logging(log_dir)
        log_files = os.listdir(log_dir)
        assert len(log_files) >= 1
        assert any(f.startswith("training_") and f.endswith(".log") for f in log_files)

    def test_default_log_dir(self, monkeypatch, tmp_path):
        # Change cwd so the default "logs" dir is created somewhere safe
        monkeypatch.chdir(tmp_path)
        logger = setup_logging()
        assert os.path.isdir(tmp_path / "logs")
        assert isinstance(logger, logging.Logger)

    def test_existing_directory_no_error(self, tmp_path):
        log_dir = str(tmp_path / "logs")
        os.makedirs(log_dir)
        # Should not raise
        logger = setup_logging(log_dir)
        assert isinstance(logger, logging.Logger)

    def test_logger_name(self, tmp_path):
        logger = setup_logging(str(tmp_path / "logs"))
        assert logger.name == "src.utils.utils"


# ---------------------------------------------------------------------------
# generate_collocation_points
# ---------------------------------------------------------------------------


class TestGenerateCollocationPoints:
    def test_uniform_shape(self):
        points = generate_collocation_points(100, (0.0, 1.0), device=torch.device("cpu"))
        assert points.shape == (100, 1)

    def test_uniform_domain_range(self):
        points = generate_collocation_points(
            5000, (-2.0, 3.0), device=torch.device("cpu"), distribution="uniform"
        )
        assert points.min().item() >= -2.0
        assert points.max().item() <= 3.0

    def test_uniform_device_cpu(self):
        points = generate_collocation_points(10, (0.0, 1.0), device=torch.device("cpu"))
        assert points.device == torch.device("cpu")

    def test_default_device(self):
        # device=None should fall back to auto-detection (cpu or mps)
        points = generate_collocation_points(10, (0.0, 1.0), device=None)
        assert isinstance(points, torch.Tensor)

    def test_single_point(self):
        points = generate_collocation_points(1, (0.0, 1.0), device=torch.device("cpu"))
        assert points.shape == (1, 1)

    def test_large_num_points(self):
        points = generate_collocation_points(10000, (0.0, 1.0), device=torch.device("cpu"))
        assert points.shape == (10000, 1)

    def test_negative_domain(self):
        points = generate_collocation_points(
            500, (-10.0, -5.0), device=torch.device("cpu")
        )
        assert points.min().item() >= -10.0
        assert points.max().item() <= -5.0

    def test_unsupported_distribution_raises(self):
        with pytest.raises(ValueError, match="Unsupported distribution"):
            generate_collocation_points(
                10, (0.0, 1.0), device=torch.device("cpu"), distribution="gaussian"
            )

    def test_unsupported_distribution_message(self):
        with pytest.raises(ValueError, match="gaussian"):
            generate_collocation_points(
                10, (0.0, 1.0), device=torch.device("cpu"), distribution="gaussian"
            )

    def test_zero_width_domain(self):
        # domain (5.0, 5.0) => all points should equal 5.0
        points = generate_collocation_points(50, (5.0, 5.0), device=torch.device("cpu"))
        assert torch.allclose(points, torch.full((50, 1), 5.0))

    def test_kwargs_accepted(self):
        # Extra kwargs should not raise (they're unused for uniform but accepted by signature)
        points = generate_collocation_points(
            10, (0.0, 1.0), device=torch.device("cpu"), distribution="uniform", seed=42
        )
        assert points.shape == (10, 1)


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------


def _make_simple_model():
    """Create a minimal nn.Module for testing."""
    model = torch.nn.Linear(2, 1)
    return model


class TestSaveModel:
    def test_saves_state_dict(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "models" / "model.pth")
        save_model(model, path)
        assert os.path.isfile(path)

    def test_creates_parent_directory(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "deep" / "nested" / "dir" / "model.pth")
        save_model(model, path)
        assert os.path.isfile(path)

    def test_saves_config_json(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        config = {"lr": 0.001, "epochs": 100}
        save_model(model, path, config=config)

        config_path = str(tmp_path / "model_config.json")
        assert os.path.isfile(config_path)
        with open(config_path, "r") as f:
            loaded = json.load(f)
        assert loaded == config

    def test_no_config_no_json(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        save_model(model, path, config=None)
        config_path = str(tmp_path / "model_config.json")
        assert not os.path.exists(config_path)

    def test_save_state_method_object(self, tmp_path):
        """Test the hasattr(model, 'save_state') branch."""

        class FakeRLAgent:
            def __init__(self):
                self.saved_path = None

            def save_state(self, path):
                self.saved_path = path
                # Write something so we can verify
                with open(path, "w") as f:
                    f.write("agent_state")

        agent = FakeRLAgent()
        path = str(tmp_path / "agent.pth")
        save_model(agent, path)
        assert agent.saved_path == path
        assert os.path.isfile(path)

    def test_unsupported_model_raises(self, tmp_path):
        path = str(tmp_path / "bad.pth")
        with pytest.raises(ValueError, match="Model must be a PyTorch model or RLAgent"):
            save_model("not_a_model", path)

    def test_config_with_nested_values(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        config = {"arch": {"hidden": [64, 64]}, "pde": "heat"}
        save_model(model, path, config=config)
        config_path = str(tmp_path / "model_config.json")
        with open(config_path, "r") as f:
            loaded = json.load(f)
        assert loaded["arch"]["hidden"] == [64, 64]


class TestLoadModel:
    def test_load_weights(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        torch.save(model.state_dict(), path)

        new_model = torch.nn.Linear(2, 1)
        loaded_model, config = load_model(new_model, path, device=torch.device("cpu"))
        # Weights should match
        for p1, p2 in zip(model.parameters(), loaded_model.parameters()):
            assert torch.allclose(p1, p2)
        # No config file => config is None
        assert config is None

    def test_load_with_config(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        config = {"lr": 0.005}
        save_model(model, path, config=config)

        new_model = torch.nn.Linear(2, 1)
        loaded_model, loaded_config = load_model(
            new_model, path, device=torch.device("cpu"), load_config=True
        )
        assert loaded_config == config

    def test_load_config_false(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        config = {"lr": 0.005}
        save_model(model, path, config=config)

        new_model = torch.nn.Linear(2, 1)
        _, loaded_config = load_model(
            new_model, path, device=torch.device("cpu"), load_config=False
        )
        assert loaded_config is None

    def test_load_config_missing_file(self, tmp_path):
        """When load_config=True but no config JSON exists, config should be None."""
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        torch.save(model.state_dict(), path)

        new_model = torch.nn.Linear(2, 1)
        _, loaded_config = load_model(
            new_model, path, device=torch.device("cpu"), load_config=True
        )
        assert loaded_config is None

    def test_model_in_eval_mode(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        torch.save(model.state_dict(), path)

        new_model = torch.nn.Linear(2, 1)
        loaded_model, _ = load_model(new_model, path, device=torch.device("cpu"))
        assert not loaded_model.training

    def test_default_device(self, tmp_path):
        model = _make_simple_model()
        path = str(tmp_path / "model.pth")
        torch.save(model.state_dict(), path)

        new_model = torch.nn.Linear(2, 1)
        loaded_model, _ = load_model(new_model, path, device=None)
        assert not loaded_model.training

    def test_roundtrip_save_load(self, tmp_path):
        """Full roundtrip: save_model then load_model."""
        model = _make_simple_model()
        # Set specific weights
        with torch.no_grad():
            model.weight.fill_(0.42)
            model.bias.fill_(-0.1)

        path = str(tmp_path / "model.pth")
        config = {"arch": "linear", "epochs": 50}
        save_model(model, path, config=config)

        new_model = torch.nn.Linear(2, 1)
        loaded_model, loaded_config = load_model(
            new_model, path, device=torch.device("cpu")
        )
        assert torch.allclose(loaded_model.weight, torch.tensor([[0.42, 0.42]]))
        assert torch.allclose(loaded_model.bias, torch.tensor([-0.1]))
        assert loaded_config == config


# ---------------------------------------------------------------------------
# save_training_metrics
# ---------------------------------------------------------------------------


class TestSaveTrainingMetrics:
    def test_creates_experiment_dir(self, tmp_path):
        exp_dir = str(tmp_path / "experiment_1")
        history = {"loss": [1.0, 0.5, 0.1]}
        save_training_metrics(history, exp_dir)
        assert os.path.isdir(exp_dir)

    def test_saves_metrics_json(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        history = {"loss": [1.0, 0.5]}
        save_training_metrics(history, exp_dir)

        with open(os.path.join(exp_dir, "metrics.json"), "r") as f:
            data = json.load(f)
        assert data == history

    def test_saves_history_json(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        history = {"loss": [1.0, 0.5]}
        save_training_metrics(history, exp_dir)

        with open(os.path.join(exp_dir, "history.json"), "r") as f:
            data = json.load(f)
        assert data == history

    def test_returns_history_file_path(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        history = {"loss": [1.0]}
        result = save_training_metrics(history, exp_dir)
        assert result == os.path.join(exp_dir, "history.json")

    def test_no_metadata_no_metadata_file(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        save_training_metrics({"loss": []}, exp_dir, metadata=None)
        assert not os.path.exists(os.path.join(exp_dir, "metadata.json"))

    def test_saves_metadata_json(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        metadata = {"pde": "heat", "arch": "feedforward"}
        save_training_metrics({"loss": [0.1]}, exp_dir, metadata=metadata)

        with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        assert data["pde"] == "heat"
        assert data["arch"] == "feedforward"
        assert "last_updated" in data

    def test_metadata_has_timestamp(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        save_training_metrics({"loss": []}, exp_dir, metadata={"key": "val"})

        with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        # Timestamp should be YYYY-MM-DD HH:MM:SS format
        assert len(data["last_updated"]) == 19

    def test_metadata_merges_with_existing(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)

        # Pre-populate metadata.json
        existing = {"pde": "heat", "run_id": 1}
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(existing, f)

        new_metadata = {"arch": "resnet", "epochs": 100}
        save_training_metrics({"loss": []}, exp_dir, metadata=new_metadata)

        with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        # Should have both old and new keys
        assert data["pde"] == "heat"
        assert data["run_id"] == 1
        assert data["arch"] == "resnet"
        assert data["epochs"] == 100

    def test_metadata_overwrites_existing_keys(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)

        existing = {"pde": "heat", "epochs": 50}
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            json.dump(existing, f)

        save_training_metrics({"loss": []}, exp_dir, metadata={"epochs": 200})

        with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        assert data["epochs"] == 200

    def test_corrupted_existing_metadata_handled(self, tmp_path):
        """If existing metadata.json is corrupt, it should be overwritten gracefully."""
        exp_dir = str(tmp_path / "exp")
        os.makedirs(exp_dir)

        # Write invalid JSON
        with open(os.path.join(exp_dir, "metadata.json"), "w") as f:
            f.write("{corrupted json!!")

        # Should not raise due to bare except in source
        save_training_metrics({"loss": []}, exp_dir, metadata={"key": "val"})

        with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        assert data["key"] == "val"
        assert "last_updated" in data

    def test_numpy_arrays_serialized(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        history = {
            "loss": np.array([1.0, 0.5, 0.1]),
            "lr": np.array([0.01, 0.005]),
        }
        save_training_metrics(history, exp_dir)

        with open(os.path.join(exp_dir, "metrics.json"), "r") as f:
            data = json.load(f)
        assert data["loss"] == [1.0, 0.5, 0.1]
        assert data["lr"] == [0.01, 0.005]

    def test_nested_numpy_arrays(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        history = {
            "metrics": {
                "train_loss": np.array([1.0, 0.5]),
                "val_loss": np.array([1.2, 0.6]),
            },
            "grads": [np.array([0.1, 0.2]), np.array([0.05, 0.1])],
        }
        save_training_metrics(history, exp_dir)

        with open(os.path.join(exp_dir, "metrics.json"), "r") as f:
            data = json.load(f)
        assert data["metrics"]["train_loss"] == [1.0, 0.5]
        assert data["grads"][0] == [0.1, 0.2]

    def test_metadata_with_numpy_values(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        metadata = {"best_loss": np.float64(0.001), "dims": np.array([64, 64])}
        save_training_metrics({"loss": []}, exp_dir, metadata=metadata)

        with open(os.path.join(exp_dir, "metadata.json"), "r") as f:
            data = json.load(f)
        assert data["dims"] == [64, 64]

    def test_empty_history(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        result = save_training_metrics({}, exp_dir)

        with open(os.path.join(exp_dir, "metrics.json"), "r") as f:
            data = json.load(f)
        assert data == {}
        assert os.path.isfile(result)

    def test_empty_metadata_dict_not_saved(self, tmp_path):
        """Empty dict is falsy, so metadata file should NOT be created."""
        exp_dir = str(tmp_path / "exp")
        save_training_metrics({"loss": []}, exp_dir, metadata={})
        assert not os.path.exists(os.path.join(exp_dir, "metadata.json"))

    def test_plain_python_types_passthrough(self, tmp_path):
        """Non-numpy, non-dict, non-list values pass through convert_to_serializable."""
        exp_dir = str(tmp_path / "exp")
        history = {"epoch": 10, "name": "test", "flag": True, "rate": 0.5}
        save_training_metrics(history, exp_dir)

        with open(os.path.join(exp_dir, "metrics.json"), "r") as f:
            data = json.load(f)
        assert data == history
