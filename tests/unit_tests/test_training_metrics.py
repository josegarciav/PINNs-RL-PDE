"""Tests for pinnrl.training.trainer.save_training_metrics."""

import json
import os

import numpy as np

from pinnrl.training.trainer import save_training_metrics

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

        with open(os.path.join(exp_dir, "metrics.json")) as f:
            data = json.load(f)
        assert data == history

    def test_saves_history_json(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        history = {"loss": [1.0, 0.5]}
        save_training_metrics(history, exp_dir)

        with open(os.path.join(exp_dir, "history.json")) as f:
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

        with open(os.path.join(exp_dir, "metadata.json")) as f:
            data = json.load(f)
        assert data["pde"] == "heat"
        assert data["arch"] == "feedforward"
        assert "last_updated" in data

    def test_metadata_has_timestamp(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        save_training_metrics({"loss": []}, exp_dir, metadata={"key": "val"})

        with open(os.path.join(exp_dir, "metadata.json")) as f:
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

        with open(os.path.join(exp_dir, "metadata.json")) as f:
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

        with open(os.path.join(exp_dir, "metadata.json")) as f:
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

        with open(os.path.join(exp_dir, "metadata.json")) as f:
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

        with open(os.path.join(exp_dir, "metrics.json")) as f:
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

        with open(os.path.join(exp_dir, "metrics.json")) as f:
            data = json.load(f)
        assert data["metrics"]["train_loss"] == [1.0, 0.5]
        assert data["grads"][0] == [0.1, 0.2]

    def test_metadata_with_numpy_values(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        metadata = {"best_loss": np.float64(0.001), "dims": np.array([64, 64])}
        save_training_metrics({"loss": []}, exp_dir, metadata=metadata)

        with open(os.path.join(exp_dir, "metadata.json")) as f:
            data = json.load(f)
        assert data["dims"] == [64, 64]

    def test_empty_history(self, tmp_path):
        exp_dir = str(tmp_path / "exp")
        result = save_training_metrics({}, exp_dir)

        with open(os.path.join(exp_dir, "metrics.json")) as f:
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

        with open(os.path.join(exp_dir, "metrics.json")) as f:
            data = json.load(f)
        assert data == history
