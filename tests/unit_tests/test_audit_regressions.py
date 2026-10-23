"""Regression tests for the bugs found by the docstring/docs audit.

Each test fails on the code before its fix.
"""

import pytest
import torch

from pinnrl.config import Config, ModelConfig
from pinnrl.neural_networks import PINNModel
from pinnrl.training.adaptive_weights import AdaptiveLossWeights


def _model(architecture, arch_cfg):
    config = Config.__new__(Config)
    config.device = torch.device("cpu")
    config.model = ModelConfig.from_architecture(architecture, arch_cfg, input_dim=2, output_dim=1)
    return PINNModel(config, device=torch.device("cpu"))


# ── 1. optional architecture keys fall back to the network defaults ─────


@pytest.mark.parametrize(
    "architecture, arch_cfg",
    [
        ("fno", {"hidden_dim": 32}),
        ("siren", {"hidden_dims": [16, 16]}),
        ("attention", {"hidden_dim": 16, "num_layers": 2}),
        ("autoencoder", {"hidden_dims": [16, 8, 16]}),
    ],
)
def test_missing_optional_keys_use_network_defaults(architecture, arch_cfg):
    model = _model(architecture, arch_cfg)
    assert model(torch.rand(5, 2)).shape == (5, 1)


def test_model_config_get_treats_none_as_missing():
    mc = ModelConfig.from_architecture("siren", {"hidden_dims": [8]}, 2, 1)
    assert mc.omega_0 is None
    assert mc.get("omega_0", 30.0) == 30.0


# ── 2. LRW ignores components without a gradient ───────────────────────


def test_lrw_gives_zero_weight_to_a_zero_gradient_term():
    weights = AdaptiveLossWeights(strategy="lrw", alpha=0.0)
    weights.update(gradients=torch.tensor([1.0, 0.0, 0.5]))
    w = weights.update(gradients=torch.tensor([1.0, 0.0, 0.5]))
    assert w[1] == 0.0
    assert torch.allclose(w, torch.tensor([1 / 3, 0.0, 2 / 3]), atol=1e-4)
    assert torch.isclose(w.sum(), torch.tensor(1.0))


def test_lrw_uniform_when_no_term_has_a_gradient():
    weights = AdaptiveLossWeights(strategy="lrw", alpha=0.0)
    weights.update(gradients=torch.zeros(3))
    assert torch.allclose(weights.update(gradients=torch.zeros(3)), torch.full((3,), 1 / 3))


# ── 3. --dataset applies the dataset's recommended mode ───────────────


def _yaml():
    import yaml

    from pinnrl.config import DEFAULT_CONFIG_PATH

    with open(DEFAULT_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def test_dataset_uses_its_recommended_mode():
    from pinnrl.datasets import get_entry
    from pinnrl.training.train import build_config_dict

    config = build_config_dict(_yaml(), "heat", "fno", dataset={"name": "active_matter"})
    assert config["training"]["mode"] == get_entry("active_matter").recommended_mode == "data_only"


def test_explicit_mode_beats_the_dataset_default():
    from pinnrl.training.train import build_config_dict

    dataset = {"name": "active_matter", "mode": "data_augmented"}
    config = build_config_dict(_yaml(), "heat", "fno", dataset=dataset)
    assert config["training"]["mode"] == "data_augmented"


def test_build_config_dict_does_not_mutate_its_input():
    from pinnrl.training.train import build_config_dict

    raw = _yaml()
    before = raw["training"]["num_epochs"], raw["rl"]["enabled"]
    build_config_dict(raw, "heat", "fno", use_rl=True, epochs=7)
    assert (raw["training"]["num_epochs"], raw["rl"]["enabled"]) == before


# ── 5. the dashboard moves to the next port when one is taken ──────────


def test_dashboard_skips_a_busy_port(monkeypatch):
    import socket

    import pinnrl.dashboard
    from pinnrl import main

    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        served, opened = [], []
        monkeypatch.setattr(pinnrl.dashboard.app, "run", lambda **kw: served.append(kw["port"]))
        monkeypatch.setattr(main.webbrowser, "open", opened.append)
        main.run_dashboard(port, open_browser=True)
    assert served and served[0] != port
    assert opened == [f"http://127.0.0.1:{served[0]}/"]  # the browser goes to the real port


def test_dashboard_gives_up_after_three_busy_ports(monkeypatch):
    from pinnrl import main

    monkeypatch.setattr(
        main, "find_free_port", lambda port: (_ for _ in ()).throw(RuntimeError("busy"))
    )
    with pytest.raises(SystemExit):
        main.run_dashboard(8050)


# ── 6. run directory names round-trip ─────────────────────────────────


@pytest.mark.parametrize(
    "subject, arch, rl",
    [
        ("Heat Equation", "fourier", False),
        ("active_matter", "fno", True),
        ("Heat Equation 2D", "siren", False),
    ],
)
def test_run_names_round_trip(subject, arch, rl):
    from datetime import datetime

    from pinnrl.training.train import parse_run_name, run_name

    name = run_name(subject, arch, rl, when=datetime(2026, 10, 23, 10, 15, 0))
    assert parse_run_name(name) == {
        "started": "2026-10-23 10:15:00",
        "subject": subject,
        "architecture": arch,
        "rl": rl,
    }


def test_dashboard_names_runs_correctly():
    from pinnrl.dashboard import describe_run, get_experiment_name

    path = "/runs/20261023_101500_Heat Equation_fourier_no_rl"
    assert get_experiment_name(path) == "2026-10-23 10:15 - Heat Equation (fourier) [no_rl]"
    assert describe_run("20261023_101500_active_matter_fno_rl") == (
        "2026-10-23 10:15 - active_matter (fno) [rl]"
    )
    assert describe_run("my_custom_folder") == "my_custom_folder"


# ── 7. loss breakdown x positions ─────────────────────────────────────


def test_loss_curve_epochs():
    from pinnrl.plots import loss_curve_epochs

    history = {"train_loss": [1.0] * 100, "residual_loss": [1.0] * 100, "val_loss": [1.0] * 10}
    epochs, val = loss_curve_epochs(history, validation_frequency=10)
    assert epochs == list(range(1, 101))  # every per-epoch component gets all 100 points
    assert val == [1, 11, 21, 31, 41, 51, 61, 71, 81, 91]  # validated at 0-based 0, 10, 20, ...


# ── 8. the history plot draws losses only, at their own epochs ────────


def test_training_history_plot_draws_only_losses(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from pinnrl import plots

    written = []
    monkeypatch.setattr(plots, "_write_plotly", lambda fig, path: written.append(fig))
    history = {
        "train_loss": [1.0, 0.5, 0.25, 0.1],
        "residual_loss": [1.0, 0.5, 0.25, 0.1],
        "val_loss": [0.9, 0.2],
        "learning_rate": [1e-3] * 4,
        "param_alpha": [0.5, 0.4, 0.3, 0.2],
        "loss_weights": [],
    }
    trainer = SimpleNamespace(history=history, validation_frequency=2, use_adaptive_weights=False)
    plots.plot_training_history(trainer, save_path=str(tmp_path / "h.png"))
    traces = {trace.name: trace for trace in written[0].data}
    assert set(traces) == {"train_loss", "residual_loss", "val_loss"}
    assert list(traces["val_loss"].x) == [1, 3]


# ── 9. Fourier builds one hidden layer per hidden width ───────────────


@pytest.mark.parametrize("widths", [[16], [16, 16, 16], [32, 16]])
def test_fourier_hidden_layers_match_hidden_dims(widths):
    model = _model("fourier", {"hidden_dims": widths, "mapping_size": 8})
    linear = [m for m in model.model.layers if isinstance(m, torch.nn.Linear)]
    assert [layer.out_features for layer in linear] == [*widths, 1]
    assert linear[0].in_features == 16  # sin and cos of 8 random features


# ── 10. train(num_epochs=...) re-plans the schedule; L-BFGS phase is full batch ──


def _trainer(optimizer="adam", scheduler="cosine"):
    from tests.unit_tests.test_trainer import _make_config, _make_trainer

    config = _make_config(scheduler_type=scheduler)
    config.training.optimizer = optimizer
    config.training.num_epochs = 100
    return _make_trainer(config=config)


def test_cosine_schedule_follows_num_epochs_passed_to_train():
    trainer = _trainer()
    trainer.train(num_epochs=4, batch_size=32, num_points=32)
    assert trainer.scheduler.T_max == 4
    assert trainer.history["learning_rate"][-1] == pytest.approx(
        trainer.config.training.learning_rate_scheduler.min_lr, rel=1e-6
    )


def test_adam_lbfgs_switch_follows_num_epochs_passed_to_train():
    trainer = _trainer(optimizer="adam_lbfgs")
    trainer.train(num_epochs=10, batch_size=16, num_points=64)
    assert trainer._switch_epoch == 7  # 0.7 * 10, not 0.7 * 100
    assert trainer._is_lbfgs


def test_lbfgs_phase_uses_the_full_batch():
    trainer = _trainer(optimizer="adam_lbfgs")
    sizes = []
    original = trainer._sample_batch
    trainer._sample_batch = lambda n, progress: (sizes.append(n), original(n, progress))[1]
    trainer.train(num_epochs=4, batch_size=16, num_points=64)
    # 0.7 * 4 -> switch after epoch 2: two Adam epochs of 4 mini-batches, then full batches
    assert sizes == [16] * 8 + [64, 64]


def test_1d_solution_view_is_a_surface_even_with_a_time_slider():
    """The Collocation tab always passes a time; a 1-D slice used to be an n x 1 'surface'."""
    from pinnrl.plots import plot_solution
    from tests.unit_tests.test_physics_consistency import ExactModel, build

    pde = build("heat")
    exact_fig, predicted_fig = plot_solution(
        ExactModel(pde), pde, num_points=400, return_figs=True, time_point=0.5
    )
    assert exact_fig.data[0].z.shape == (20, 20)
    assert predicted_fig.data[0].z.shape == (20, 20)
