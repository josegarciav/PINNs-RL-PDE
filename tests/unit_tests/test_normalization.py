"""Non-dimensionalization: the model's affine maps, the loss in field units, a problem in
physical units trained normalized vs raw, and the CLI / load_run round trip (roadmap v0.6 item 5)."""

import math

import pytest
import torch
import yaml

from pinnrl import CustomPDE
from pinnrl.training.train import build_config_dict, create_pde, load_run, run_training

L, T, ALPHA = 1000.0, 3600.0, 50.0  # millimetres, seconds, mm^2/s


def _physical(**kwargs):
    kwargs.setdefault("scales", {"u": (300.0, 50.0)})
    return CustomPDE(
        "u_t - alpha*u_xx",
        domain={"x": (0, L), "t": (0, T)},
        parameters={"alpha": ALPHA},
        initial_condition=f"300 + 50*sin(pi*x/{L})",
        boundary_conditions={"dirichlet": {"value": 300.0}},
        exact_solution=f"300 + 50*exp(-alpha*pi^2*t/{L}^2)*sin(pi*x/{L})",
        **kwargs,
    )


def test_model_maps_inputs_to_the_unit_cube_and_outputs_to_physical_units():
    pde = _physical()
    trainer = pde.build_trainer("feedforward", model={"hidden_dims": [8, 8]}, normalize=True)
    model = trainer.model
    assert model.scaled
    assert trainer.config.model.input_bounds == [[0.0, L], [0.0, T]]
    assert trainer.config.model.output_offset == [300.0] and trainer.config.model.output_scale == [
        50.0
    ]
    x = torch.tensor([[0.0, 0.0], [L, T], [L / 2, T / 4]])
    inner = model.model(2.0 * (x - model._in_lo) / model._in_span - 1.0)
    assert torch.allclose(model(x), 300.0 + 50.0 * inner)
    # without normalize the model is the bare network
    raw = _physical().build_trainer("feedforward", model={"hidden_dims": [8, 8]}).model
    assert not raw.scaled and torch.allclose(raw(x), raw.model(x))
    with pytest.raises(ValueError, match="not a field"):
        _physical(scales={"v": 1.0})
    with pytest.raises(ValueError, match="positive"):
        _physical(scales={"u": -1.0})


def test_loss_terms_are_in_field_units_when_normalized():
    pde = _physical()
    net = torch.nn.Sequential(torch.nn.Linear(2, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
    x, t = pde.generate_collocation_points(64)
    pde.build_trainer("feedforward", model={"hidden_dims": [8, 8]}, normalize=True)
    pde._normalized = False  # the same points and options, in physical units
    raw = pde.compute_loss(net, x, t)
    pde._normalized = True
    assert pde._multipliers("error") == [1 / 50.0]
    assert pde._multipliers("residual") == [T / 50.0]
    scaled = pde.compute_loss(net, x, t)
    assert scaled["initial"].item() == pytest.approx(raw["initial"].item() / 50.0**2, rel=1e-5)
    assert scaled["boundary"].item() == pytest.approx(raw["boundary"].item() / 50.0**2, rel=1e-5)
    assert scaled["residual"].item() == pytest.approx(
        raw["residual"].item() * (T / 50.0) ** 2, rel=1e-4
    )
    pde.generate_synthetic_observations(20, 0.0, seed=0)
    scaled_data = pde.compute_loss(net, x, t)["data"].item()
    pde._normalized = False
    assert scaled_data == pytest.approx(
        pde.compute_loss(net, x, t)["data"].item() / 50.0**2, rel=1e-5
    )


def test_physical_units_train_like_the_unit_problem_when_normalized():
    torch.manual_seed(0)
    errors = {}
    for normalize in (False, True):
        torch.manual_seed(0)
        pde = _physical()
        trainer = pde.build_trainer(
            "feedforward",
            model={"hidden_dims": [32, 32, 32], "layer_norm": False},
            normalize=normalize,
            num_collocation_points=512,
            batch_size=512,
            num_boundary_points=128,
            num_initial_points=128,
            learning_rate=3e-3,
            scheduler_type="constant",
            early_stopping={"enabled": False},
        )
        trainer.train_steps(num_steps=600, resample_every=50)
        errors[normalize] = pde.validate(trainer.model, 2000)["max_error"]
    assert errors[True] < 10.0  # kelvin, on a 50 K amplitude
    assert errors[False] > 5 * errors[True]


def test_cli_normalize_round_trips_through_the_run_snapshot(tmp_path):
    from pinnrl.config import DEFAULT_CONFIG_PATH

    spec = _physical().to_dict()
    assert spec["scales"] == {"u": [300.0, 50.0]}
    with open(DEFAULT_CONFIG_PATH) as f:
        base = yaml.safe_load(f)
    config = build_config_dict(base, None, "feedforward", custom=spec, epochs=2, normalize=True)
    assert config["model"]["input_bounds"] == [[0.0, L], [0.0, T]]
    assert config["model"]["output_offset"] == [300.0] and config["model"]["output_scale"] == [50.0]
    config["training"].update({"num_collocation_points": 64, "batch_size": 64})
    config["model"]["hidden_dims"] = [8, 8]
    config["paths"]["results_dir"] = str(tmp_path)
    pde = create_pde(config, torch.device("cpu"))
    assert pde._normalized
    run_training(config, torch.device("cpu"))
    run_dir = next(p for p in tmp_path.iterdir() if p.is_dir())
    pde2, model, _ = load_run(run_dir)
    assert model.scaled and torch.allclose(model._out_offset, torch.tensor([[300.0]]))
    x = torch.tensor([[L / 3, T / 2]])
    assert math.isfinite(model(x).item())
