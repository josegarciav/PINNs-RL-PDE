"""``_apply_well_dataset_defaults`` should respect a user-supplied training mode.

The function fills in domain / dimension / output_dim from the registry
entry, but ``training.mode`` is special: a user passing ``--mode inverse``
alongside ``--dataset`` wants to recover trainable parameters, not run the
dataset's default ``data_only`` regression. The fix keeps an explicit mode
intact while still defaulting unset modes to the registry's recommendation.
"""

from pinnrl.training.train import _apply_well_dataset_defaults


def _dataset_cfg():
    return {
        "name": "active_matter",
        "split": "train",
        "n_traj": 1,
        "n_points": 64,
        "seed": 0,
        "base": None,
        "use_defaults": True,
    }


def test_default_mode_is_taken_from_registry_when_user_did_not_set_one():
    config: dict = {}
    _apply_well_dataset_defaults(config, _dataset_cfg())
    # active_matter has no analytical PDE, so the registry says data_only.
    assert config["training"]["mode"] == "data_only"


def test_user_supplied_inverse_mode_is_preserved():
    config = {"training": {"mode": "inverse"}}
    _apply_well_dataset_defaults(config, _dataset_cfg())
    assert config["training"]["mode"] == "inverse"


def test_user_supplied_forward_mode_is_preserved():
    config = {"training": {"mode": "forward"}}
    _apply_well_dataset_defaults(config, _dataset_cfg())
    assert config["training"]["mode"] == "forward"


def test_pde_shape_defaults_still_applied_alongside_user_mode():
    """Mode must be sticky, but the rest of the dataset defaults should still flow in."""
    config = {"training": {"mode": "inverse"}}
    _apply_well_dataset_defaults(config, _dataset_cfg())
    assert config["pde"]["dimension"] == 2
    assert config["pde"]["output_dim"] == 11
    assert config["model"]["output_dim"] == 11
    assert config["pde"]["observation_data"]["source"] == "well"
