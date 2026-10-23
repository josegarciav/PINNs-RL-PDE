"""``_apply_well_dataset_defaults`` and the training mode.

The function fills in domain / dimension / output_dim from the registry
entry and sets ``training.mode`` to the dataset's recommendation. An explicit
choice (``--mode`` alongside ``--dataset``) travels in the dataset block as
``mode`` and wins. A mode already in the config does not count as a choice:
``config.yaml`` always has one (``forward``), so treating it as explicit meant
the recommendation never applied.
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


def test_explicit_inverse_mode_is_preserved():
    config: dict = {}
    _apply_well_dataset_defaults(config, {**_dataset_cfg(), "mode": "inverse"})
    assert config["training"]["mode"] == "inverse"


def test_config_yaml_forward_mode_is_replaced_by_the_recommendation():
    config = {"training": {"mode": "forward"}}
    _apply_well_dataset_defaults(config, _dataset_cfg())
    assert config["training"]["mode"] == "data_only"


def test_pde_shape_defaults_still_applied_alongside_user_mode():
    """An explicit mode does not stop the rest of the dataset defaults."""
    config: dict = {}
    _apply_well_dataset_defaults(config, {**_dataset_cfg(), "mode": "inverse"})
    assert config["pde"]["dimension"] == 2
    assert config["pde"]["output_dim"] == 11
    assert config["model"]["output_dim"] == 11
    assert config["pde"]["observation_data"]["source"] == "well"
