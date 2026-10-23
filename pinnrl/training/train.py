#!/usr/bin/env python3
"""Headless training script for PINNs-RL-PDE.

Launched by the dashboard or CLI (``pinnrl-train``). Reads config.yaml, accepts
PDE/architecture and hyperparameter overrides via CLI arguments, and runs
training to completion in a run folder under ``paths.results_dir`` (``./runs/``
by default). :func:`load_run` rebuilds a finished run.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

from pinnrl.config import (
    DEFAULT_CONFIG_PATH,
    Config,
    ModelConfig,
    RLConfig,
    TrainingConfig,
    resolve_device,
)
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes.pde_base import PDEConfig
from pinnrl.pdes.registry import PDE_SPECS, get_pde_spec
from pinnrl.training.trainer import PDETrainer


def _apply_well_dataset_defaults(config: dict, dataset_cfg: dict) -> dict:
    """Overlay a Well-dataset block onto the config in place.

    Always sets ``pde.observation_data`` to the Well spec. With
    ``use_defaults`` (the default) it also overwrites the PDE domain,
    time domain, dimension and input/output dims (and the model's
    input/output dims) with the registry entry's values, and sets
    ``training.mode`` to the entry's ``recommended_mode`` only if no mode is
    present. No-op when ``dataset_cfg`` has no ``name``.
    """
    from pinnrl.datasets import get_entry

    name = dataset_cfg.get("name")
    if not name:
        return config
    entry = get_entry(name)

    pde_block = config.setdefault("pde", {})
    pde_block["observation_data"] = {
        "source": "well",
        "name": name,
        "split": dataset_cfg.get("split", "train"),
        "n_traj": int(dataset_cfg.get("n_traj", 1)),
        "n_points": int(dataset_cfg.get("n_points", 4096)),
        "seed": int(dataset_cfg.get("seed", 0)),
        "base": dataset_cfg.get("base"),
    }

    if dataset_cfg.get("use_defaults", True):
        # Picking a Well dataset is an explicit "I want THIS dataset's
        # defaults" — overwrite the PDE shape fields the analytical config
        # populated, since they no longer match the data we'll fit.
        pde_block["domain"] = [list(b) for b in entry.domain]
        pde_block["time_domain"] = list(entry.time_domain)
        pde_block["dimension"] = entry.n_spatial_dims
        pde_block["input_dim"] = entry.default_input_dim
        pde_block["output_dim"] = entry.default_output_dim
        model_block = config.setdefault("model", {})
        model_block["input_dim"] = entry.default_input_dim
        model_block["output_dim"] = entry.default_output_dim
        # Mode is the one default we don't force: a user who passed
        # --mode inverse alongside --dataset wants the inverse-problem
        # objective, not the dataset's recommended_mode. setdefault keeps
        # any training.mode already present, including the one in config.yaml.
        training_block = config.setdefault("training", {})
        training_block.setdefault("mode", entry.recommended_mode)
    return config


def build_config_dict(
    yaml_config,
    pde_name,
    arch_type,
    use_rl=False,
    epochs=None,
    dataset=None,
):
    """Build a full config dict from yaml base + overrides.

    Args:
        yaml_config: Parsed ``config.yaml``.
        pde_name: PDE key or display name (resolved through the registry).
        arch_type: Architecture key from the ``architectures:`` block.
        use_rl: Enable the experimental RL bandit sampler.
        epochs: Optional override of ``training.num_epochs``.
        dataset: Optional Well dataset block, e.g.
            ``{"name": "active_matter", "split": "train", "n_traj": 1,
            "n_points": 4096, "seed": 0, "base": None,
            "use_defaults": True}``. When supplied, registry defaults are
            overlaid on the PDE/model blocks before training starts.

    Returns:
        The config dict: a shallow copy of ``yaml_config`` (its nested
        ``training`` and ``rl`` blocks are updated in place) with new ``pde``,
        ``model`` and ``pde_type`` entries.
    """
    config = yaml_config.copy()

    spec = get_pde_spec(pde_name)
    pde_name, pde_key = spec.display_name, spec.key
    pde_config = yaml_config.get("pde_configs", {}).get(pde_key, {})
    arch_config = yaml_config.get("architectures", {}).get(arch_type, {})

    if epochs is not None:
        config["training"]["num_epochs"] = epochs

    config["rl"]["enabled"] = use_rl

    config["pde"] = {
        "name": pde_name,
        "domain": pde_config.get("domain"),
        "time_domain": pde_config.get("time_domain"),
        "parameters": pde_config.get("parameters", {}),
        "boundary_conditions": pde_config.get("boundary_conditions", {}),
        "initial_condition": pde_config.get("initial_condition", {}),
        "exact_solution": pde_config.get("exact_solution", {}),
        "dimension": pde_config.get("dimension", 1),
        "input_dim": pde_config.get("input_dim", 2),
        "output_dim": pde_config.get("output_dim", 1),
        "architecture": arch_type,
    }

    hidden_dim = arch_config.get("hidden_dim", 128)
    if "hidden_dims" in arch_config:
        hidden_dim = arch_config["hidden_dims"][0]

    config["model"] = {
        "architecture": arch_type,
        "input_dim": pde_config.get("input_dim", 2),
        "hidden_dim": hidden_dim,
        "output_dim": pde_config.get("output_dim", 1),
        "num_layers": arch_config.get("num_layers", len(arch_config.get("hidden_dims", [128] * 4))),
        **arch_config,
    }

    config["pde_type"] = pde_key

    if dataset:
        _apply_well_dataset_defaults(config, dataset)

    return config


def create_pde(config_dict, device):
    """Create the PDE described by a ``build_config_dict`` result.

    In inverse mode with trainable parameters but no observation data,
    synthetic noisy observations are generated (``inverse.obs_points``,
    ``obs_noise``, ``obs_seed``; defaults 200, 0.01, 0).
    """
    pde_cls = get_pde_spec(config_dict["pde"]["name"]).cls

    training_cfg = config_dict["training"]
    pde_cfg = config_dict["pde"]

    pde_config = PDEConfig(
        name=pde_cfg["name"],
        domain=pde_cfg["domain"],
        time_domain=pde_cfg["time_domain"],
        parameters=pde_cfg.get("parameters", {}),
        boundary_conditions=pde_cfg["boundary_conditions"],
        initial_condition=pde_cfg["initial_condition"],
        exact_solution=pde_cfg["exact_solution"],
        dimension=pde_cfg["dimension"],
        device=device,
        training=TrainingConfig.from_dict(training_cfg),
        trainable_parameters=list(pde_cfg.get("trainable_parameters", []) or []),
        parameter_initial_guesses=dict(pde_cfg.get("parameter_initial_guesses", {}) or {}),
        observation_data=pde_cfg.get("observation_data"),
    )

    pde = pde_cls(config=pde_config)

    # If the user requested inverse mode without supplying real observations,
    # generate synthetic noisy observations from the analytical solution so the
    # data-fitting loss has something to anchor the parameter recovery to.
    mode = training_cfg.get("mode", "forward")
    inverse_cfg = config_dict.get("inverse", {})
    if mode == "inverse" and pde.observation_data is None and pde_config.trainable_parameters:
        n_obs = int(inverse_cfg.get("obs_points", 200))
        noise = float(inverse_cfg.get("obs_noise", 0.01))
        seed = int(inverse_cfg.get("obs_seed", 0))
        pde.generate_synthetic_observations(n_points=n_obs, noise_std=noise, seed=seed)
    return pde


def build_model(config_dict, device):
    """Build the PINNModel described by a ``build_config_dict`` result.

    Returns:
        ``(model, config)`` with the model on ``device`` and the
        :class:`~pinnrl.config.Config` it was built from.
    """
    arch_type = config_dict["model"]["architecture"]
    config_obj = Config()
    config_obj.device = device
    config_obj.model = ModelConfig.from_architecture(
        arch_type,
        config_dict["architectures"].get(arch_type, {}),
        input_dim=config_dict["model"]["input_dim"],
        output_dim=config_dict["model"]["output_dim"],
    )
    config_obj.training = TrainingConfig.from_dict(config_dict["training"])
    return PINNModel(config=config_obj, device=device).to(device), config_obj


def load_run(run_dir, device=None):
    """Rebuild the PDE and trained model of a finished run.

    Reads the run's ``config.yaml`` snapshot and ``final_model.pt`` and goes
    through the same builders as :func:`run_training`, so the result matches
    what was trained. The PDE is the forward problem at the configured (true)
    parameters, so its exact solution is the reference even for inverse runs;
    observation data is not reloaded.

    Args:
        run_dir: Run folder written by :func:`run_training`.
        device: Torch device; defaults to CPU.

    Returns:
        ``(pde, model, config_dict)`` with the model in eval mode.
    """
    device = device or torch.device("cpu")
    run_dir = Path(run_dir)
    with open(run_dir / "config.yaml") as f:
        config_dict = yaml.safe_load(f)
    pde_only = {
        **config_dict,
        "pde": {
            **config_dict["pde"],
            "observation_data": None,
            "trainable_parameters": [],
            "parameter_initial_guesses": {},
        },
    }
    pde_only["training"] = {**config_dict["training"], "mode": "forward"}
    pde = create_pde(pde_only, device)
    model, _ = build_model(config_dict, device)
    model.load_state_dict(torch.load(run_dir / "final_model.pt", map_location=device))
    model.eval()
    return pde, model, config_dict


def run_training(config_dict, device):
    """Run a full training session.

    Creates ``<paths.results_dir>/<timestamp>_<pde name or dataset>_<arch>_<rl|no_rl>``,
    writes the ``config.yaml`` snapshot there and trains into it. On failure
    the error is recorded in ``metadata.json`` (``status: failed``) and re-raised.
    """
    arch_type = config_dict["model"]["architecture"]
    pde_name = config_dict["pde"]["name"]
    rl_enabled = config_dict["rl"]["enabled"]

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rl_status = "rl" if rl_enabled else "no_rl"
    obs = config_dict.get("pde", {}).get("observation_data") or {}
    dataset_tag = obs.get("name") if isinstance(obs, dict) and obs.get("source") == "well" else None
    if dataset_tag:
        experiment_name = f"{timestamp}_{dataset_tag}_{arch_type}_{rl_status}"
    else:
        experiment_name = f"{timestamp}_{pde_name}_{arch_type}_{rl_status}"
    experiment_dir = Path(config_dict["paths"]["results_dir"]) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # The config snapshot is what load_run() rebuilds the run from. Everything
    # else in the run dir (.running, metadata.json, weights, plots) is the
    # trainer's job.
    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f)

    print(f"Experiment: {experiment_name}")
    print(f"Directory: {experiment_dir}")

    try:
        # Create PDE
        pde = create_pde(config_dict, device)

        model, config_obj = build_model(config_dict, device)

        # Experimental RL bandit sampler (1-D PDEs) when --rl / rl.enabled is set
        rl_config = RLConfig.from_dict(config_dict["rl"])
        bandit_sampler = rl_config.build_sampler(pde, device) if rl_config.enabled else None

        # Create trainer
        trainer = PDETrainer(
            model=model,
            pde=pde,
            config=config_obj,
            device=device,
            bandit_sampler=bandit_sampler,
        )

        # The trainer writes final_model.pt, history/metrics and the completed metadata.
        trainer.train(experiment_dir=str(experiment_dir))

        print("Training completed successfully.")

    except Exception as e:
        # Record the failure for the dashboard, then let the traceback surface.
        metadata_path = experiment_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
        metadata.update(
            status="failed",
            error=str(e),
            pde_name=pde_name,
            architecture=arch_type,
            end_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
        metadata_path.write_text(json.dumps(metadata, indent=2))
        raise
    finally:
        (experiment_dir / ".running").unlink(missing_ok=True)


def main():
    """Entry point of ``pinnrl-train``: parse flags, build the config and train."""
    # Headless CLI: save figures without a GUI backend. Library imports leave the
    # user's backend alone (inline plots in notebooks keep working).
    import matplotlib

    matplotlib.use("agg")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Train a PINN model")
    parser.add_argument(
        "--pde",
        required=True,
        help="PDE config key or display name (e.g. 'heat' or 'Heat Equation')",
    )
    parser.add_argument("--arch", required=True, help="Architecture (e.g. 'fourier')")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument(
        "--collocation-points", type=int, default=None, help="Override collocation points"
    )
    parser.add_argument(
        "--boundary-points", type=int, default=None, help="Override boundary points"
    )
    parser.add_argument("--initial-points", type=int, default=None, help="Override initial points")
    parser.add_argument(
        "--rl",
        action="store_true",
        help="Enable the experimental RL bandit collocation sampler (1-D PDEs)",
    )
    parser.add_argument(
        "--collocation-distribution",
        choices=["uniform", "stratified", "residual_based"],
        default=None,
        help=("Collocation point sampling strategy. Overridden to 'adaptive' when --rl is set."),
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "lbfgs", "adam_lbfgs"],
        default=None,
        help="Optimizer to use",
    )
    parser.add_argument(
        "--mode",
        choices=["forward", "inverse", "data_only", "data_augmented"],
        default=None,
        help=(
            "Training mode: forward (residual + IC/BC), inverse (residual + "
            "IC/BC + data + trainable parameters), data_only (regression on "
            "observed snapshots only), data_augmented (residual + IC/BC + data)."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Well dataset name (registered in pinnrl.datasets.WELL_REGISTRY)",
    )
    parser.add_argument(
        "--dataset-split", default="train", help="Well dataset split (train/valid/test)"
    )
    parser.add_argument(
        "--dataset-traj",
        type=int,
        default=1,
        help="Number of WellDataset samples to draw (random, without replacement)",
    )
    parser.add_argument(
        "--dataset-points",
        type=int,
        default=4096,
        help="Number of (x, t, u) points to sub-sample from the loaded slice",
    )
    parser.add_argument("--dataset-seed", type=int, default=0, help="RNG seed for sub-sampling")
    parser.add_argument(
        "--dataset-base",
        default=None,
        help="Local Well download dir; omit to stream from Hugging Face",
    )
    parser.add_argument(
        "--identify",
        action="append",
        default=[],
        help="Name of a PDE parameter to identify in inverse mode (repeatable)",
    )
    parser.add_argument(
        "--initial-guess",
        action="append",
        default=[],
        help="Initial guess for an identifiable parameter, e.g. 'alpha=0.5' (repeatable)",
    )
    parser.add_argument(
        "--obs-path",
        default=None,
        help="Path to an .npz file with observation data (keys 'x','t','u')",
    )
    parser.add_argument(
        "--obs-noise",
        type=float,
        default=None,
        help="Stddev of Gaussian noise added to synthetic observations",
    )
    parser.add_argument(
        "--obs-points",
        type=int,
        default=None,
        help="Number of synthetic observation points to generate",
    )
    parser.add_argument(
        "--loss-function",
        choices=["mse", "mae", "huber"],
        default=None,
        help="Reduction for residual/BC/IC/data losses",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=None,
        help="Delta for Huber loss (only used when --loss-function=huber)",
    )
    parser.add_argument(
        "--adaptive-weights",
        choices=["off", "lrw", "rbw"],
        default=None,
        help=(
            "Adaptive loss-term weighting: 'off' for static weights, "
            "'lrw' (gradient-balanced), or 'rbw' (loss-balanced)."
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=["cosine", "reduce_lr"],
        default=None,
        help=(
            "Learning-rate scheduler: 'cosine' annealing or 'reduce_lr' on a plateau "
            "of the epoch training loss."
        ),
    )
    parser.add_argument(
        "--early-stopping",
        choices=["on", "off"],
        default=None,
        help="Toggle early stopping on validation loss.",
    )
    parser.add_argument(
        "--gradient-clipping",
        type=float,
        default=None,
        help="Max gradient norm; pass 0 to disable clipping.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        help="L2 regularisation coefficient on optimiser weights.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to config.yaml")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    try:
        get_pde_spec(args.pde)
    except KeyError:
        print(f"Unknown PDE: {args.pde}")
        print(f"Available: {', '.join(spec.key for spec in PDE_SPECS)}")
        sys.exit(1)

    with open(args.config) as f:
        yaml_config = yaml.safe_load(f)

    if args.device:
        yaml_config["device"] = args.device

    # Apply hyperparameter overrides from CLI
    if args.lr is not None:
        yaml_config.setdefault("training", {}).setdefault("optimizer_config", {})[
            "learning_rate"
        ] = args.lr
    if args.batch_size is not None:
        yaml_config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.collocation_points is not None:
        yaml_config.setdefault("training", {})["num_collocation_points"] = args.collocation_points
    if args.boundary_points is not None:
        yaml_config.setdefault("training", {})["num_boundary_points"] = args.boundary_points
    if args.initial_points is not None:
        yaml_config.setdefault("training", {})["num_initial_points"] = args.initial_points
    if args.collocation_distribution is not None:
        yaml_config.setdefault("training", {})["collocation_distribution"] = (
            args.collocation_distribution
        )
    if args.optimizer is not None:
        yaml_config.setdefault("training", {})["optimizer"] = args.optimizer
    if args.mode is not None:
        yaml_config.setdefault("training", {})["mode"] = args.mode
    if args.loss_function is not None:
        yaml_config.setdefault("training", {})["loss_function"] = args.loss_function
    if args.huber_delta is not None:
        yaml_config.setdefault("training", {})["huber_delta"] = args.huber_delta
    if args.adaptive_weights is not None:
        aw_block = yaml_config.setdefault("training", {}).setdefault("adaptive_weights", {})
        if args.adaptive_weights == "off":
            aw_block["enabled"] = False
        else:
            aw_block["enabled"] = True
            aw_block["strategy"] = args.adaptive_weights
    if args.lr_scheduler is not None:
        yaml_config.setdefault("training", {})["scheduler_type"] = args.lr_scheduler
    if args.early_stopping is not None:
        es_block = yaml_config.setdefault("training", {}).setdefault("early_stopping", {})
        es_block["enabled"] = args.early_stopping == "on"
    if args.gradient_clipping is not None:
        yaml_config.setdefault("training", {})["gradient_clipping"] = args.gradient_clipping
    if args.weight_decay is not None:
        yaml_config.setdefault("training", {}).setdefault("optimizer_config", {})[
            "weight_decay"
        ] = args.weight_decay

    device = resolve_device(yaml_config.get("device", "cpu"))
    dataset_block = None
    if args.dataset:
        dataset_block = {
            "name": args.dataset,
            "split": args.dataset_split,
            "n_traj": args.dataset_traj,
            "n_points": args.dataset_points,
            "seed": args.dataset_seed,
            "base": args.dataset_base,
            "use_defaults": True,
        }
    config_dict = build_config_dict(
        yaml_config, args.pde, args.arch, args.rl, args.epochs, dataset=dataset_block
    )
    config_dict["device"] = str(device)

    # Wire inverse-problem flags through to the PDE config + run-time options.
    if args.identify:
        config_dict["pde"]["trainable_parameters"] = list(args.identify)
    if args.initial_guess:
        guesses = {}
        for spec in args.initial_guess:
            if "=" not in spec:
                print(f"Ignoring malformed --initial-guess '{spec}' (expected name=value)")
                continue
            name, value = spec.split("=", 1)
            try:
                guesses[name.strip()] = float(value)
            except ValueError:
                print(f"Ignoring non-numeric --initial-guess '{spec}'")
        if guesses:
            config_dict["pde"]["parameter_initial_guesses"] = guesses
    if args.obs_path:
        config_dict["pde"]["observation_data"] = {"path": args.obs_path}
    inverse_runtime = {}
    if args.obs_noise is not None:
        inverse_runtime["obs_noise"] = args.obs_noise
    if args.obs_points is not None:
        inverse_runtime["obs_points"] = args.obs_points
    if inverse_runtime:
        config_dict["inverse"] = inverse_runtime

    run_training(config_dict, device)


if __name__ == "__main__":
    main()
