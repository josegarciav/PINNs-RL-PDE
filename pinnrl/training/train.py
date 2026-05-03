#!/usr/bin/env python3
"""Headless training script for PINNs-RL-PDE.

Launched by the dashboard or CLI. Reads config.yaml, accepts PDE/architecture
overrides via CLI arguments, and runs training to completion.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pinnrl.config import (
    AdaptiveWeightsConfig,
    Config,
    EarlyStoppingConfig,
    LBFGSConfig,
    LearningRateSchedulerConfig,
    ModelConfig,
    TrainingConfig,
)
from pinnrl.neural_networks import PINNModel
from pinnrl.pdes.pde_base import PDEConfig
from pinnrl.rl.rl_agent import RLAgent
from pinnrl.training.trainer import PDETrainer

# PDE name -> (module, class, config key)
PDE_REGISTRY = {
    "Heat Equation": ("pinnrl.pdes.heat_equation", "HeatEquation", "heat"),
    "Heat Equation 2D": ("pinnrl.pdes.heat_equation", "HeatEquation", "heat_2d"),
    "Burgers Equation": ("pinnrl.pdes.burgers_equation", "BurgersEquation", "burgers"),
    "Wave Equation": ("pinnrl.pdes.wave_equation", "WaveEquation", "wave"),
    "Convection Equation": ("pinnrl.pdes.convection_equation", "ConvectionEquation", "convection"),
    "KdV Equation": ("pinnrl.pdes.kdv_equation", "KdVEquation", "kdv"),
    "Pendulum Equation": ("pinnrl.pdes.pendulum_equation", "PendulumEquation", "pendulum"),
    "Allen-Cahn Equation": ("pinnrl.pdes.allen_cahn", "AllenCahnEquation", "allen_cahn"),
    "Cahn-Hilliard Equation": (
        "pinnrl.pdes.cahn_hilliard",
        "CahnHilliardEquation",
        "cahn_hilliard",
    ),
    "Black-Scholes Equation": (
        "pinnrl.pdes.black_scholes",
        "BlackScholesEquation",
        "black_scholes",
    ),
}


def _build_training_config(training_cfg: dict) -> TrainingConfig:
    """Construct a TrainingConfig from a config dict (shared by create_pde and run_training)."""
    lbfgs_cfg_dict = training_cfg.get("lbfgs", {})
    return TrainingConfig(
        num_epochs=training_cfg["num_epochs"],
        batch_size=training_cfg["batch_size"],
        num_collocation_points=training_cfg["num_collocation_points"],
        num_boundary_points=training_cfg["num_boundary_points"],
        num_initial_points=training_cfg["num_initial_points"],
        learning_rate=training_cfg["optimizer_config"]["learning_rate"],
        weight_decay=training_cfg["optimizer_config"].get("weight_decay", 0.0001),
        gradient_clipping=training_cfg.get("gradient_clipping", 1.0),
        early_stopping=EarlyStoppingConfig(
            enabled=training_cfg["early_stopping"]["enabled"],
            patience=training_cfg["early_stopping"]["patience"],
            min_delta=training_cfg["early_stopping"]["min_delta"],
        ),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type=training_cfg["scheduler_type"],
            warmup_epochs=training_cfg.get("warmup_epochs", 0),
            min_lr=training_cfg["reduce_lr_params"]["min_lr"],
            factor=training_cfg["reduce_lr_params"]["factor"],
            patience=training_cfg["reduce_lr_params"]["patience"],
        ),
        adaptive_weights=AdaptiveWeightsConfig(
            enabled=training_cfg["adaptive_weights"]["enabled"],
            strategy=training_cfg["adaptive_weights"]["strategy"],
            alpha=training_cfg["adaptive_weights"]["alpha"],
            eps=training_cfg["adaptive_weights"]["eps"],
        ),
        loss_weights=training_cfg["loss_weights"],
        optimizer=training_cfg.get("optimizer", "adam"),
        adam_lbfgs_switch_ratio=training_cfg.get("adam_lbfgs_switch_ratio", 0.7),
        lbfgs=LBFGSConfig(
            history_size=lbfgs_cfg_dict.get("history_size", 50),
            max_iter=lbfgs_cfg_dict.get("max_iter", 20),
            line_search_fn=lbfgs_cfg_dict.get("line_search_fn", "strong_wolfe"),
            tolerance_grad=lbfgs_cfg_dict.get("tolerance_grad", 1e-7),
            tolerance_change=lbfgs_cfg_dict.get("tolerance_change", 1e-9),
        ),
        mode=training_cfg.get("mode", "forward"),
        loss_function=training_cfg.get("loss_function", "mse"),
        huber_delta=training_cfg.get("huber_delta", 1.0),
    )


def _apply_well_dataset_defaults(config: dict, dataset_cfg: dict) -> dict:
    """Overlay a Well-dataset block onto the config in place.

    Adds an ``observation_data`` Well spec under the active PDE and, when
    the dataset has a matched analytical PDE in the registry, fills in the
    domain / time / dimension / output_dim defaults that a fresh user
    would otherwise have to enter by hand. The function is a no-op for
    config keys the user has already populated.
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
        # objective, not the dataset's recommended_mode. Only fill it in
        # when the caller hasn't already set one.
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
        dataset: Optional Well dataset block, e.g.
            ``{"name": "active_matter", "split": "train", "n_traj": 1,
            "n_points": 4096, "seed": 0, "base": None,
            "use_defaults": True}``. When supplied, registry defaults are
            overlaid on the PDE/model blocks before training starts.
    """
    config = yaml_config.copy()

    pde_key = PDE_REGISTRY[pde_name][2]
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
    """Create a PDE instance from config dict."""
    pde_name = config_dict["pde"]["name"]
    module_path, cls_name, _ = PDE_REGISTRY[pde_name]

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
        training=_build_training_config(training_cfg),
        trainable_parameters=list(pde_cfg.get("trainable_parameters", []) or []),
        parameter_initial_guesses=dict(pde_cfg.get("parameter_initial_guesses", {}) or {}),
        observation_data=pde_cfg.get("observation_data"),
    )

    mod = __import__(module_path, fromlist=[cls_name])
    pde_cls = getattr(mod, cls_name)
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


def run_training(config_dict, device):
    """Run a full training session."""
    arch_type = config_dict["model"]["architecture"]
    arch_config = config_dict["architectures"].get(arch_type, {})
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
    (experiment_dir / "visualizations").mkdir(exist_ok=True)

    # Marker file
    running_file = experiment_dir / ".running"
    running_file.touch()

    # Save config
    with open(experiment_dir / "config.yaml", "w") as f:
        yaml.dump(config_dict, f)

    # Initial metadata
    metadata = {
        "status": "running",
        "pde": pde_name,
        "architecture": arch_type,
        "rl_enabled": rl_enabled,
        "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "num_epochs": config_dict["training"]["num_epochs"],
    }
    with open(experiment_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Experiment: {experiment_name}")
    print(f"Directory: {experiment_dir}")

    try:
        # Create PDE
        pde = create_pde(config_dict, device)

        # Create model config
        config_obj = Config()
        config_obj.device = device

        hidden_dim = config_dict["model"].get("hidden_dim", 128)
        config_obj.model = ModelConfig(
            input_dim=config_dict["model"]["input_dim"],
            hidden_dim=hidden_dim,
            output_dim=config_dict["model"]["output_dim"],
            num_layers=config_dict["model"].get("num_layers", 4),
            activation=arch_config.get("activation", "tanh"),
            fourier_features=arch_type == "fourier",
            fourier_scale=arch_config.get("scale", 1.0) if arch_type == "fourier" else None,
            dropout=arch_config.get("dropout", 0.0),
            layer_norm=arch_config.get("layer_norm", False),
            architecture=arch_type,
        )

        # Architecture-specific params
        if arch_type == "resnet":
            config_obj.model.num_blocks = arch_config.get("num_blocks", 4)
        for key in [
            "mapping_size",
            "scale",
            "omega_0",
            "num_heads",
            "hidden_dims",
            "latent_dim",
            "modes",
            "periodic",
        ]:
            if key in arch_config:
                setattr(config_obj.model, key, arch_config[key])

        # Training config (shared builder keeps create_pde and run_training in sync)
        training_cfg = config_dict["training"]
        config_obj.training = _build_training_config(training_cfg)

        # Create model
        model = PINNModel(config=config_obj, device=device).to(device)

        # Create RL agent if enabled
        rl_agent = None
        if rl_enabled:
            rl_cfg = config_dict["rl"]
            rl_agent = RLAgent(
                state_dim=rl_cfg["state_dim"],
                action_dim=rl_cfg["action_dim"],
                hidden_dim=rl_cfg["hidden_dim"],
                learning_rate=rl_cfg["learning_rate"],
                gamma=rl_cfg["gamma"],
                epsilon_start=rl_cfg["epsilon_start"],
                epsilon_end=rl_cfg["epsilon_end"],
                epsilon_decay=rl_cfg["epsilon_decay"],
                memory_size=rl_cfg["memory_size"],
                batch_size=rl_cfg["batch_size"],
                target_update=rl_cfg["target_update"],
                reward_weights=rl_cfg["reward_weights"],
                device=device,
            )

        # Create trainer
        trainer = PDETrainer(
            model=model,
            pde=pde,
            optimizer_config=training_cfg["optimizer_config"],
            config=config_obj,
            device=device,
            rl_agent=rl_agent,
            validation_frequency=training_cfg["validation_frequency"],
            early_stopping_config=training_cfg["early_stopping"],
        )

        # Train
        history = trainer.train(
            num_epochs=config_obj.training.num_epochs,
            batch_size=config_obj.training.batch_size,
            num_points=config_obj.training.num_collocation_points,
            experiment_dir=str(experiment_dir),
        )

        # Save final model
        torch.save(model.state_dict(), experiment_dir / "final_model.pt")

        # Save metrics
        from pinnrl.utils.utils import save_training_metrics

        final_metadata = {
            **metadata,
            "status": "completed",
            "end_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_training_metrics(history, str(experiment_dir), final_metadata)

        print("Training completed successfully.")

    except Exception as e:
        print(f"Training error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        metadata["status"] = "failed"
        metadata["error"] = str(e)
        with open(experiment_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    finally:
        if running_file.exists():
            running_file.unlink()


def main():
    parser = argparse.ArgumentParser(description="Train a PINN model")
    parser.add_argument("--pde", required=True, help="PDE name (e.g. 'Heat Equation')")
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
    parser.add_argument("--rl", action="store_true", help="Enable RL adaptive sampling")
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
        "--dataset-traj", type=int, default=1, help="Number of trajectories to load"
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
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    default_config = os.path.join(project_root, "src", "config", "config.yaml")
    if not os.path.exists(default_config):
        default_config = os.path.join(project_root, "config.yaml")
    parser.add_argument("--config", default=default_config, help="Path to config.yaml")
    parser.add_argument("--device", default=None, help="Device (cpu, mps, cuda)")
    args = parser.parse_args()

    if args.pde not in PDE_REGISTRY:
        print(f"Unknown PDE: {args.pde}")
        print(f"Available: {', '.join(PDE_REGISTRY.keys())}")
        sys.exit(1)

    with open(args.config, "r") as f:
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
    if args.optimizer is not None:
        yaml_config.setdefault("training", {})["optimizer"] = args.optimizer
    if args.mode is not None:
        yaml_config.setdefault("training", {})["mode"] = args.mode
    if args.loss_function is not None:
        yaml_config.setdefault("training", {})["loss_function"] = args.loss_function
    if args.huber_delta is not None:
        yaml_config.setdefault("training", {})["huber_delta"] = args.huber_delta

    device = torch.device(yaml_config.get("device", "cpu"))
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
