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


def build_config_dict(yaml_config, pde_name, arch_type, use_rl=False, epochs=None):
    """Build a full config dict from yaml base + overrides."""
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
        training=TrainingConfig(
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
        ),
    )

    mod = __import__(module_path, fromlist=[cls_name])
    pde_cls = getattr(mod, cls_name)
    return pde_cls(config=pde_config)


def run_training(config_dict, device):
    """Run a full training session."""
    arch_type = config_dict["model"]["architecture"]
    arch_config = config_dict["architectures"].get(arch_type, {})
    pde_name = config_dict["pde"]["name"]
    rl_enabled = config_dict["rl"]["enabled"]

    # Experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rl_status = "rl" if rl_enabled else "no_rl"
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

        # Training config
        training_cfg = config_dict["training"]
        config_obj.training = TrainingConfig(
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
        )

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

    device = torch.device(yaml_config.get("device", "cpu"))
    config_dict = build_config_dict(yaml_config, args.pde, args.arch, args.rl, args.epochs)
    config_dict["device"] = str(device)

    run_training(config_dict, device)


if __name__ == "__main__":
    main()
