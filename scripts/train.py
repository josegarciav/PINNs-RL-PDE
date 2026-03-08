"""CLI training script for PINNs-RL-PDE.

Usage:
    python scripts/train.py --pde heat --arch fourier --epochs 3000
    python scripts/train.py --pde burgers --arch resnet --epochs 5000 --device cpu
    python scripts/train.py --pde wave --arch siren --epochs 3000 --lr 0.005
"""

import argparse
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from src.config import (
    Config,
    ModelConfig,
    TrainingConfig,
    EarlyStoppingConfig,
    LearningRateSchedulerConfig,
    AdaptiveWeightsConfig,
)
from src.neural_networks import PINNModel
from src.trainer import PDETrainer
from src.pdes.pde_base import PDEConfig
from src.pdes.heat_equation import HeatEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.pendulum_equation import PendulumEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.convection_equation import ConvectionEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.black_scholes import BlackScholesEquation

PDE_CLASSES = {
    "heat": HeatEquation,
    "burgers": BurgersEquation,
    "wave": WaveEquation,
    "pendulum": PendulumEquation,
    "kdv": KdVEquation,
    "convection": ConvectionEquation,
    "allen_cahn": AllenCahnEquation,
    "cahn_hilliard": CahnHilliardEquation,
    "black_scholes": BlackScholesEquation,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PINN on a PDE")
    parser.add_argument(
        "--pde",
        type=str,
        default="heat",
        choices=list(PDE_CLASSES.keys()),
        help="PDE to solve",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default=None,
        help="Neural network architecture (overrides config.yaml default for the PDE)",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Device to train on",
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--points", type=int, default=None, help="Number of collocation points"
    )
    return parser.parse_args()


def build_config(args):
    """Build Config object from config.yaml with CLI overrides."""
    with open(args.config, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    pde_key = args.pde
    pde_yaml = yaml_cfg.get("pde_configs", {}).get(pde_key, {})
    training_yaml = yaml_cfg.get("training", {})
    arch_type = args.arch or pde_yaml.get("architecture", "fourier")
    arch_yaml = yaml_cfg.get("architectures", {}).get(arch_type, {})

    # Device
    device_str = args.device or yaml_cfg.get("device", "cpu")
    if device_str == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    elif device_str == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Optimizer config
    optimizer_cfg = training_yaml.get("optimizer_config", {})
    learning_rate = args.lr or optimizer_cfg.get(
        "learning_rate", training_yaml.get("learning_rate", 0.001)
    )
    weight_decay = optimizer_cfg.get(
        "weight_decay", training_yaml.get("weight_decay", 0.0001)
    )

    # Loss weights — normalize pde -> residual
    raw_loss_weights = training_yaml.get("loss_weights", None)
    if raw_loss_weights is not None and "pde" in raw_loss_weights:
        raw_loss_weights = dict(raw_loss_weights)
        raw_loss_weights["residual"] = raw_loss_weights.pop("pde")

    num_epochs = args.epochs or training_yaml.get("num_epochs", 3000)
    num_points = args.points or training_yaml.get("num_collocation_points", 5000)

    early_stopping_cfg = training_yaml.get("early_stopping", {})
    reduce_lr_cfg = training_yaml.get("reduce_lr_params", {})

    training_config = TrainingConfig(
        num_epochs=num_epochs,
        batch_size=training_yaml.get("batch_size", 2048),
        num_collocation_points=num_points,
        num_boundary_points=training_yaml.get("num_boundary_points", 5000),
        num_initial_points=training_yaml.get("num_initial_points", 5000),
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_clipping=training_yaml.get("gradient_clipping", 1.0),
        early_stopping=EarlyStoppingConfig(
            enabled=early_stopping_cfg.get("enabled", True),
            patience=early_stopping_cfg.get("patience", 100),
            min_delta=early_stopping_cfg.get("min_delta", 1e-7),
        ),
        learning_rate_scheduler=LearningRateSchedulerConfig(
            type=training_yaml.get("scheduler_type", "cosine"),
            warmup_epochs=0,
            min_lr=reduce_lr_cfg.get("min_lr", 1e-6),
            factor=reduce_lr_cfg.get("factor", 0.5),
            patience=reduce_lr_cfg.get("patience", 50),
        ),
        adaptive_weights=AdaptiveWeightsConfig(
            enabled=training_yaml.get("adaptive_weights", {}).get("enabled", False),
        ),
        loss_weights=raw_loss_weights,
    )

    pde_config = PDEConfig(
        name=pde_yaml.get("name", pde_key),
        domain=pde_yaml.get("domain", [[0.0, 1.0]]),
        time_domain=pde_yaml.get("time_domain", [0.0, 1.0]),
        parameters=pde_yaml.get("parameters", {}),
        boundary_conditions=pde_yaml.get("boundary_conditions", {}),
        initial_condition=pde_yaml.get("initial_condition", {}),
        exact_solution=pde_yaml.get("exact_solution", {}),
        dimension=pde_yaml.get("dimension", 1),
        device=device,
        training=training_config,
    )

    input_dim = pde_yaml.get("input_dim", 2)
    output_dim = pde_yaml.get("output_dim", 1)

    model_config = ModelConfig(
        input_dim=input_dim,
        hidden_dim=arch_yaml.get("hidden_dim", 128),
        output_dim=output_dim,
        num_layers=arch_yaml.get("num_layers", 4),
        activation=arch_yaml.get("activation", "tanh"),
        dropout=arch_yaml.get("dropout", 0.0),
        layer_norm=arch_yaml.get("layer_norm", True),
        architecture=arch_type,
    )
    # Inject architecture-specific params
    for key in ("hidden_dims", "mapping_size", "scale", "omega_0", "num_heads", "num_blocks", "latent_dim"):
        if key in arch_yaml:
            setattr(model_config, key, arch_yaml[key])

    config = Config.__new__(Config)
    config.device = device
    config.model = model_config
    config.training = training_config
    config.pde_config = pde_config

    return config, pde_config, pde_yaml


def main():
    args = parse_args()
    print(f"Training PDE: {args.pde} | Architecture: {args.arch or 'from config'}")

    config, pde_config, pde_yaml = build_config(args)

    pde_class = PDE_CLASSES[args.pde]
    pde = pde_class(config=pde_config)

    model = PINNModel(config=config, device=config.device)
    print(f"Model: {config.model.architecture} | Device: {config.device}")
    print(f"Epochs: {config.training.num_epochs} | LR: {config.training.learning_rate}")

    optimizer_config = {
        "learning_rate": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }

    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config=optimizer_config,
        config=config,
        device=config.device,
        early_stopping_config=config.training.early_stopping,
    )

    trainer.train(num_epochs=config.training.num_epochs)
    print("Training complete.")


if __name__ == "__main__":
    main()
