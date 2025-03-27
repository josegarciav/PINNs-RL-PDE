import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import time
from typing import Dict, List, Tuple, Optional
import json

from src.neural_networks.neural_networks import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.convection_equation import ConvectionEquation
from src.pdes.kdv_equation import KdVEquation
from src.pdes.pendulum_equation import PendulumEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import (
    setup_logging,
    save_model,
    plot_architecture_comparison,
    create_interactive_report,
)


def benchmark_pde(
    pde_class,
    pde_params: Dict,
    model_params: Dict,
    training_params: Dict,
    rl_params: Optional[Dict] = None,
    device: Optional[torch.device] = None,
) -> Dict:
    """
    Benchmark a PDE implementation.

    Args:
        pde_class: PDE class to benchmark
        pde_params: Parameters for PDE initialization
        model_params: Parameters for model initialization
        training_params: Parameters for training
        rl_params: Optional parameters for RL agent
        device: Computing device

    Returns:
        Dictionary of benchmark results
    """
    # Initialize PDE
    pde = pde_class(**pde_params)  # Device is already in pde_params

    # Initialize model with different architectures
    architectures = ["fourier", "resnet", "siren", "autoencoder", "feedforward"]
    results = {}
    metrics = {
        'training_loss': {},
        'validation_loss': {},
        'computation_time': {}
    }

    for arch in architectures:
        model_params["architecture"] = arch
        model = PINNModel(**model_params).to(device)

        # Initialize RL agent if parameters provided
        rl_agent = None
        if rl_params:
            rl_agent = RLAgent(**rl_params)

        # Initialize trainer
        trainer = PDETrainer(
            model=model,
            pde=pde,
            optimizer_config=training_params["optimizer_config"],
            device=device,
        )

        # Training loop
        if device.type == "cuda":
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
            torch.cuda.reset_peak_memory_stats()
        else:
            start_time = time.time()

        # Train the model
        history = trainer.train(
            num_epochs=training_params["num_epochs"],
            batch_size=training_params["batch_size"],
            num_points=training_params["num_collocation_points"],
            validation_frequency=training_params["validation_frequency"],
        )

        if device.type == "cuda":
            end_time.record()
            torch.cuda.synchronize()
            training_time = start_time.elapsed_time(end_time)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
        else:
            training_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            peak_memory = 0

        # Final evaluation
        final_metrics = trainer.evaluate()

        # Compute model size
        model_size = (
            sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        )  # MB
        rl_size = (
            sum(p.numel() * p.element_size() for p in rl_agent.parameters()) / 1024**2
            if rl_agent
            else 0
        )

        # Update metrics for interactive report
        metrics['training_loss'][f"{pde.__class__.__name__}_{arch}"] = history['train_loss']
        metrics['validation_loss'][f"{pde.__class__.__name__}_{arch}"] = history['val_loss']
        metrics['computation_time'][f"{pde.__class__.__name__}_{arch}"] = training_time

        results[arch] = {
            "training_time": training_time,
            "peak_memory": peak_memory,
            "model_size": model_size,
            "rl_size": rl_size,
            "final_metrics": final_metrics,
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "rl_agent_parameters": (
                sum(p.numel() for p in rl_agent.parameters()) if rl_agent else 0
            ),
            "training_history": history,
        }

    return results, metrics


def main():
    # Setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/benchmark_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    logger = setup_logging(results_dir / "benchmark.log")
    logger.info("Starting PDE benchmarking")

    # Common parameters
    model_params = {
        "input_dim": 2,
        "hidden_dim": 128,  # Increased for better performance
        "output_dim": 1,
        "num_layers": 6,  # Increased for better expressiveness
        "activation": "tanh",
        "fourier_features": True,
        "fourier_scale": 10.0,
        "dropout": 0.1,
        "layer_norm": True,
        "device": device,  # Pass device to model
    }

    training_params = {
        "num_epochs": 1000,
        "batch_size": 1000,
        "num_collocation_points": 10000,
        "validation_frequency": 100,
        "optimizer_config": {
            "name": "adam",
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "patience": 10,
        },
        "rl_update_frequency": 10,
    }

    rl_params = {
        "state_dim": 2,
        "action_dim": 1,
        "hidden_dim": 128,  # Increased to match model
        "learning_rate": 0.001,
        "gamma": 0.99,
        "epsilon_start": 1.0,
        "epsilon_end": 0.01,
        "epsilon_decay": 0.995,
        "memory_size": 10000,
        "batch_size": 64,
        "target_update": 100,
        "reward_weights": {
            "residual": 1.0,
            "boundary": 1.0,
            "initial": 1.0,
            "exploration": 0.1,
        },
        "device": device,  # Pass device to RL agent
    }

    # PDE configurations
    pde_configs = [
        {
            "name": "Heat Equation",
            "class": HeatEquation,
            "params": {
                "alpha": 0.01,
                "domain": [(0.0, 1.0)],  # List of tuples for spatial dimensions
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,  # Explicitly set dimension
                "device": device,  # Pass device to PDE
            },
        },
        {
            "name": "Wave Equation",
            "class": WaveEquation,
            "params": {
                "c": 1.0,
                "domain": [(0.0, 1.0)],  # List of tuples for spatial dimensions
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,  # Explicitly set dimension
                "device": device,  # Pass device to PDE
            },
        },
        {
            "name": "Burgers Equation",
            "class": BurgersEquation,
            "params": {
                "nu": 0.01,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        },
        {
            "name": "Allen-Cahn Equation",
            "class": AllenCahnEquation,
            "params": {
                "epsilon": 0.01,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        },
        {
            "name": "Black-Scholes Equation",
            "class": BlackScholesEquation,
            "params": {
                "sigma": 0.2,
                "r": 0.05,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        },
        {
            "name": "Cahn-Hilliard Equation",
            "class": CahnHilliardEquation,
            "params": {
                "epsilon": 0.01,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        },
        {
            "name": "Convection Equation",
            "class": ConvectionEquation,
            "params": {
                "c": 1.0,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        },
        {
            "name": "KdV Equation",
            "class": KdVEquation,
            "params": {
                "epsilon": 0.01,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        },
        {
            "name": "Pendulum Equation",
            "class": PendulumEquation,
            "params": {
                "g": 9.81,
                "L": 1.0,
                "domain": [(0.0, 1.0)],
                "time_domain": (0.0, 1.0),
                "boundary_conditions": {"dirichlet": {"type": "fixed", "value": 0.0}},
                "initial_condition": {
                    "type": "sine",
                    "amplitude": 1.0,
                    "frequency": 2.0,
                },
                "exact_solution": {"amplitude": 1.0, "frequency": 2.0},
                "dimension": 1,
                "device": device,
            },
        }
    ]

    # Run benchmarks for each PDE
    all_results = {}
    all_metrics = {
        'training_loss': {},
        'validation_loss': {},
        'computation_time': {}
    }

    for pde_config in pde_configs:
        logger.info(f"\nBenchmarking {pde_config['name']}...")
        try:
            results, metrics = benchmark_pde(
                pde_config["class"],
                pde_config["params"],
                model_params,
                training_params,
                rl_params,
                device,
            )
            all_results[pde_config["name"]] = results
            
            # Merge metrics
            for key in all_metrics:
                all_metrics[key].update(metrics[key])
                
        except Exception as e:
            logger.error(f"Error benchmarking {pde_config['name']}: {str(e)}")
            continue

    # Save benchmark results
    results_path = results_dir / "benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=4)

    # Create interactive report
    report_path = results_dir / "interactive_report.html"
    create_interactive_report(
        experiment_dir=str(results_dir),
        pdes=[pde_config["class"](**pde_config["params"]) for pde_config in pde_configs],
        architectures=[{'name': arch, 'model': None} for arch in results.keys()],
        metrics=all_metrics,
        config={
            'model': model_params,
            'training': training_params,
            'rl': rl_params
        },
        save_path=str(report_path)
    )

    logger.info("Benchmarking completed successfully")
    logger.info(f"Results saved in: {results_dir}")


if __name__ == "__main__":
    main()
