import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import json

from src.pinn import PINNModel
from src.pdes.heat_equation import HeatEquation
from src.pdes.wave_equation import WaveEquation
from src.pdes.burgers_equation import BurgersEquation
from src.pdes.allen_cahn import AllenCahnEquation
from src.pdes.black_scholes import BlackScholesEquation
from src.pdes.cahn_hilliard import CahnHilliardEquation
from src.pdes.convection_equation import ConvectionEquation
from src.pdes.kdv_equation import KdVEquation
from src.trainer import PDETrainer
from src.rl_agent import RLAgent
from src.utils import setup_logging

def benchmark_pde(
    pde_class,
    pde_params: Dict,
    model_params: Dict,
    training_params: Dict,
    rl_params: Optional[Dict] = None,
    device: Optional[torch.device] = None
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
    pde = pde_class(**pde_params, device=device)
    
    # Initialize model
    model = PINNModel(**model_params).to(device)
    
    # Initialize RL agent if parameters provided
    rl_agent = None
    if rl_params:
        rl_agent = RLAgent(**rl_params, device=device)
    
    # Initialize trainer
    trainer = PDETrainer(
        model=model,
        pde=pde,
        optimizer_config=training_params['optimizer_config'],
        device=device
    )
    
    # Training loop
    start_time = torch.cuda.Event(enable_timing=True)
    end_time = torch.cuda.Event(enable_timing=True)
    
    start_time.record()
    
    for epoch in range(training_params['num_epochs']):
        # Generate collocation points
        if rl_agent and epoch % training_params['rl_update_frequency'] == 0:
            rl_agent.update_epsilon(epoch)
            x, t = pde.generate_adaptive_collocation_points(
                training_params['num_collocation_points'],
                rl_agent
            )
        else:
            x, t = pde.generate_collocation_points(
                training_params['num_collocation_points'],
                strategy='latin_hypercube'
            )
        
        # Training step
        train_loss = trainer.train_step(x, t, training_params['batch_size'])
        
        # Validation
        if (epoch + 1) % training_params['validation_frequency'] == 0:
            val_loss = trainer._compute_validation_loss()
            print(f"Epoch {epoch+1}/{training_params['num_epochs']}")
            print(f"Training Loss: {train_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")
            if rl_agent:
                print(f"RL Agent Epsilon: {rl_agent.epsilon:.6f}")
    
    end_time.record()
    torch.cuda.synchronize()
    
    # Final evaluation
    final_metrics = trainer.evaluate()
    
    return {
        'training_time': start_time.elapsed_time(end_time),
        'final_metrics': final_metrics,
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'rl_agent_parameters': sum(p.numel() for p in rl_agent.parameters()) if rl_agent else 0
    }

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        'input_dim': 2,
        'hidden_dim': 64,
        'output_dim': 1,
        'num_layers': 4,
        'activation': 'tanh',
        'fourier_features': True,
        'fourier_scale': 10.0,
        'dropout': 0.1,
        'layer_norm': True
    }
    
    training_params = {
        'num_epochs': 1000,
        'batch_size': 1000,
        'num_collocation_points': 10000,
        'validation_frequency': 100,
        'optimizer_config': {
            'name': 'adam',
            'lr': 0.001,
            'weight_decay': 1e-5
        },
        'rl_update_frequency': 10
    }
    
    rl_params = {
        'state_dim': 2,
        'action_dim': 1,
        'hidden_dim': 64,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'memory_size': 10000,
        'batch_size': 64,
        'target_update': 100,
        'reward_weights': {
            'residual': 1.0,
            'boundary': 1.0,
            'initial': 1.0,
            'exploration': 0.1
        }
    }
    
    # PDE configurations
    pde_configs = [
        {
            'name': 'Heat Equation',
            'class': HeatEquation,
            'params': {'alpha': 0.01, 'domain': (0, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'Wave Equation',
            'class': WaveEquation,
            'params': {'c': 1.0, 'domain': (0, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'Burgers Equation',
            'class': BurgersEquation,
            'params': {'nu': 0.01, 'domain': (-1, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'Allen-Cahn Equation',
            'class': AllenCahnEquation,
            'params': {'epsilon': 0.01, 'domain': (-1, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'Black-Scholes Equation',
            'class': BlackScholesEquation,
            'params': {'sigma': 0.2, 'r': 0.05, 'domain': (0, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'Cahn-Hilliard Equation',
            'class': CahnHilliardEquation,
            'params': {'epsilon': 0.01, 'domain': (-1, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'Convection Equation',
            'class': ConvectionEquation,
            'params': {'c': 1.0, 'domain': (0, 1), 'time_domain': (0, 1)}
        },
        {
            'name': 'KdV Equation',
            'class': KdVEquation,
            'params': {'epsilon': 0.01, 'domain': (-1, 1), 'time_domain': (0, 1)}
        }
    ]
    
    # Run benchmarks
    results = {}
    for config in pde_configs:
        logger.info(f"Benchmarking {config['name']}...")
        
        # Run with standard sampling
        standard_results = benchmark_pde(
            config['class'],
            config['params'],
            model_params,
            training_params,
            device=device
        )
        
        # Run with RL-based adaptive sampling
        rl_results = benchmark_pde(
            config['class'],
            config['params'],
            model_params,
            training_params,
            rl_params,
            device=device
        )
        
        results[config['name']] = {
            'standard': standard_results,
            'rl_adaptive': rl_results
        }
        
        logger.info(f"Completed {config['name']}")
    
    # Save results
    with open(results_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\nBenchmark Summary:")
    print("-" * 80)
    for pde_name, pde_results in results.items():
        print(f"\n{pde_name}:")
        print(f"Standard Sampling:")
        print(f"  Training Time: {pde_results['standard']['training_time']:.2f} ms")
        print(f"  Final L2 Error: {pde_results['standard']['final_metrics']['l2_error']:.6f}")
        print(f"RL Adaptive Sampling:")
        print(f"  Training Time: {pde_results['rl_adaptive']['training_time']:.2f} ms")
        print(f"  Final L2 Error: {pde_results['rl_adaptive']['final_metrics']['l2_error']:.6f}")
        print(f"  Model Parameters: {pde_results['rl_adaptive']['model_parameters']}")
        print(f"  RL Agent Parameters: {pde_results['rl_adaptive']['rl_agent_parameters']}")
    
    logger.info("Benchmarking completed successfully")

if __name__ == "__main__":
    main()
