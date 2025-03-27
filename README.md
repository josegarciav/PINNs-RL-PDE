# PINNs-RL-PDE

A final thesis project on solving Partial Differential Equations (PDEs) using Physics-Informed Neural Networks (PINNs) and Reinforcement Learning (RL).

## Overview

This project combines the power of Physics-Informed Neural Networks with Reinforcement Learning to solve complex PDEs. It implements a research framework for:

- Solving various types of PDEs with physics-informed loss functions
- Using reinforcement learning for adaptive selection of collocation points
- Comparing different neural network architectures for solving PDEs
- Visualizing and analyzing results through interactive reports

## Project Description  
This final thesis project explores the integration of **Reinforcement Learning (RL) and Physics-Informed Neural Networks (PINNs)** to improve the selection of collocation points in solving **Partial Differential Equations (PDEs)**.  
By dynamically adapting collocation points, we aim to enhance convergence speed, reduce error rates, and improve generalization across different PDEs.

## Setup
Clone the repository and install dependencies:  
```bash
git clone https://github.com/josegarciav/PINNs-RL-PDE.git
cd PINNs-RL-PDE
pip install -r requirements.txt
```

## Running Experiments

All experiment parameters are defined in the `config.yaml` file. To run different experiments, modify this file according to your needs.

```bash
# Run training with the heat equation
python experiments/train_heat_equation.py

# Run benchmark across all architectures
python experiments/run_all_architectures.py 

# Run benchmark across different PDEs
python experiments/benchmark_pdes.py
```

## Project Structure

- `src/` - Core implementation of the framework
  - `neural_networks/` - Neural network architectures
  - `pdes/` - PDE implementations
  - `utils/` - Utility functions
- `experiments/` - Experiment scripts
- `tests/` - Test cases
- `results/` - Output directory for experiment results
- `config.yaml` - Configuration file for all experiments
