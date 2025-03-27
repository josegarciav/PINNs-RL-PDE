# PINNs-RL-PDE

A Python package for solving Partial Differential Equations (PDEs) using Physics-Informed Neural Networks (PINNs) and Reinforcement Learning (RL).

## Overview

This package combines the power of Physics-Informed Neural Networks with Reinforcement Learning to solve complex PDEs. It provides a flexible framework for:

- Implementing custom PDEs
- Training PINNs with physics-informed loss functions
- Incorporating reinforcement learning for optimal control
- Visualizing solutions and results

## Installation

```bash
pip install pinns-rl-pde
```

## Quick Start

```python
from pinns_rl_pde.models import PINN
from pinns_rl_pde.pdes import HeatEquation

# Define your PDE
pde = HeatEquation()

# Create and train PINN
model = PINN(pde)
model.train()
```

## Features

- Modular architecture for easy extension
- Support for various PDE types
- Built-in visualization tools
- Integration with PyTorch
- Reinforcement learning capabilities

## Documentation

For detailed documentation, please visit [documentation link].

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Project Description  
This project explores the integration of **Reinforcement Learning (RL) and Physics-Informed Neural Networks (PINNs)** to improve the selection of collocation points in solving **Partial Differential Equations (PDEs)**.  
By dynamically adapting collocation points, we aim to enhance convergence speed, reduce error rates, and improve generalization across different PDEs.

## Installation  
Clone the repository and install dependencies:  
```bash
git clone https://github.com/josegarciav/PINNs-RL-PDE.git
cd PINNs-RL-PDE
pip install -r requirements.txt
