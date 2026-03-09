# pinnrl Documentation

**Solve PDEs with neural networks that learn where to look.**

`pinnrl` combines Physics-Informed Neural Networks (PINNs) with Deep Q-Network reinforcement learning to place collocation points adaptively — concentrating compute where the residual is highest, not where a grid happens to land.

## Quick navigation

| I want to... | Go to |
|---|---|
| Get running in 5 minutes | [Start Here](start_here.md) |
| Install and configure the library | [Setup](setup.md) |
| Understand how the architecture works | [Architecture Overview](architecture/overview.md) |
| See where the project is going | [Roadmap](roadmap.md) |
| Contribute a PDE or architecture | [Contributing](../CONTRIBUTING.md) |

## Supported PDEs and architectures

Nine PDEs, six neural architectures, one consistent API. See the [README](https://github.com/josegarciav/PINNs-RL-PDE#readme) for the full table.

## Install

```bash
uv add pinnrl
# or
pip install pinnrl
```
