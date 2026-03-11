# pinnrl Documentation

**Solve PDEs with neural networks that learn where to look.**

`pinnrl` combines Physics-Informed Neural Networks (PINNs) with Deep Q-Network reinforcement learning to place collocation points adaptively — concentrating compute where the residual is highest, not where a grid happens to land.

## Quick navigation

| I want to... | Go to |
|---|---|
| Get running in 5 minutes | [Start Here](start_here.md) |
| Install and configure the library | [Setup](setup.md) |
| Launch and use the web dashboard | [Dashboard Guide](dashboard.md) |
| Understand how the architecture works | [Architecture Overview](ARCHITECTURE.md) |
| See where the project is going | [Roadmap](roadmap.md) |
| Understand sampling strategies | [Sampling Strategies](sampling_strategies.md) |
| Contribute a PDE or architecture | [Contributing](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/CONTRIBUTING.md) |

## Supported PDEs and architectures

Nine PDEs, seven neural architectures, one consistent API. See the [README](https://github.com/josegarciav/PINNs-RL-PDE#readme) for the full table.

## Install

```bash
uv add pinnrl
# or
pip install pinnrl
```
