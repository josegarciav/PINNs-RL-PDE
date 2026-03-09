# Contributing to pinnrl

Thank you for considering a contribution to **pinnrl** — a library for solving partial differential equations with Physics-Informed Neural Networks (PINNs) and reinforcement-learning-based adaptive sampling.

We welcome contributions from scientists, engineers, PhD students, and ML practitioners. You do not need to be a deep-learning expert to contribute. If you know the physics of a PDE, or have ideas about boundary conditions, exact solutions, or numerical benchmarks, your contribution is valuable.

---

## Table of Contents

1. [Welcome and scope](#1-welcome-and-scope)
2. [Development environment setup](#2-development-environment-setup)
3. [How to add a new PDE](#3-how-to-add-a-new-pde)
4. [How to add a new neural network architecture](#4-how-to-add-a-new-neural-network-architecture)
5. [Code style](#5-code-style)
6. [Pull request guidelines](#6-pull-request-guidelines)
7. [Governance: core contributors vs. community](#7-governance-core-contributors-vs-community)
8. [Issue labels](#8-issue-labels)
9. [Code of conduct](#9-code-of-conduct)

---

## 1. Welcome and scope

pinnrl exists because the scientific computing community deserves mesh-free, GPU-accelerated PDE solvers that are as easy to extend as they are to run. Every PDE added to the library, every architecture benchmarked, and every documentation improvement makes it more useful to the next researcher who opens a terminal.

**You can contribute if you:**

- Know the governing equations and boundary conditions for a PDE that is not yet implemented.
- Have implemented a neural network architecture you want to evaluate on PDE problems.
- Found a bug, a numerical instability, or a misleading error message.
- Want to improve documentation, tutorials, or example notebooks.
- Have a new sampling strategy or loss-weighting technique you want to test.

**You do not need to be an ML expert.** The architecture for adding a new PDE is a short Python class with four required methods. If you can write the physics, the framework handles the training loop, RL-based adaptive sampling, and visualization.

---

## 2. Development environment setup

### Fork and clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/<your-username>/PINNs-RL-PDE.git
cd PINNs-RL-PDE
git remote add upstream https://github.com/josegarciav/PINNs-RL-PDE.git
```

### Install dependencies

The project uses [uv](https://github.com/astral-sh/uv) for fast, reproducible dependency management. Install all extras (dev, viz, notebooks, docs) with:

```bash
uv sync --all-extras
```

If you prefer pip:

```bash
pip install -e ".[dev,viz,notebooks]"
```

### Install pre-commit hooks

Pre-commit runs `black` (formatting) and `ruff` (linting) automatically before every commit:

```bash
uv run pre-commit install
```

To run the hooks manually on all files:

```bash
uv run pre-commit run --all-files
```

### Running the test suite

```bash
uv run pytest tests/ -v
```

To run only a specific test module:

```bash
uv run pytest tests/unit_tests/test_pdes.py -v
```

To run with coverage:

```bash
uv run pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests are discovered automatically from the `tests/` directory. The suite covers PDE residual computation, neural network forward passes, RL agent behavior, and sampling strategies.

---

## 3. How to add a new PDE

Adding a new PDE requires changes in five places. The steps below walk through each one using a hypothetical `FisherEquation` (reaction-diffusion) as a running example.

### Step 1 — Create `src/pdes/my_equation.py`

All PDEs subclass `PDEBase` from `src/pdes/pde_base.py`. Create a new file:

```bash
touch src/pdes/fisher_equation.py
```

The required structure is:

```python
# Fisher equation: du/dt = D * d2u/dx2 + r * u * (1 - u)
# Application domain: population genetics, reaction-diffusion fronts
# Complexity: Nonlinear, 2nd-order

import torch
from .pde_base import PDEBase, PDEConfig
from typing import Dict, Any, Optional
from src.rl_agent import RLAgent


class FisherEquation(PDEBase):
    """
    Implementation of the Fisher-KPP equation:
        du/dt = D * d2u/dx2 + r * u * (1 - u)

    where D is the diffusion coefficient and r is the reaction rate.
    """

    def __init__(self, config: PDEConfig, **kwargs):
        super().__init__(config)

    def _validate_parameters(self):
        """Validate required parameters."""
        super()._validate_parameters()
        self.get_parameter("D", required=True)
        self.get_parameter("r", required=True)

    @property
    def D(self) -> float:
        """Diffusion coefficient."""
        return self.get_parameter("D", required=True)

    @property
    def r(self) -> float:
        """Reaction (growth) rate."""
        return self.get_parameter("r", required=True)

    def compute_residual(
        self,
        model: torch.nn.Module,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the PDE residual: u_t - D * u_xx - r * u * (1 - u) = 0

        Args:
            model: Neural network that approximates u(x, t)
            x: Spatial collocation points, shape (N, 1)
            t: Temporal collocation points, shape (N, 1)

        Returns:
            Residual tensor of shape (N, 1)
        """
        x = x.detach().requires_grad_(True)
        t = t.detach().requires_grad_(True)

        # Forward pass
        xt = torch.cat([x, t], dim=1)
        u = model(xt)

        # First-order time derivative
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        # Second-order spatial derivative
        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        # PDE residual
        residual = u_t - self.D * u_xx - self.r * u * (1 - u)
        return residual

    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Return the exact (or reference) solution if available.
        Return None if no closed-form solution exists for your parameter choice.

        For traveling-wave Fisher fronts a closed-form exists only for specific
        wavespeeds; return None and let the trainer skip exact-loss terms.

        Args:
            x: Spatial coordinates
            t: Temporal coordinates

        Returns:
            Exact solution tensor or None
        """
        return None  # Replace with analytic expression when available

    def initial_condition(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the initial condition u(x, 0).

        Args:
            x: Spatial coordinates at t=0

        Returns:
            Initial condition values
        """
        # Gaussian pulse as a typical Fisher IC
        return torch.exp(-10.0 * x**2)

    def boundary_condition(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        Evaluate the boundary condition (Dirichlet: u=0 at spatial boundaries).

        Args:
            x: Boundary spatial coordinates
            t: Boundary temporal coordinates

        Returns:
            Boundary condition values
        """
        return torch.zeros_like(x)
```

**Key rules:**

- `compute_residual()` must return a tensor that is zero when the PDE is satisfied. Use `torch.autograd.grad` with `create_graph=True` so gradients flow through training.
- `exact_solution()` may return `None` if no closed-form solution exists; the trainer handles this gracefully.
- `initial_condition()` and `boundary_condition()` are used to construct the boundary loss terms.
- Do not add dropout inside the PDE class — dropout hurts PDE residual convergence.

### Step 2 — Add to `config.yaml`

Open `config.yaml` and add an entry under `pde_configs`:

```yaml
pde_configs:
  # ... existing entries ...

  fisher:
    name: "Fisher-KPP Equation"
    architecture: "fourier"          # recommended starting architecture
    input_dim: 2                     # (x, t)
    output_dim: 1                    # u(x, t)
    parameters:
      D: 0.01                        # diffusion coefficient
      r: 1.0                         # reaction rate
    domain: [[0, 1]]
    time_domain: [0, 5]
    dimension: 1
    initial_condition:
      type: "gaussian"
      amplitude: 1.0
      width: 0.1
    boundary_conditions:
      dirichlet: { value: 0.0 }
    exact_solution:
      type: "none"
```

Also add `"fisher"` to the `pde_type` options comment at the top of the file.

### Step 3 — Register in `src/interactive_trainer.py`

Add an import and register the class in the `create_pde()` method:

```python
# At the top of the file, with the other imports:
from src.pdes.fisher_equation import FisherEquation

# Inside create_pde(), in the pde_class_map dictionary:
pde_class_map = {
    "heat": HeatEquation,
    "wave": WaveEquation,
    # ... existing entries ...
    "fisher": FisherEquation,   # <-- add this line
}
```

### Step 4 — Register in `src/interactive_trainer.py`

Add the import and case in `create_pde()`:

```python
from src.pdes.fisher_equation import FisherEquation

# Inside create_pde(), in the pde_class_map dictionary:
pde_class_map = {
    "heat": HeatEquation,
    "wave": WaveEquation,
    # ... existing entries ...
    "fisher": FisherEquation,   # <-- add this line
}
```

### Step 5 — Write a test in `tests/unit_tests/test_pdes.py`

Add a test class following the existing pattern:

```python
class TestFisherEquation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        config = PDEConfig(
            name="Fisher-KPP Equation",
            domain=[(0.0, 1.0)],
            time_domain=(0.0, 5.0),
            parameters={"D": 0.01, "r": 1.0},
            boundary_conditions={"dirichlet": {"type": "fixed", "value": 0.0}},
            initial_condition={"type": "gaussian", "amplitude": 1.0, "width": 0.1},
            exact_solution={"type": "none"},
            dimension=1,
            device=self.device,
        )
        self.pde = FisherEquation(config=config)
        self.model = FeedForwardNetwork(
            {
                "input_dim": 2,
                "hidden_dims": [32, 32],
                "output_dim": 1,
                "activation": "tanh",
                "device": self.device,
            }
        )

    def test_compute_residual_shape(self):
        """Residual must have the same shape as the input batch."""
        N = 50
        x = torch.rand(N, 1, requires_grad=True)
        t = torch.rand(N, 1, requires_grad=True)
        residual = self.pde.compute_residual(self.model, x, t)
        self.assertEqual(residual.shape, (N, 1))

    def test_initial_condition(self):
        """Initial condition must return finite values."""
        x = torch.linspace(0, 1, 100).unsqueeze(1)
        ic = self.pde.initial_condition(x)
        self.assertTrue(torch.all(torch.isfinite(ic)))

    def test_boundary_condition(self):
        """Boundary condition must equal zero at the boundaries."""
        x_boundary = torch.tensor([[0.0], [1.0]])
        t_boundary = torch.rand(2, 1)
        bc = self.pde.boundary_condition(x_boundary, t_boundary)
        self.assertTrue(torch.allclose(bc, torch.zeros(2, 1)))
```

After all steps are complete, verify end-to-end with:

```bash
uv run pytest tests/unit_tests/test_pdes.py::TestFisherEquation -v
uv run python src/interactive_trainer.py  # Select "Fisher" PDE and "Fourier" architecture
```

---

## 4. How to add a new neural network architecture

### Step 1 — Create `src/neural_networks/my_arch.py`

All architectures subclass `BaseNetwork` from `src/neural_networks/base_network.py`. The `BaseNetwork` class inherits from `torch.nn.Module` and provides `save_state()`, `load_state()`, `count_parameters()`, and `_prepare_input()`. You only need to implement `__init__` and `forward`.

```python
"""Kolmogorov-Arnold Network (KAN) approximation for PINNs."""

import torch
import torch.nn as nn
from .base_network import BaseNetwork
from typing import Dict, List


class KANNetwork(BaseNetwork):
    """
    Simplified KAN-inspired network using learnable B-spline basis functions.

    Config keys:
        input_dim  (int)         - number of input features
        output_dim (int)         - number of output features
        hidden_dims (List[int])  - widths of each hidden layer
        grid_size (int)          - number of B-spline grid points (default 5)
        device                   - torch device
    """

    def __init__(self, config: Dict) -> None:
        super().__init__(config)
        input_dim = config["input_dim"]
        output_dim = config["output_dim"]
        hidden_dims: List[int] = config.get("hidden_dims", [64, 64])

        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for in_d, out_d in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.Tanh())
        # Remove final activation
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x):
        x = self._prepare_input(x)
        return self.net(x)
```

Docstring conventions:
- List every config key the architecture reads.
- Note any hyperparameter choices that differ from defaults.
- Keep the file under 300 lines; split into a separate file for helper modules (e.g., `kan_layers.py`).

### Step 2 — Add to `src/neural_networks/__init__.py`

```python
from .kan_network import KANNetwork

# Add to __all__:
__all__ = [
    ...,
    "KANNetwork",
]
```

### Step 3 — Add to `PINNModel` factory in `src/neural_networks/__init__.py`

Inside the `PINNModel.__init__` method, add a branch:

```python
elif self.architecture == "kan":
    self.model = KANNetwork(config_with_device)
```

### Step 4 — Add to `config.yaml`

```yaml
architectures:
  # ... existing entries ...
  kan:
    hidden_dims: [64, 64, 64]
    grid_size: 5
    activation: "tanh"
    dropout: 0.0
```

After registering, run the architecture test suite to confirm the forward pass works:

```bash
uv run pytest tests/unit_tests/test_neural_networks.py -v
```

---

## 5. Code style

pinnrl uses **black** for formatting and **ruff** for linting. Both are enforced via pre-commit and CI.

| Tool | Role | Config location |
|------|------|-----------------|
| black | Opinionated auto-formatter | `[tool.black]` in `pyproject.toml` |
| ruff | Fast linter (replaces flake8 + isort) | `[tool.ruff]` in `pyproject.toml` |
| mypy | Optional static type checking | `[tool.mypy]` in `pyproject.toml` |

**Line length:** 100 characters (matches both `black` and `ruff` configs).

**Formatting rules that matter for this codebase:**

- Use `torch.autograd.grad(..., create_graph=True)` — never `backward()` — inside PDE residuals; backward breaks higher-order derivative computation.
- Type-annotate public method signatures. Use `Optional[torch.Tensor]` when a return value can be `None` (e.g., `exact_solution`).
- Do not use `dropout > 0` inside PDE classes. Dropout randomizes residual evaluation and hurts convergence.
- Docstrings follow Google style with `Args:` and `Returns:` sections.
- No emojis in source code or docstrings.
- Variable names for spatial coordinates: `x`, `y`, `z`. Time: `t`. Solution: `u`. Residual: `residual` or `r`.

To format and lint before committing:

```bash
uv run black src/ tests/ scripts/
uv run ruff check src/ tests/ scripts/ --fix
```

---

## 6. Pull request guidelines

### Before opening a PR

- Run the full test suite: `uv run pytest tests/ -v`
- Run pre-commit on all files: `uv run pre-commit run --all-files`
- Make sure your branch is up to date with `main`: `git fetch upstream && git rebase upstream/main`

### What a good PR looks like

A well-scoped PR does one thing — adds a PDE, fixes a bug, or improves a specific aspect of documentation — and is small enough to review in under 30 minutes.

**PR description checklist:**

- [ ] What problem does this PR solve or what feature does it add?
- [ ] Which files were changed and why?
- [ ] What PDE(s) or architecture(s) are affected?
- [ ] Were new tests added? Do existing tests still pass?
- [ ] If adding a PDE: is there a reference (paper, textbook) for the governing equations and exact solution?

### Test coverage requirements

- New PDE classes must include tests for `compute_residual` shape, `initial_condition` finiteness, and `boundary_condition` correctness.
- New architecture classes must include a forward-pass test covering batch sizes of 1 and 64.
- Bug fixes must include a regression test that fails before the fix and passes after.
- Coverage for `src/` should not decrease. Check with `uv run pytest --cov=src --cov-report=term-missing`.

### Commit message convention

Use the imperative mood and a short summary line under 72 characters:

```
feat: add Fisher-KPP reaction-diffusion PDE
fix: correct wave equation boundary condition for periodic domain
docs: add architecture diagram to README
test: add residual shape test for Cahn-Hilliard equation
```

---

## 7. Governance: core contributors vs. community

pinnrl follows a **BDFL-lite** model: the project maintainer ([@josegarciav](https://github.com/josegarciav)) makes final decisions on architecture and release scope, but substantive design decisions are discussed openly in GitHub issues before implementation begins.

**Core contributors** have write access to the repository and are responsible for reviewing PRs, triaging issues, and making releases. Core contributor status is earned through sustained, high-quality contributions.

**Community contributors** submit PRs from forks. All contributions are welcome regardless of seniority or affiliation.

**Decision-making process:**

1. Open an issue describing the proposed change before implementing it for anything non-trivial (new PDE families, major architecture changes, API changes).
2. Allow at least 48 hours for feedback from maintainers and community.
3. Once consensus is reached in the issue, implement the change and open a PR.
4. At least one core contributor review is required before merging.

**Becoming a core contributor:** If you have merged three or more substantial PRs, open an issue requesting core contributor status. The maintainer will add you to `CONTRIBUTORS.md` and grant repository write access.

---

## 8. Issue labels

Use these labels when opening or triaging issues:

| Label | Meaning |
|-------|---------|
| `bug` | Something produces incorrect output or raises an unexpected error |
| `enhancement` | Improvement to existing functionality (performance, usability, API) |
| `PDE` | Request or PR adding a new partial differential equation |
| `architecture` | Request or PR adding a new neural network architecture |
| `docs` | Documentation improvement (README, docstrings, tutorials, notebooks) |
| `good-first-issue` | Well-scoped, self-contained task suitable for first-time contributors |
| `rl-sampling` | Related to the reinforcement-learning adaptive collocation component |
| `benchmarks` | Numerical accuracy comparisons or performance profiling |
| `question` | Usage question — may be redirected to Discussions |

When opening a new issue, apply the most specific label that fits. Core contributors will add additional labels during triage.

---

## 9. Code of conduct

pinnrl adopts the [Contributor Covenant Code of Conduct, version 2.1](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating in this project you agree to abide by its terms.

In short: be respectful, assume good faith, and remember that behind every GitHub handle is a person trying to do good science.

---

*Thank you for making pinnrl better.*
