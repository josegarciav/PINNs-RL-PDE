# Contributing to pinnrl

Contributions are welcome: bug reports, new PDEs, new architectures, benchmarks, docs. For anything large, open an issue first so we can agree on the approach.

## Setup

```bash
git clone https://github.com/josegarciav/PINNs-RL-PDE.git
cd PINNs-RL-PDE
uv sync --all-extras            # creates .venv with every extra and the dev tools
uv run pre-commit install       # ruff (lint + format), and output stripping for experiments/ notebooks
```

## Checks (the same ones CI runs)

```bash
uv run pytest                                   # the full suite, about 30 s on a laptop
uv run pytest --cov=pinnrl --cov-fail-under=80  # coverage gate
uv run ruff format --check . && uv run ruff check .
uv run mkdocs build --strict                    # docs, including the API reference
scripts/verify_release.sh 3.12                  # clean wheel: entry points, doc snippets, every notebook
```

## Rules that keep results trustworthy

1. **Physics changes keep `tests/unit_tests/test_physics_consistency.py` green.** It checks that every exact solution satisfies its PDE, initial condition and boundary conditions; that manufactured solutions give zero residual; and that every configured parameter actually enters the residual. If you add a PDE or a boundary condition, add its case there.
2. **A bug fix comes with a test that fails without it.**
3. **One entry point per action.** Training goes through `PDETrainer` (`train` or `train_steps`), PDE lookup through `pinnrl.pdes.registry`, and the CLIs are `pinnrl-train`, `pinnrl-dashboard` and `pinnrl-benchmark`. Don't add a second training loop or a second registry.
4. **Claims need an experiment.** Statements like "X improves accuracy" belong in `experiments/<ID>_<slug>/` with the backlog protocol: paired seeds, equal budgets, overhead counted, median and IQR, and a Wilcoxon test. See `experiments/README.md`.

## Adding a PDE

1. Create `pinnrl/pdes/my_equation.py` with a `PDEBase` subclass. Implement `compute_residual(model, x, t)` using `self.compute_derivatives(...)`, and `_exact_solution(x, t)` if a closed form exists. Read parameters with `self.get_parameter(name)` so inverse mode can make them trainable.
2. Register it with one line in `pinnrl/pdes/registry.py`: `PDESpec("my_eq", "My Equation", MyEquation)`. The CLI and the dashboard pick it up from there.
3. Add a `pde_configs.my_eq` block to `pinnrl/config/config.yaml` with domain, parameters, boundary conditions, initial condition and exact solution.
4. Add its consistency case to `test_physics_consistency.py` and a short training smoke test.
5. Document it in a row of `docs/equations.md`: the PDE, where it arises, and what makes it hard.

## Adding an architecture

1. Create `pinnrl/neural_networks/my_arch.py` (subclass `BaseNetwork`).
2. Add a branch to the `PINNModel` factory in `pinnrl/neural_networks/__init__.py` and an entry under `architectures:` in `config.yaml`.
3. Add it to the PDE × architecture smoke matrix in `tests/unit_tests/test_pde_arch_matrix.py`.

## Style

- Ruff handles formatting (line length 100) and linting, including Google-style docstrings for public functions.
- Type hints on public APIs. Keep comment density modest: say why, not what.
- Research notebooks in `experiments/` are committed without outputs (pre-commit strips them). Tutorial notebooks in `notebooks/` keep theirs.

## Pull requests

- Use conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `perf:`, `style:` or `chore:`.
- Keep each PR to one topic, with a description of what changed, why, and how it was tested.
- User-visible changes get a line in `CHANGELOG.md` under *Unreleased*.

## Releases (maintainers)

Bump `version` in `pyproject.toml`, date the CHANGELOG section, and run `scripts/verify_release.sh`. Then publish a GitHub Release tagged `vX.Y.Z`. The publish workflow refuses to run if the tag doesn't match the version, re-runs the clean-wheel checks, and uploads to PyPI via trusted publishing.

## Conduct

Be respectful and constructive. Report unacceptable behaviour to the maintainer via GitHub.
