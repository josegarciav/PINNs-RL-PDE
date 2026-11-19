# pinnrl

`pinnrl` is a Python package for physics-informed neural networks (PINNs) on forward and inverse problems, with a training step compiled per equation. It measures how far an estimate can be trusted and ships the classical method for each problem, so every result has a baseline.

$$
u_\theta(x, t) \approx u(x, t), \qquad
\mathcal{L}(\theta) = \mathcal{L}_{\text{residual}} + \mathcal{L}_{\text{boundary}} + \mathcal{L}_{\text{initial}} + \mathcal{L}_{\text{data}}
$$

- **PINN**: a network $u_\theta$ trained to satisfy a differential equation. Unknown parameters of the equation can be trained with its weights $\theta$.
- **Residual**: what is left when $u_\theta$ is put into the equation. It is zero for an exact solution.
- **Collocation point**: a sampled point where the residual is evaluated. No mesh is needed.
- **$\mathcal{L}_{\text{data}}$**: the misfit to measurements. It is optional.

## Who it is for

- **Researchers** studying PINN training, sampling or identifiability, who need a fast trainer, reference solutions and multi-seed protocols.
- **Process and control engineers** who want to know what their records can determine before buying a sensor or running a campaign.
- **Teachers and students**: the test suite checks every exact solution against its equation and conditions.

## When to use a PINN

The package's own experiments give a narrow answer. Every number and its source is on [When to use a PINN, and when not to](when_to_use.md).

**Do not use a PINN when**

- **The forward problem is well posed and a classical solver exists.** On the 1-D heat equation, finite differences reach a relative L2 error of about $7 \times 10^{-4}$ in about 30 ms ([classical baselines](benchmarks.md#classical-baselines)). A PINN with L-BFGS reaches $2 \times 10^{-4}$ in 120 s ([D01](benchmarks.md#optimizers-and-the-floor-experiments-d01-d02)).
- **The model has a cheap, trusted forward solve.** On the evaporator, a Kalman filter tracks the fouling to 3.5 % in 0.2 s. The PINN twin reaches 9–16 % in five minutes and fails to train on 22 of 36 runs (J03, J05, J06).
- **You need an interval you can trust and have not checked it.** The twin's own 90 % interval covers the truth 25 % of the time ([J06](benchmarks.md#calibrated-fouling-intervals-experiment-j06)).

**Consider a PINN when**

- **There is no trusted forward solver**, or it is too expensive to iterate.
- **An unknown coefficient or function sits inside the equation.** It is learned with the solution from sparse, noisy measurements, with no adjoint code to write.
- **One network should replace many solves**, or the problem has awkward geometry or many dimensions. A parametric surrogate covers a parameter range to 0.2–2 % ([K45](benchmarks.md#parametric-surrogate-as-the-forward-emulator-experiment-k45)).

In every case, run the baseline. The package ships it.

The long-term goal is **digital twins**: models of one piece of process equipment, kept in step with its sensor records, that estimate its unmeasured states. Read [Applications](applications.md) and the [Research program](research.md).

## What is in the package

| Part | What it gives |
|---|---|
| Your own equations | Written as text, `CustomPDE("u_t + u*u_x - nu*u_xx", ...)`: PDEs, ODEs, steady problems and systems ([Custom equations](custom_equations.md)) |
| Shipped equations | Ten configurations: heat (1-D and 2-D), wave, Burgers, convection, KdV, Allen–Cahn, Cahn–Hilliard, Black–Scholes and a damped pendulum |
| Training | Seven architectures, adaptive collocation sampling, adaptive loss weighting |
| Inverse problems | Parameter identification |
| Data | Training on [The Well](https://github.com/PolymathicAI/the_well) datasets |
| Engine | A compiled training step, on by default ([The native engine](engine.md)) |
| Dashboard | A Dash app for launching and inspecting runs |

## Where to go next

| I want to ... | Page |
|---|---|
| Install and run a first problem | [Getting started](getting_started.md) |
| Decide whether a PINN fits my problem | [When to use a PINN](when_to_use.md) |
| Understand how a PINN is trained | [Concepts](concepts.md) |
| See the shipped equations | [Equations](equations.md) |
| Write my own equation | [Custom equations](custom_equations.md) |
| Choose training settings | [Training](training.md) |
| Identify a parameter from measurements | [Inverse problems](inverse_problems.md) |
| Know what the data can determine | [Theory](theory/index.md) |
| Estimate unmeasured states of process equipment | [Process twins](process_twins.md) |
| Train on simulation datasets | [Datasets](datasets.md) |
| Launch and inspect runs in a browser | [Dashboard](dashboard.md) |
| See the measured results and baselines | [Benchmarks](benchmarks.md) |
| Know how the compiled step works | [The native engine](engine.md) |
| Fix a problem | [Troubleshooting](troubleshooting.md) |
| Look up a flag, a config key or a class | [CLI](reference/cli.md), [Config](reference/config.md), [API](api/index.md) |
| See what is planned and what was released | [Roadmap](roadmap.md), [History](history.md) |

## Install and first run

```bash
pip install pinnrl
pinnrl-train --pde burgers --arch fourier --epochs 2000
pinnrl-dashboard
```

The three lines install the package (Python 3.10–3.13), train a PINN on the Burgers equation into `./runs/`, and open a browser app to inspect the run. See [Getting started](getting_started.md).

## Origins

`pinnrl` grew out of the master's [thesis](assets/thesis.pdf) *On Solving Partial Differential Equations (PDEs) Using Physics-Informed Neural Networks (PINNs)* by J. García Ventura (Master in Mathematical Modeling, Research, Statistics, and Computation, University of the Basque Country UPV/EHU, 2025; advisors C. M. Cuesta Romero and F. de la Hoz Méndez).

```bibtex
@software{garcia_pinnrl,
  author = {Garcia Ventura, Jose},
  title  = {pinnrl: physics-informed neural networks with adaptive collocation sampling},
  url    = {https://github.com/josegarciav/PINNs-RL-PDE}
}
```
