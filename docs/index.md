# pinnrl

**Physics-informed neural networks for forward and inverse PDE problems, evaluated against numerical references.**

A physics-informed neural network (PINN) represents the solution `u(x, t)` of a PDE as a neural network. It is trained to make the PDE residual, the boundary conditions and the initial condition vanish at sampled *collocation points*, optionally together with measured data. No mesh is needed, and the PDE's parameters can be learned alongside the solution.

`pinnrl` packages that workflow:

- your own equations, written as text: `CustomPDE("u_t + u*u_x - nu*u_xx", ...)` for PDEs, ODEs, steady problems and systems ([Custom equations](custom_equations.md));
- ten shipped PDE configurations: heat (1-D and 2-D), wave, Burgers, convection, KdV, Allen–Cahn, Cahn–Hilliard, Black–Scholes and a damped pendulum. Each exact solution is checked against its PDE, initial condition and boundary conditions by the test suite.
- seven network architectures;
- adaptive collocation sampling;
- adaptive loss weighting;
- inverse problems (parameter identification);
- data-driven training on [The Well](https://github.com/PolymathicAI/the_well) datasets;
- a Dash dashboard for launching and inspecting runs.

```bash
pip install pinnrl
pinnrl-train --pde burgers --arch fourier --epochs 2000
pinnrl-dashboard
```

## When to use a PINN

Be clear about the trade-off. For a well-posed forward problem in one to three dimensions, classical solvers (finite differences, finite elements, spectral methods) are faster and orders of magnitude more accurate. For the 1-D periodic heat equation with $\alpha = 0.1$ and $u_0 = \sin(2\pi x)$, an explicit finite-difference scheme on 100 cells reaches a relative L2 error of about $6 \times 10^{-4}$ in roughly 30 ms of CPU time; a PINN needs minutes of training to reach about $10^{-2}$.

PINNs earn their place elsewhere:

| Situation | Why a PINN helps |
|---|---|
| **Inverse problems** | Unknown coefficients or fields are learned jointly with the solution from sparse, noisy measurements. There is no adjoint code to write. |
| **Physics plus data** | Measurements and conservation laws enter the same loss, so gaps in the data are filled by the physics. |
| **Awkward geometry or many dimensions** | It is mesh-free: points are sampled, not meshed. |
| **Differentiable surrogates** | The trained network is a smooth function of `(x, t)` that is cheap to evaluate and differentiate. |

The long-term goal of the project is the first row applied to industry: **digital twins** that estimate unmeasured states of real process equipment. See [Applications](applications.md) and the [Research program](research.md).

## Where to go next

- [Getting started](getting_started.md): install, your first run, and the Python API.
- [Concepts](concepts.md): how a PINN is trained, and what each technique in the package is for.
- [Equations](equations.md): the ten PDE configurations, where they arise, and why each is hard for a PINN.
- Guides: [Training](training.md), [Inverse problems](inverse_problems.md), [Datasets](datasets.md), [Dashboard](dashboard.md), [Benchmarks](benchmarks.md).

## Origins

`pinnrl` grew out of the master's thesis *On Solving Partial Differential Equations (PDEs) Using Physics-Informed Neural Networks (PINNs)* (J. García Ventura, Master in Mathematical Modeling, Research, Statistics, and Computation, University of the Basque Country UPV/EHU, 2025; advisors C. M. Cuesta Romero and F. de la Hoz Méndez). Read the [thesis](assets/thesis.pdf).

```bibtex
@software{garcia_pinnrl,
  author = {Garcia Ventura, Jose},
  title  = {pinnrl: physics-informed neural networks with adaptive collocation sampling},
  url    = {https://github.com/josegarciav/PINNs-RL-PDE}
}
```
