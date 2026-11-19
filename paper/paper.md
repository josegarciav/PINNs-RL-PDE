---
title: "pinnrl: physics-informed neural networks for forward and inverse problems, with a compiled training step, an inverse-modelling toolbox and classical baselines"
author:
  - name: Jose Garcia Ventura
date: 19 November 2026
bibliography: paper.bib
---

# Abstract

`pinnrl` is an open-source Python package for physics-informed neural networks (PINNs). It has three parts. The first is a trainer for equations written as text, whose training step is compiled per equation: on nine benchmark problems from second to fourth order it is the fastest step we measured on an Apple M4 (1.1–1.8 times JAX's best differentiation mode, 2.4–7.7 times an eager PyTorch loop, 2.7–9.9 times DeepXDE), while on a Linux x86 machine JAX's forward mode is a few percent ahead at low order. The second is a toolbox for inverse problems that reports whether an estimate can be trusted: Fisher information and Cramér–Rao bounds from a trained network, stability constants, calibrated intervals, equivalence regions and a misfit test against a reference solver. The third is a set of process-engineering units (a falling-film evaporator, a drying droplet, co-current and counter-current spray dryers, and a line that chains them) that ship with their classical estimators, so every PINN result is graded against the Kalman filter, moving-horizon estimation or least squares through a reference solver. The package reports its negative results with the same weight as its positive ones: on every unit with a cheap forward solve the classical estimator beat the PINN twin, and the reinforcement-learning collocation sampler does not beat uniform sampling at equal cost. This paper states the goals, describes the design, and collects the benchmarks, each reproducible from a script in the repository.

# 1. Statement of need

## 1.1 What the package is for

A PINN represents the solution of a differential equation by a neural network trained to satisfy the equation, its initial and boundary conditions, and any measured data [@raissi2019; @karniadakis2021]. The method is easy to state and hard to use well. Training fails in documented ways [@krishnapriyan2021; @wang2021], comparisons with classical solvers are often made against weak baselines [@mcgreivy2024], and on well-posed forward problems in low dimension a finite-element or finite-difference solver is faster and more accurate [@grossmann2024].

`pinnrl` was written with three goals that follow from that situation.

1. **Make the training step cheap enough that the method can be studied properly.** Multi-seed studies, identifiability analyses and calibration experiments need thousands of training runs. The package's step is compiled for the equation at hand and is, on the problems of Section 4.1, the fastest we could measure on a CPU.
2. **Say when an inverse estimate can be trusted.** A PINN returns a number for an unknown parameter whether or not the data determine it. The package computes what the data allow (the Cramér–Rao bound), what the surrogate adds (its floor), and whether the fitted model reproduces the records (a misfit test), and reports intervals whose coverage has been measured.
3. **Grade every result against the classical method for the same problem.** The package ships finite-difference baselines, reference solvers, an extended Kalman filter, a moving-horizon estimator and least-squares fits through reference solvers, and uses them as controls.

The application that drives the work is the digital twin of process equipment: estimating quantities a plant does not measure (the fouling of an evaporator tube, the drying-rate curve of a product, the stickiness margin of a powder) from the sensors it has.

## 1.2 When to use a PINN, and when not to

The package's own experiments give a narrower answer than much of the literature.

**Not for a well-posed forward problem with a cheap classical solve.** An explicit finite-difference scheme on the 1-D heat equation with 101 points reaches a relative $L^2$ error of about $7 \times 10^{-4}$ in about 30 ms. A PINN with L-BFGS reaches $2 \times 10^{-4}$ on heat in 120 s (Section 4.3).

**Not for estimating states or parameters of a model that has a cheap, trusted forward solve.** On the evaporator an 11-cell extended Kalman filter tracks the fouling drift to 3.5 % in 0.2 s where the PINN twin reaches 9–16 % in five minutes, and over repeated noise draws the PINN twin fails to train on 22 of 36 runs (Section 4.6). On the drying droplet a monotone closure fitted by shooting through the reference solver reaches 1.5–1.6 % on every seed in 8–10 s against the PINN's six to eight minutes and one lost seed in three. On the counter-current dryer, the unit chosen because its steady state is a boundary-value problem, bounded least squares through the transient recovers three parameters in 10 s and every PINN twin run is rejected by the misfit test.

**Where the case for a PINN remains.** Models with no trusted forward solve (no method-of-lines form at all); unknown functions inside the equations, with constraints that keep them physical; and amortised or parametric use, where one network replaces many solves (a parametric surrogate covers a parameter range to 0.2–2 % at a quarter to a half of the cost of five dedicated networks, Section 4.5). The package is built to test that case honestly, and its roadmap says where the remaining tests are.

## 1.3 Who it is for

Researchers who study PINN training and need a fast, instrumented trainer with reference solutions; engineers who want to know, before buying a sensor or running a campaign, what their records can determine; and teachers, since each exact solution in the package is checked by the test suite against its equation, initial condition and boundary conditions.

# 2. State of the field

**DeepXDE** [@lu2021] is the most widely used PINN library. It supports several backends (TensorFlow, PyTorch, JAX, PaddlePaddle), complex geometries by constructive solid geometry, operator learning, and a large set of documented examples. Its derivatives are taken by the backend's reverse-mode automatic differentiation.

**NVIDIA PhysicsNeMo**, formerly Modulus and before that SimNet [@hennigh2021], targets industrial scale: GPU clusters, parameterised geometries, and neural operators next to PINNs. It is the reference for large three-dimensional problems on accelerators.

**JAX-based stacks.** JAX [@jax2018] compiles a whole training step with XLA and offers forward mode and Taylor mode [@bettencourt2019] next to reverse mode. Several research PINN codes are built on it. Section 4.1 shows that the differentiation mode matters more than the framework: inside JAX itself, reverse over reverse is 1.4–6 times behind forward or Taylor mode.

**NeuralPDE.jl** [@zubov2021] is the PINN package of the Julia SciML ecosystem, with symbolic problem definitions and access to that ecosystem's stiff solvers and quadrature. **SciANN** [@haghighat2021] is a Keras wrapper aimed at ease of use.

**Classical baselines.** For forward problems: finite differences, finite elements and the method of lines with stiff integrators. For state and parameter estimation: the extended Kalman filter with a smoother, moving-horizon estimation, and least squares through a forward solver. For discovering terms of an equation: sparse regression [@brunton2016]. These are what a PINN result has to be compared with.

**Where `pinnrl` sits.** It is narrower than DeepXDE and PhysicsNeMo: boxes rather than general geometries, small networks on a CPU rather than large ones on a GPU, PyTorch [@paszke2019] only. Within that scope it differs in three ways: the training step is generated and compiled per equation; the inverse problem comes with identifiability, stability and calibration tools; and the process units come with classical estimators as controls. It also includes adaptive collocation sampling (residual-based adaptive distribution, RAD [@wu2023], and a reinforcement-learning bandit) and loaders for the simulation datasets of The Well [@ohana2024].

# 3. Design

## 3.1 Problems as text

A problem is a residual written as text, with the domain, the conditions and the parameters:

```python
from pinnrl import CustomPDE

pde = CustomPDE("u_t - alpha*u_xx", domain={"x": (0, 1), "t": (0, 1)},
                parameters={"alpha": 0.1}, initial_condition="sin(pi*x)",
                boundary_conditions={"dirichlet": {"value": 0.0}},
                exact_solution="exp(-alpha*pi^2*t)*sin(pi*x)")
trainer = pde.build_trainer("fourier", num_epochs=150)
trainer.train(progress=False)
```

`u_xx`, `u_xxt` and `theta_tt` are read as derivatives; other names are fields, coordinates, functions or parameters. The same class covers ordinary differential equations, steady problems and systems in any number of dimensions; parameters can be declared trainable; an `UnknownFunction` (a small network, optionally positive, monotone, or anchored to a value at a point) can stand inside the equation; sensors can observe any expression of the fields. A catalogue of 43 equations and ten registry problems with checked exact solutions come with the package, and seven architectures (MLP, ResNet, Fourier features, SIREN, FNO-style, attention, autoencoder).

## 3.2 The trainer

`PDETrainer` runs Adam, L-BFGS or Adam followed by L-BFGS, with loss weights (fixed or adaptive), schedulers, early stopping, time-marching windows, restarts judged by a sensor misfit, and four collocation samplers (uniform, stratified, RAD and the bandit). Inputs and outputs can be non-dimensionalised so the equations stay in physical units. A dashboard launches, monitors and compares runs.

## 3.3 The inverse-modelling toolbox

- **Identifiability** (`pinnrl.analysis`). `surrogate_sensitivities` computes $\partial u / \partial \theta$ from a trained network by the implicit-function theorem with matrix-free Hessian blocks; `fisher_information` turns it into a Cramér–Rao bound; `sensor_jacobian`, `posterior_covariance` and `greedy_design` price a sensor layout or a campaign before it is run.
- **Conditioning and stability.** `loss_hessian_spectrum` and `tangent_kernel_spectrum` give the spectra that limit training; `stability_ratio` measures the constant between residual and error; `perturb_observations` and `linear_response` predict what a sensor bias or a delay does to an estimate.
- **Uncertainty** (`pinnrl.uq`). A Laplace posterior from the surrogate's Fisher information, ensembles, and a particle swarm that samples the equivalence region of a misfit.
- **The misfit test.** For a unit with a reference solver, the fitted parameters or closure are run through that solver and the reduced $\chi^2$ of the records is returned. It is the acceptance test a twin applies to itself.

## 3.4 Process units

`pinnrl.process` contains a falling-film evaporator (steady and method-of-lines solvers, and the same model as a `CustomPDE`), a drying droplet with a characteristic drying curve, a co-current spray dryer, a counter-current dryer whose steady state is a two-point boundary-value problem, and `ProcessLine`, which chains evaporator passes into the dryer and propagates each unit's uncertainty to the powder. With them come an extended Kalman filter with a Rauch–Tung–Striebel smoother, a moving-horizon estimator, a fouling monitor and a historian adapter.

## 3.5 The native engine

A PINN residual such as $u_t - \alpha u_{xx}$ is usually built by applying reverse-mode automatic differentiation twice and backpropagating through that graph a third time. The package replaces this in three layers.

**A general Taylor-mode kernel.** For a feed-forward `tanh` network, with or without layer normalization, a C++ kernel carries the Taylor coefficients of the output with respect to chosen inputs forward through the layers, next to the value, and takes the weight gradient by one hand-written reverse pass. The coefficients of all points are stacked so each layer is one matrix product. Any order and mixed derivatives are supported (a multi-index set closed downward). It is wrapped as a `torch.autograd.Function`, so the rest of the loss is ordinary PyTorch.

**A step compiled per equation.** The general kernel walks index tables, one loop per term of each recurrence, and leaves the residual to PyTorch. For equations given as text the package instead writes C++ for those equations: the recurrences through `tanh` as one loop each way with a straight-line body for exactly the coefficient streams the equations use, and the residual, its square and its adjoint inlined (SymPy differentiates the expressions and prints them as C). The source is compiled once and cached; one call returns the mean squared residual and its gradient with respect to every weight. The general kernel serves the first 32 evaluations of a problem while the compiler runs, and the switch happens at exactly that evaluation, so a run does not depend on how long the build took. This is the specialisation a tracing compiler such as XLA applies to a whole step, built on the package's own kernel.

**A rational `tanh`.** The activation is an odd rational function on a clamped range (the form Eigen uses): 0.4 ns an element against 1.1 ns for Apple's `vvtanhf`, within $2.4 \times 10^{-7}$ of the exact value.

`derivatives="auto"` is the default: the engine is used wherever it applies and autograd elsewhere, without notice. A gradient that is itself differentiated (a Hessian-vector product) is taken by autograd on the same function, and inside a `torch.func` transform the network is the ordinary module, so no analysis tool has to know which engine ran. While building the kernel's layer-normalization pass we found that `torch.nn.LayerNorm` (PyTorch 2.10) returns a wrong weight gradient for any loss that contains second derivatives of the network with respect to its inputs (2–3 % off in norm on the heat problem, confirmed against finite differences in float64); the package's networks use `ExactLayerNorm`, the same function written in elementary operations.

# 4. Benchmarks

Timings are on an Apple M4, one thread, PyTorch 2.10, unless stated; Section 4.1 also reports a Linux x86 machine. Each subsection names the experiment folder under `experiments/` whose script reproduces it.

## 4.1 The training step against the alternatives (I17)

The same step in every framework: a `tanh` network $d$-64-64-64-64-$k$ in float32, the loss $\text{mean}(r^2) + 10\,\text{mean}(ic^2) + 10\,\text{mean}(bc^2)$ on 4,096 collocation, 512 initial and 512 Dirichlet boundary points drawn afresh every step, Adam. Each arm runs in its own process, checks its residual against a reverse-mode reference before it is timed, and keeps the best of three rounds of 100 steps over two passes. Milliseconds per step:

| Implementation | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | **3.7** | **3.7** | **3.7** | **4.4** | **5.5** | **5.7** | **5.7** | **3.8** | **5.1** |
| JAX, `jit`, forward mode (nested `jvp`) | 4.3 | 4.2 | 4.2 | 6.1 | 6.6 | 10.4 | 10.4 | 4.4 | 6.6 |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 4.8 | 4.8 | 4.8 | 6.8 | 8.9 | 9.6 | 9.6 | 5.1 | 7.3 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 5.2 | 5.1 | 5.1 | 7.2 | 8.4 | 10.0 | 10.1 | 5.5 | 7.7 |
| JAX, `jit`, reverse over reverse | 6.0 | 7.7 | 6.5 | 15.7 | 15.6 | 56.8 | 56.5 | 11.4 | 9.6 |
| PyTorch, nested `autograd.grad` | 8.7 | 8.8 | 8.8 | 17.7 | 17.7 | 43.6 | 43.6 | 17.7 | 12.0 |
| `pinnrl`, `derivatives="autograd"` | 10.0 | 10.2 | 10.1 | 19.1 | 19.1 | 45.0 | 45.1 | 15.6 | 14.5 |
| DeepXDE (PyTorch backend) | 10.1 | 10.0 | 10.0 | 21.5 | 21.9 | 56.0 | 55.7 | 18.8 | 14.5 |
| NVIDIA PhysicsNeMo, `PhysicsInformer` (autodiff) ¹ | 10.9 | 10.7 | 10.7 | n/s | n/s | n/s | n/s | 17.7 | 14.3 |
| `torch.func`, nested `jvp` | 16.8 | 17.1 | 16.9 | 35.7 | 30.9 | 92.1 | 92.3 | 17.1 | 27.3 |
| `torch.compile` | fails | fails | fails | fails | fails | fails | fails | fails | fails |

¹ PhysicsNeMo 2.2.2, timed in its own environment (PyTorch 2.15). `PhysicsInformer` takes derivatives up to second order (n/s: not supported); `nvidia-physicsnemo-sym` does not install without CUDA.

BBM is the Benjamin–Bona–Mahony equation, with a mixed third derivative $u_{xxt}$; K–S is Kuramoto–Sivashinsky; Schrödinger is the nonlinear Schrödinger equation as two real fields.

On the M4 `pinnrl` is first on every equation: 1.13–1.16 times JAX's best mode on the second-order problems in one dimension, 1.2–1.4 times at third order and in two dimensions, 1.8 times at fourth order; 2.4–7.7 times the textbook PyTorch loop and 2.7–9.9 times DeepXDE. It was not first when the experiment began. The first run had JAX's forward mode 1.3 times ahead on heat, and the two mechanisms of Section 3.5 (per-equation fused loops and the rational `tanh`) are what that run identified. The step compiled per equation is worth 1.3–1.7 times over the general kernel. `torch.compile` cannot run the step: with `fullgraph=True` it stops at `torch.autograd.grad` inside the function.

With layer normalization before each `tanh` (the kind of network the package ships as its default), same machine:

| Implementation | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | **7.7** | **7.5** | **7.5** | **9.6** | **12.6** | **12.7** | **12.7** | **7.6** | **11.4** |
| JAX, `jit`, forward mode (nested `jvp`) | 9.7 | 9.2 | 9.2 | 14.0 | 15.4 | 24.5 | 24.2 | 9.2 | 14.3 |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 10.3 | 10.1 | 9.8 | 13.7 | 17.6 | 18.4 | 18.2 | 10.0 | 14.6 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 10.5 | 10.7 | 10.8 | 13.9 | 17.3 | 18.9 | 19.0 | 10.9 | 15.4 |
| PyTorch, nested `autograd.grad` | 26.3 | 26.2 | 26.1 | 59.8 | 59.6 | 152.3 | 152.1 | 60.6 | 38.9 |
| `pinnrl`, `derivatives="autograd"` | 31.9 | 31.6 | 31.1 | 65.2 | 65.3 | 157.8 | 157.5 | 53.8 | 49.2 |

On a Linux x86 machine (a GitHub Actions runner, AMD EPYC 7763, OpenBLAS, one pass on a shared machine, so ratios within a column are what to read):

| Implementation | heat | Burgers | Allen–Cahn | KdV | BBM | Cahn–Hilliard | K–S | Schrödinger | heat 2-D |
|---|---|---|---|---|---|---|---|---|---|
| **`pinnrl`, the default (step compiled per equation)** | 24.2 | 24.0 | 23.9 | 30.7 | 38.1 | **38.4** | **38.6** | 24.2 | 37.3 |
| JAX, `jit`, forward mode (nested `jvp`) | **22.3** | **22.9** | **22.7** | **30.4** | **34.1** | 47.4 | 46.5 | **23.5** | **33.9** |
| `pinnrl`, general kernel (`PINNRL_NATIVE_SPECIALIZE=off`) | 27.4 | 27.4 | 27.5 | 35.0 | 41.9 | 43.1 | 43.0 | 27.9 | 40.7 |
| JAX, `jit`, Taylor mode (`jax.experimental.jet`) | 30.2 | 30.8 | 30.7 | 39.0 | 45.1 | 51.5 | 50.6 | 31.2 | 43.5 |
| PyTorch, nested `autograd.grad` | 55.1 | 61.2 | 58.9 | 135.2 | 128.5 | 349.2 | 343.9 | 99.4 | 79.5 |
| `pinnrl`, `derivatives="autograd"` | 45.9 | 46.8 | 45.7 | 84.4 | 81.4 | 185.3 | 183.8 | 74.5 | 66.2 |
| DeepXDE (PyTorch backend) | 71.3 | 70.3 | 64.6 | 188.5 | 182.9 | 466.4 | 492.6 | 141.8 | 95.2 |

There the lead over JAX does not hold on the plain network: JAX's forward mode is 1–11 % ahead on seven of the nine problems, and `pinnrl` leads by 1.2 times at fourth order. Without Apple's matrix unit the matrix products are most of the step, both engines call a tuned product, and the fused loops gain only 1.1 times over the general kernel. With layer normalization `pinnrl` leads on the five problems measured, by 1.3–1.7 times. Against eager PyTorch and DeepXDE the margins are those of the M4 (2.1–9 and 2.6–13 times).

Limits of these tables: one thread and one network size; the shared runner was measured once; and they give the cost of a step, not the error reached in a given time. Per-residual derivative kernels inside PyTorch and JAX on a GPU have been reported in a recent preprint (Cao, Lu and Zhang, arXiv 2606.29702), which we have not reproduced; what is measured here is a generated forward and reverse pass on a CPU.

## 4.2 Where the time of a step goes (I11, I13, I14, I15)

The heat step written ten ways (I11), 1 thread / 4 threads:

| Implementation | ms per step |
|---|---|
| C++ and Rust with no framework, Accelerate for the matrix products and `tanh` | 3.6 / 3.7 |
| `pinnrl` trainer, general kernel as released in v0.8 | 5.4 / 5.5 |
| PyTorch autograd, a minimal loop | 8.8 / 6.6 |
| `pinnrl` trainer, autograd | 10.6 / 8.3 |
| C++ and Rust with plain loops | 45 / 47 |

Sampling, the optimizer and bookkeeping are a tenth of a step. Taylor-mode derivatives written in tensor operations do not pay (9.3–12.4 ms against 8.8 for double backward); they pay with fused loops in compiled code. C++ and Rust are indistinguishable. A framework-free PINN is only worth writing on top of a tuned BLAS: with plain loops both languages are five times slower than PyTorch.

The general kernel against the package's own autograd step on higher derivatives (I13), medians of three seeds:

| Equation | plain `tanh` network | with layer normalization |
|---|---|---|
| heat, $u_{xx}$ | 1.9× (11.1 → 5.7 ms) | 3.0× (33.8 → 11.1 ms) |
| KdV, $u_{xxx}$ | 2.6× | 4.7× |
| BBM, $u_{xxt}$ | 2.0× | 3.6× |
| Euler–Bernoulli beam, $w_{xxxx}$ | 3.8× | 7.1× |
| Cahn–Hilliard, $u_{xxxx}$ | 4.0× | 7.2× |

Residual, loss and weight gradient agree with autograd to $10^{-7}$–$10^{-5}$ relative on every equation. The gain grows with the order, because autograd nests a graph per derivative and the kernel adds a stream per multi-index.

On Linux with OpenBLAS in Docker's virtual machine on the same M4 (I14) the general kernel gains 1.4 times (heat, plain) to 3.0 times (KdV, layer normalization), against 1.9–4.7 under Accelerate: without Apple's matrix units the twelve matrix products of a step are two thirds of it. Through the registry's built-in classes (I15) heat, KdV, Burgers and the pendulum gain 1.7, 2.7, 2.0 and 1.7 times on a plain network and 2.5, 4.7, 3.1 and 2.3 times with layer normalization.

## 4.3 Forward accuracy against classical solvers (classical baselines, D01)

| Method on 1-D heat | Relative $L^2$ error | Time |
|---|---|---|
| explicit finite differences, 101 points | about $7 \times 10^{-4}$ | about 30 ms |
| PINN, Adam | $10^{-3}$ | 120 s |
| PINN, L-BFGS | $2 \times 10^{-4}$ | 120 s |

The finite-difference scheme converges at second order. L-BFGS is 2–3 times better than Adam on Burgers and halves Allen–Cahn's error where Adam makes no progress; an early switch from Adam to L-BFGS diverged in 3 of 6 runs on Burgers. For a forward problem of this kind the classical solver reaches a comparable error several thousand times sooner, which is the comparison the package asks its users to make.

## 4.4 Collocation sampling at equal compute (B01, B06, B02–B08)

A 4 × 64 `tanh` network, 15,000 Adam steps on 1,000 points, 10 paired seeds; medians of the relative $L^2$ error, with a Wilcoxon signed-rank test against uniform:

| PDE | Sampler | Equal steps | vs uniform | Equal evaluations | vs uniform |
|---|---|---|---|---|---|
| Burgers | uniform | 0.055 | – | 0.055 | – |
| | stratified | 0.038 | 6/10, p = 0.63 | 0.038 | 6/10, p = 0.63 |
| | RAD | 0.136 | 3/10, p = 0.11 | 0.090 | 2/10, p = 0.06 |
| | RL bandit | 0.117 | 1/10, p = 0.06 | 0.105 | 2/10, p = 0.01 (worse) |
| Allen–Cahn | uniform | 0.568 | – | 0.568 | – |
| | RAD | 0.533 | 10/10, p = 0.002 | 0.547 | 10/10, p = 0.002 |
| | RL bandit | 0.520 | 10/10, p = 0.002 | 0.518 | 10/10, p = 0.002 |

On Burgers no sampler is significantly better than uniform, and charged its probe cost the bandit is significantly worse; seed-to-seed variation (six-fold) swamps any sampling effect. On Allen–Cahn the adaptive samplers win on every seed, but every method stays near the wrong solution (error above 0.5). On heat, convection, wave and the KdV soliton no sampler differs significantly from fresh uniform points (B02). On Burgers the resampling cadence changes the error three-fold, more than any choice of sampler at a fixed cadence (B04), and the error falls as about $N^{-1/3}$ in the number of points for uniform and RAD alike (B08). The package's recommendation is uniform or stratified by default and RAD for sharp features, measured against uniform at equal budget; the bandit that gives the package the second half of its name is labelled experimental.

## 4.5 Parameter identification (F01–F11, K01, K02, K45–K48)

**The heat diffusivity from noisy observations (F01).** $\alpha = 0.1$ started at 0.5, 10 seeds, median relative error of $\hat\alpha$, with weight decay and gradient clipping off:

| Noise $\sigma$ | $N_{\text{obs}} = 20$ | 50 | 200 | 1000 |
|---|---|---|---|---|
| 0 | 0.0010 | 0.0015 | 0.0017 | 0.0018 |
| 0.01 | 0.0098 | 0.0073 | 0.0024 | 0.0022 |
| 0.05 | 0.024 | 0.028 | 0.011 | 0.0052 |
| 0.1 | 0.044 | 0.055 | 0.016 | 0.010 |

With the packaged regularization the same runs stalled at 1.5–2 % whatever the number of observations: the defaults that help a forward fit act as priors on the surrogate that the parameter absorbs, so inverse runs start from `weight_decay=0`, `gradient_clipping=0`. Above the floor the spread of $\hat\alpha$ across seeds matches the Cramér–Rao bound to within about 20 % in most cells. For a fixed data set, ten starting values from 0.1 to 10 times the truth land within 0.3 % of each other (F04). One probe at mid-domain has a lower bound than scattered points (1.2 % against 1.7 % at noise 0.05) and the PINN delivers it (1.1 % against 1.9 %, F05).

**Where the surrogate limits the estimate (F02, F03, K01).** For the two coefficients of the KdV soliton the bound says 200 clean points determine them to 0.6 % and 1.4 %; the PINN returns them 18 % and 25 % low on every seed. For the Burgers viscosity at $\nu = 0.01/\pi$ the network resolves a shock several times wider than the real one and reports a viscosity 2.5–5 times too large. The Fisher information computed from the trained network agrees with the exact one to 1–4 % on heat and KdV and is 2–25 times too optimistic on the Burgers shock. Field error does not predict parameter error.

**Functions as unknowns (F09, F11, K48).** With 200 observations at noise 0.01, relative $L^2$ error of the identified coefficient:

| Truth | Network unknown | A scalar model instead |
|---|---|---|
| $\alpha(t) = 0.05 + 0.1t$ | 0.037 | 0.32 |
| $\alpha(t)$ with a step at $t = 0.5$ | 0.254 | 0.50 |
| $\alpha(x)$, centred bump | 0.405 | 0.50 |
| $\alpha(x)$, two layers | 0.125 | 0.62 |

A linear drift is recovered to a few percent. A step comes back as a ramp; parametrised as a step, its time is identified to the bound (median error 0.0075 against 0.0062 at noise 0.01, F11). A field is identified only where the solution feels it: with an exact forward model the same data determine all eight cosine modes of either field to 0.001–0.006 (K48), so the collapse of the bump is the surrogate's limit, not the data's.

**Calibration (F06, K46, K47).** Over 30 synthetic data sets the Laplace interval's standard deviation equals the spread of the estimate to three digits and covers the truth 77 % of the time at the 90 % level, because every estimate in that study was 2 % low; an ensemble over initializations covers 13 %; the swarm's likelihood-ratio region covers 90 %. Weighted as a posterior sample, the swarm is centred correctly but 0.5–0.7 as wide, with 80–87 % coverage.

**A parametric surrogate as the forward model (K45).** One network with the parameters as inputs covers its range to 0.2–0.3 % on heat and Burgers and 0.2–2 % on KdV at a quarter to a half of the cost of five dedicated networks (9–19 minutes against 23–56). The swarm run on the surrogate's misfit agrees with the swarm on the closed form: best fits to 0.04–0.3 % of the truth, 90 % widths to 1–3 %, at 0.1–0.2 s per data set.

## 4.6 Fouling of an evaporator: the PINN twin against classical estimators (J03–J06, J09, I16)

Two outlet sensors every 5 s over 20 minutes, the fouling resistance a drifting function of time.

| Estimator | $L^2$ error of the drift | Error at the end | Time |
|---|---|---|---|
| best constant | 53–55 % | | |
| PINN twin (J03, three seeds) | 9–16 % | 10–49 % | five minutes |
| extended Kalman filter, 11 cells (J05) | 3.5 % | 2 % | 0.2 s |
| smoother, 51 cells (J05) | 2.3 % | 0.7 % | |
| moving-horizon estimator, 11 cells (J05b) | 3.4 % | 2.1 % | 10–35 times the filter |
| what the data allow (Fisher bound, J04) | | 5 % | |

With the model known the filter beats the PINN twin by a wide margin at a thousandth of the cost. The identifiability analysis (J04) says why some things cannot be estimated by any method: the clean heat-transfer coefficient and the constant fouling are one unknown, and 0.5 K of steam bias reproduces the records of a 30 % fouling error.

Coverage of the 90 % interval over 100 noise draws (J06):

| Estimator and assumption | Coverage |
|---|---|
| batch fits through the reference solver, model right | 0.92–0.96 |
| trend filter, 51 cells | 0.91–0.99 |
| trend filter, 11 cells (a model biased by 2.7 %) | 0.33–0.69 |
| any estimator, a steam record uncertain by 0.3 K taken at face value | 0.21–0.52 |
| batch fits with the steam offset estimated under its prior | 0.89 at ±12–15 % |
| PINN twin, accepted runs, Fisher width | 0.25 |
| PINN twin, ensembles at ±31–45 % | 0.60–0.80 |

The PINN twin fails to train on 22 of 36 runs, which its own sensor misfit detects; its accepted runs are 26 % off at the end of the record against the 5 % the data allow. A measurement-design calculation (J09) then prices the remedies for the classical estimate before any is bought: with every instrument error admitted the end value is known to 10.6 %; a condensate meter brings it to 4.9 %, and with a 4 K steam swing to 2.8 %; 200 simulated fits per design land within 5 % of the predicted spreads.

The evaporator's reference solve was itself moved to C++ (I16): a batch fit of the fouling goes from 106 s with SciPy's BDF to 0.68 s with the same estimate, and one solve on a record with 10 s knots from 0.7–6.5 s to 3.5–108 ms depending on grid and tolerance.

## 4.7 A drying closure: droplet and dryer (J07, J08, J11)

The drying-rate curve $f(X)$ identified as an unknown function from a mass-and-temperature record of one droplet:

| Method | Error of the closure | Time | Reliability |
|---|---|---|---|
| PINN, unknown function from a cold start | collapses to a constant | | every seed |
| PINN refining a pointwise estimate of the record (J07b) | 1.7–2.1 % | six to eight minutes | one seed in three destroyed |
| pointwise estimate with the closure anchored at equilibrium (J07e) | 2–3 % | one second | |
| monotone closure by shooting through the reference solver (J07d) | 1.5–1.6 % | 8–10 s | every seed |
| least squares with the right parametric form | parameters to 1 % | 0.2 s | |

The misfit test through the reference solver (J07c) returns a reduced $\chi^2$ of 0.9 for the true curve, 1.5–1.7 for the two good refinements and 390–765 for the failed runs; it catches every failure, including one the network's own fields hide (fields within 0.8 %, closure 20 % off).

At dryer scale (J11) one solve takes 5 ms, so the estimator is least squares through it. Five operating points give the two parameters of a known curve form to ±0.03 each. A free-form closure is not identifiable from a tower's instruments (4 to 6 of 11 directions), yet the powder moisture outside the campaign is predicted to 0.9–1.6 %. For the stickiness margin of the powder (J08), standard deviation of the estimate at the outlet:

| Estimate | Error |
|---|---|
| moisture from the water balance of the air | 23–59 K |
| lab assay of the powder, hours late | 1.6–3.2 K |
| the twin, identified on a five-point campaign and run forward | 1.0–1.5 K |

## 4.8 A unit without a cheap initial-value solve (J13)

The counter-current dryer was the test the roadmap set for the PINN twin's case.

| Solve | Time |
|---|---|
| co-current steady state | 4 ms |
| counter-current steady state, shooting | 103 ms |
| transient, 600 s, 51 / 201 / 401 cells | 0.04 / 0.07 / 0.13 s |
| least squares, three parameters from one transient record | 10 s |
| PINN forward solve / PINN twin | 96 s / 110–140 s |

The boundary-value problem is the expensive part and is not expensive enough. The PINN forward solve is 12–15 % off on the moisture on two seeds of three. Bounded least squares through the transient lands within the linear prediction on all three data sets; the PINN twin's six runs are rejected by the misfit test (reduced $\chi^2$ 5–9). The case for the PINN twin is therefore not made on this unit, and the open test is a model with no method-of-lines form at all (a droplet size distribution, wall deposition).

## 4.9 A line of units (J10)

Three evaporator passes feeding the spray dryer over a four-hour shift. Standard deviation of the powder moisture at the end of the shift:

| Estimate | Std |
|---|---|
| propagation, fouling unmonitored | 0.16 |
| propagation, fouling as the monitors have it | 0.094 |
| posterior from the dryer's online records alone | 0.0054 |
| dryer's records and four lab assays | 0.0022 |
| every record of the line | 0.0011 |

Propagating uncertainty down the line is the wrong estimator for the product: the dryer's own instruments give a posterior 17 times narrower. Information flows downstream only, and the first pass's fouling is one unknown with the feed flow (0.99 collinear).

## 4.10 Conditioning, stability and dimension (K03, K05–K09, H04)

Across eight operators after 3,000 steps a `tanh` MLP's residual tangent kernel has a condition number near $10^8$ whatever the operator; Fourier features bring it down by three to five orders but unbalance the residual and boundary kernels by $10^3$–$4 \times 10^3$ on wave-type and fourth-order problems, the gradient pathology of @wang2021. A sensor bias shifts an identified parameter by the linear prediction to within 20 %; a bias of the noise level moves the heat diffusivity by 18 % and nothing in the residual flags it (K05). For heat in $d = 1 \ldots 5$ at a fixed budget the residual stays at 2–7 % while the error grows from 2 % to 44 % (K08). On systems of ordinary differential equations the PINN fails by collapsing to a trivial solution, not by stiffness; time-marching windows rescue the SIR model (error 1.0 to 0.1 %) and nothing rescues Lorenz (K09, H04).

# 5. Limitations

- **Scope.** Box domains, small networks on a CPU, PyTorch only. For complex geometries, large three-dimensional problems or GPU clusters, DeepXDE and PhysicsNeMo are the tools.
- **The compiled step does not cover everything yet.** It applies to feed-forward `tanh` networks (with or without layer normalization and input or output scaling) and to equations given as text, including trainable parameters and unknown functions of the coordinates. Other architectures and activations, steady problems, losses other than the squared one and unknown functions of a field run on the earlier paths.
- **The lead depends on the machine.** It holds on an Apple M4 and, on x86, only at fourth order and with layer normalization. Everything is one thread.
- **Simulated plants.** Every process result is on records simulated from the model that is then fitted, with noise and instrument errors added. The first comparison against a real unit is still to be made.
- **The reinforcement-learning component is one bandit**, and it does not beat uniform sampling.

# 6. Roadmap

The roadmap in the documentation orders the work as: the compiled step extended to other architectures and activations, and research bets on the error reached per second (v0.9); process units as Gymnasium environments with PID and model-predictive-control baselines that a learned policy must beat (v0.10); sequential experiment design and training itself as decision problems (v0.11); from a learned closure to an equation, and model-form error named (v0.12); parametric surrogates and a unit library (v0.13); a stable API (v1.0); and, after a rehearsal against a simulated plant behind real interfaces, a real line (v7.0).

# 7. Reproducibility

```bash
pip install pinnrl
git clone https://github.com/josegarciav/PINNs-RL-PDE && cd PINNs-RL-PDE
uv sync --all-extras
uv run pytest                                                   # the test suite
uv run --with jax --with deepxde python experiments/I17_step_head_to_head/run.py
python experiments/J06_fouling_intervals/run.py --datasets 100 --workers 6
pinnrl-benchmark sampling --seeds 10 --steps 15000 --checkpoints 50 --workers 4
```

Each experiment folder holds a `run.py`, usually a `plot.py`, and a README with the full tables, interquartile ranges and the follow-ups it motivated. The release process builds a clean wheel, runs every packaged problem, every documentation snippet and the tutorial notebooks against it before anything is published.

# Acknowledgements

`pinnrl` started as the MSc thesis *On Solving Partial Differential Equations (PDEs) Using Physics-Informed Neural Networks (PINNs)* (University of the Basque Country UPV/EHU, 2025).

# References
