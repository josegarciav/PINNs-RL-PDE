# When to use a PINN, and when not to

The package's own experiments give a narrower answer than much of the literature. Every figure here is from the [benchmarks page](benchmarks.md) or the experiment named next to it.

## What a PINN is

A network $u_w(x, t)$ trained to satisfy an equation $\mathcal{N}[u] = 0$, its conditions and any data:

$$
L(w) = \underbrace{\lVert \mathcal{N}[u_w] \rVert^2}_{\text{residual}}
     + \lambda_b \underbrace{\lVert u_w - g \rVert^2_{\partial\Omega}}_{\text{boundary}}
     + \lambda_0 \underbrace{\lVert u_w(\cdot, 0) - u_0 \rVert^2}_{\text{initial}}
     + \lambda_d \underbrace{\lVert u_w - y \rVert^2_{\text{data}}}_{\text{observations}}.
$$

- **PINN:** physics-informed neural network; the network *is* the solution, and the loss $L$ is the equation.
- **Forward problem:** the equation is known, find $u$.
- **Inverse problem:** part of the equation (a parameter $\theta$, a function) is unknown; find it from data $y$.
- **Classical solver:** finite differences, finite elements, or the method of lines; it solves the forward problem on a grid.
- **Twin:** a model of one piece of equipment, fitted to its sensors, that estimates what is not measured.
- **Misfit test:** the fitted model run through a reference solver and compared with the records; a reduced $\chi^2$ near 1 is a fit.

## Do not use a PINN when

**1. The forward problem is well posed and a classical solver exists.**

| Method on 1-D heat | Relative $L^2$ error | Time |
|---|---|---|
| Explicit finite differences, 101 points (`pinnrl-benchmark fdm --pde heat`) | about $7 \times 10^{-4}$ | about 30 ms |
| PINN, Adam ([D01](benchmarks.md#optimizers-and-the-floor-experiments-d01-d02)) | $10^{-3}$ | 120 s |
| PINN, L-BFGS (D01) | $2 \times 10^{-4}$ | 120 s |

The classical solver gets a comparable error several thousand times sooner.

**2. The model has a cheap, trusted forward solve and you want its states or parameters.**

| Unit | Classical estimator | PINN twin |
|---|---|---|
| Evaporator fouling (J03, J05, J06) | an 11-cell extended Kalman filter tracks the drift to 3.5 % in 0.2 s | 9–16 % in five minutes; fails to train on 22 of 36 repeated runs |
| Drying-rate closure of a droplet (J07) | a monotone closure by shooting through the reference solver: 1.5–1.6 % on every seed in 8–10 s | 1.7–2.1 % in six to eight minutes from a good start, one seed in three destroyed; collapses to a constant from a cold start |
| Counter-current dryer (J13) | bounded least squares recovers three parameters in 10 s | every run rejected by the misfit test (reduced $\chi^2$ 5–9) |

**3. You need an interval you can trust and have not checked it.** A PINN returns a number for a parameter whether or not the data determine it.

- On the evaporator the twin's own interval covers the truth 25 % of the time at the 90 % level (J06).
- On the KdV soliton the data determine the two coefficients to 0.6 % and 1.4 %; the PINN returns them 18 % and 25 % low on every seed (F02, K01).
- Field error does not predict parameter error.

## Consider a PINN when

- **There is no trusted forward solver** for the model, or it is too expensive to iterate. This is the open test on the [roadmap](roadmap.md): a unit with no method-of-lines form (a droplet size distribution, wall deposition).
- **An unknown function sits inside the equation** (a coefficient that drifts, a closure) and you can constrain it to stay physical. A linear drift $\alpha(t)$ of the heat equation is recovered to 4 % where a scalar model is 32 % off (F09).
- **One network should replace many solves.** A parametric surrogate covers a parameter range to 0.2–2 % at a quarter to a half of the cost of five dedicated networks, and then answers in milliseconds (K45).
- **Physics has to be combined with sparse data**, or the problem is mesh-free or high-dimensional. Check the error, not the residual: for heat in $d = 1 \ldots 5$ at a fixed budget the residual stays at 2–7 % while the error grows from 2 % to 44 % (K08).

In every case, run the baseline. The package ships it so that you can:

| Question | The control in the package |
|---|---|
| Is my forward solution right? | `pinnrl-benchmark fdm`, the reference solutions of `pinnrl.benchmarks`, checked exact solutions |
| Is my sampler helping? | `pinnrl-benchmark sampling`: paired seeds at equal steps and equal evaluations against uniform |
| Do the data determine this parameter? | `pinnrl.analysis.fisher_information`, `sensor_jacobian`, `greedy_design` |
| Is my interval calibrated? | `pinnrl.uq.coverage` over repeated data sets; the swarm's region of equivalence |
| Did my twin fit the records? | `pde.sensor_misfit`, `DryingDroplet.record_misfit`: the fitted model through a reference solver |
| Would a filter do better? | `pinnrl.process.ExtendedKalmanFilter`, `MovingHorizonEstimator`, least squares through `reference()` |

## Who the package is for

- **Researchers** studying PINN training, sampling or identifiability, who need a fast trainer, reference solutions and multi-seed protocols.
- **Process and control engineers** who want to know what their records can determine before buying a sensor or running a campaign.
- **Teachers and students**: every exact solution in the package is checked by the test suite against its equation, initial condition and boundary conditions.

## How `pinnrl` compares with the alternatives

| | Best at | Choose it over `pinnrl` when |
|---|---|---|
| [DeepXDE](https://github.com/lululxvi/deepxde) | The widest coverage: several backends, complex geometries, operator learning, many worked examples | You need a geometry that is not a box, a backend other than PyTorch, or DeepONets |
| [NVIDIA PhysicsNeMo](https://github.com/NVIDIA/physicsnemo) (formerly Modulus) | Industrial scale on GPUs: 3-D parameterised geometries, neural operators, multi-GPU training | The problem is large, three-dimensional and you have accelerators |
| JAX-based PINN code | A whole step compiled by XLA, forward and Taylor mode, GPUs, any architecture you can write | You need a custom architecture or a GPU and are willing to write the training loop |
| [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) | Symbolic problem definitions inside Julia's SciML ecosystem, with its stiff solvers | You work in Julia |
| **`pinnrl`** | The fastest step on a CPU for small `tanh` networks; identifiability, calibration and misfit tools for inverse problems; process units with classical estimators | You want to know whether a PINN is the right tool for your inverse problem, and how far to trust its answer |

`pinnrl` is narrower than the first two: box domains, small networks, PyTorch on a CPU. It runs on Apple and CUDA GPUs through autograd, without the compiled step.

**On speed.** The same training step is timed in every framework in [experiment I17](benchmarks.md#the-step-against-the-alternatives-experiment-i17). It is one step on a CPU with a small network. It says nothing about large networks on a GPU, and it measures a step, not error at equal time.

**What the others do that `pinnrl` does not:**

- geometries that are not boxes;
- neural operators (DeepONet, FNO at scale);
- multi-GPU training;
- backends other than PyTorch.

If your problem needs one of these, use the tool that has it.

**What `pinnrl` adds:**

1. A training step generated and compiled per equation ([the engine](engine.md)).
2. For an inverse problem: the bound the data allow, the floor the network adds, a measured interval and a misfit test ([inverse problems](inverse_problems.md), [theory](theory/index.md)).
3. Process units that ship with the estimators a PINN has to beat ([process twins](process_twins.md)).
