# Roadmap

What is planned, in the order the [research program](research.md) needs it. Each item states what a user gains and, where it can be said in advance, when it counts as done. Within a release, items are worked top to bottom. What is already built is on the [history page](history.md).

## Where this goes

Eight releases built three things: a PINN trainer whose derivatives run in a compiled kernel, an inverse-modelling toolbox that says when an estimate can be trusted (identifiability, stability, calibrated intervals, a misfit test), and four process simulators with their classical estimators. Two findings shape what comes next:

- **On every unit with a cheap reference solve, the classical estimator beat the PINN twin** (J05, J07d, J11, J13). The PINN's case is narrower than the literature claims: models with no trusted forward solve, unknown functions inside the equations, and amortised or parametric use, where one network replaces thousands of solves.
- **The "RL" in `pinnrl` is one contextual bandit** that chooses collocation cells and does not beat uniform sampling at equal cost (B01/B06). Meanwhile the package owns what reinforcement learning is usually short of: fast, validated, physically meaningful simulators (a line shift in 1.6 s, an evaporator solve in milliseconds) with realistic sensors, hidden states and economic stakes.

The direction follows from both: **`pinnrl` becomes the place where physics-informed models and sequential decisions meet.** PINNs estimate what the plant does not measure; reinforcement learning and optimal control decide what to do about it (how to run, when to clean, what to measure next), in environments built from the same equations, graded against the classical controller as the estimators were graded against the Kalman filter.

| Theme | Release | What a user gains |
|---|---|---|
| Accuracy per second (research bets, below) | v0.9–v0.10 | A lower error in the same time, by an optimizer, an architecture and a training controller built for physics-informed losses |
| The engine, everywhere | v0.9 | The fastest PINN step on a CPU on every problem the package solves and on every machine; custom networks at the same speed; twins that run outside Python |
| Tests that protect the architecture | every release | Nothing existing breaks when a component is added; dead code removed |
| The dashboard as an experiment tracker | v0.9–v0.10 | Runs searched, compared and traced from data to result |
| `pinnrl.gym`: process environments and control | v0.10 | Standard RL environments of real unit operations, with baselines (PID, MPC) and a twin in the loop |
| Decisions that serve the inverse problem | v0.11 | What to measure, where to sample and how to train, learned as policies and compared with the greedy designs |
| Discovering unknown physics | v0.12 | From a learned closure to an equation with an error bar; model-form error named instead of absorbed |
| Simulation at scale | v0.13 | One network across operating points; new units and geometries; inference in microseconds |
| A stable platform | v1.0 | An API that does not move under a user; the software paper |
| The platform matured (see below) | v2.0–v6.0 | Control, discovery, scale and an autonomous twin, each proven on simulated plants |
| The plant | v7.0 | Live data in, advice with intervals out, validated on a real line |

## Research bets: accuracy per second

On an Apple M4 the engine leads on the time of a step ([I17](benchmarks.md#the-step-against-the-alternatives-experiment-i17)). A user feels something else: the error reached in a given time. The bets below aim at that number, and they come before any publication. Each is a claim that can fail.

**The measure**

$$
\varepsilon(T) = \frac{\lVert u_\theta - u \rVert_2}{\lVert u \rVert_2} \quad \text{after wall-clock time } T \text{ on one CPU core}
$$

- **$u_\theta$**: the trained network.
- **$u$**: the exact or reference solution.
- **Protocol**: ten seeds, median and interquartile range, the nine equations of I17 plus published suites (PINNacle, PDEBench), the same machine for every method.

**The structure the bets use**

$$
L(\theta) = \tfrac12 \lVert r(\theta) \rVert^2, \qquad r = \big(w_r\, r_{\text{pde}},\; w_b\, r_{\text{bc}},\; w_0\, r_{\text{ic}}\big)
$$

- **$r(\theta)$**: every residual of the problem stacked in one vector (equation, boundary, initial).
- **Least-squares problem**: a loss that is a sum of squares. Adam and L-BFGS treat it as a generic function; the methods below use the squares.
- **Jacobian $J = \partial r / \partial \theta$**: how each residual moves with each weight. The engine can apply $J$ and $J^\top$ to a vector at the cost of a step, without forming $J$.

| Bet | The idea | Baseline to beat | It fails if |
|---|---|---|---|
| R1. A least-squares optimizer | Levenberg–Marquardt steps $\delta = -(J^\top J + \lambda I)^{-1} J^\top r$, solved matrix-free with the engine's $J v$ and $J^\top w$ | Adam then L-BFGS at equal time | It does not reach the published Gauss–Newton errors (about $10^{-6}$ on heat) on one CPU core in minutes, or is not at least 100 times below Adam then L-BFGS at equal time on six of the nine equations |
| R2. The last layer solved, not trained | With $u_\theta = \sum_j c_j \phi_j(x; \vartheta)$ and a linear equation, $r$ is linear in $c$: solve $c$ by least squares at every step and optimize only $\vartheta$ (variable projection) | R1 without it | No gain in $\varepsilon(T)$ on the linear problems (heat, 2-D heat) |
| R3. Conditions built into the network | $u_\theta = g + d \cdot N_\theta$ with $g$ matching the initial and boundary data and $d$ vanishing there, compiled like the rest | Penalty terms with tuned weights | The error is not lower without any loss weight to tune |
| R4. A learned training controller | A policy sets the damping $\lambda$, the resampling and the restarts of R1 from the loss history, trained across equations | Marquardt's and Nielsen's damping rules | It does not beat the rules on equations it was not trained on |
| R5. A benchmark others accept | $\varepsilon(T)$ curves against DeepXDE and JAX baselines on public suites, scripts in the repository | The best published number per task | The lead of R1–R3 does not survive on problems the package was not tuned on |
| R6. A network designed for the loop (below) | A trunk cached once and a small head per step, so a model queried millions of times in a reinforcement-learning loop costs microseconds and adapts by a least-squares update | A general network retrained by Adam; the reference solver as environment | The loop is not 10 times faster, or the policy is worse than one trained on the solver |

- **Levenberg–Marquardt**: Gauss–Newton with a damping term $\lambda I$ that keeps the step inside the region where the linearization holds.
- **Matrix-free**: the linear system is solved by conjugate gradients, which needs only products with $J$ and $J^\top$.
- **Variable projection**: eliminating the parameters a model is linear in, and optimizing over the rest.

**What is new and what is not** (from a literature review of November 2026; absence means "not found", not "proven absent"):

| Bet | Already published | Not found, and what the bet has to show |
|---|---|---|
| R1 | Gauss–Newton, Levenberg–Marquardt, natural-gradient and self-scaled quasi-Newton methods for PINNs, matrix-free and with Taylor mode, three to five orders below Adam on GPUs in double precision | The same accuracy on one CPU core in comparable time; the measured cost of $Jv$ and $J^\top w$ against one gradient; where a dense solve beats the matrix-free one; single precision; inverse problems |
| R2 | Random-feature and extreme-learning-machine solvers; variable projection on hidden weights | The solve timed in compiled code, and combined with R1 and R3 |
| R3 | Distance-function and functional-connection ansatz, also paired with an exact last-layer solve | The three together with trained hidden weights; time-dependent nonlinear problems; inverse problems |
| R4 | Learned damping on classical least squares, with thin evidence; adaptive sampling and loss weights, which rarely beat simple baselines at equal compute | Control of damping, switching and restarts for R1, trained across equations and tested on unseen ones, against Marquardt's and Nielsen's rules. A null result is worth reporting |
| R5 | Suites that report final error (PINNacle, the jaxpi leaderboard) | Error against wall-clock time, CPU timings, a small-network track, inverse problems scored on identifiability and calibration |
| R6 | The trunk-and-head form (DeepONet); online last-layer adaptation in flight control and Koopman control | Recursive least squares on a physics-trained cached basis; closed-form action gradients; latency on one CPU core; a policy trained on the model against one trained on the solver |

- **The bar for R1 is higher than first written.** Published Gauss–Newton results reach errors near $10^{-6}$ on heat and $10^{-7}$ on Poisson in minutes on a GPU. Ten times better than Adam is not enough: R1 has to reach those errors on a CPU core.
- **A warning for R6.** In one benchmark of predictive control on reactors, PINN surrogates were slower than the mechanistic model in every case, because of the cost of derivatives. R6 exists to remove that cost and must be judged against that result.
- **For industry.** No PINN validated on plant data for an evaporator or a spray dryer was found; plants run grey-box models with predictive control. Readiness there is a lifecycle matter (historian validation, shadow mode, drift monitoring, change control), which is what v6.0 and v7.0 are for.

**R1 has a research plan and a first prototype** in `experiments/D10_least_squares_optimizer/` (`PLAN.md`, `run.py`). The plan goes beyond Levenberg–Marquardt: nine moves that use what a physics-informed loss is (the residual's flow integrated like a stiff ODE, a coarse classical solve as the preconditioner, training as filtering over a stream of fresh points, small blocks solved exactly with the sampler choosing them, the linear layer solved and the basis grown, continuation in a physical parameter, the equation's own time as the training loop, multipliers in place of loss weights) and an agent that chooses among them, which is R4 widened from damping to moves. Each has a test and a condition under which it is dropped.

First measurements (one core, three seeds, heat, Burgers and KdV): at 60 s Levenberg–Marquardt reaches $6 \times 10^{-7}$ on heat and $6 \times 10^{-6}$ on KdV, 120 and 220 times below Adam then L-BFGS, and only 3 times below on Burgers. An extended Kalman filter over a stream of fresh points is slower for the first minute and then ahead on all three: at 180 s, $2 \times 10^{-5}$ on Burgers (65 times below Levenberg–Marquardt), $1.5 \times 10^{-9}$ on heat, $10^{-7}$ on KdV. Solving the last layer reaches $5 \times 10^{-9}$ on heat in 60 s. Four of the seven moves tried were dropped. At 180 s the filter is 350 (Burgers) to 35,000 (heat) times below Adam then L-BFGS. Twenty seconds of Levenberg–Marquardt before the filter remove its slow start at no loss. Open: the filter's covariance costs the square of the number of weights; kept only within each layer it holds on heat and KdV and loses a factor of 20 on Burgers. At 12,737 weights a low-rank form keeps level with Levenberg–Marquardt, not ahead: the filter's lead does not yet survive the size. A memory that grows with the stream (short early, long late) puts the filter ahead of Levenberg–Marquardt from the first seconds on heat and KdV, and 150, 17 and 12 times below it at 180 s on heat, Burgers and KdV. Adam then L-BFGS was too weak a baseline. Self-scaled Broyden (Urbán, Stefanou and Pons, 2024) with little or no Adam first reaches $2 \times 10^{-10}$ to $4 \times 10^{-10}$ on heat and KdV in 180 s. R1's bar is now the best published method at every moment of the run. Against it the filter leads until 68 s on heat and 37 s on KdV, and on Burgers throughout, 14 times at the end, on the same loss. The filter then self-scaled Broyden meets the bar on heat. On KdV nothing here does after 37 s. On Allen–Cahn the filter meets it alone: 250 times below the best self-scaled Broyden at 20 s, 2.3 times at 180 s ($3.3 \times 10^{-6}$), and it never lands on a wrong solution, where Broyden does on up to two seeds of three. Burgers at $\nu = 0.003/\pi$ defeats every method on uniform points: the front is 8 times narrower than their spacing. The filter breaks down there from covariance windup; a step limit on the change in $u$ makes it stable at no cost by steps on the smooth equations, and it then stops near 0.3. Levenberg–Marquardt on points redrawn from the residual every 5 steps fits the front to 3e-03, 50 times below self-scaled Broyden (provisional: measured on battery). On a sharp front the points decide.

Order of work: R1 as a prototype in plain PyTorch first (to measure what it buys), then in the engine; R2 and R3 on top; R5 throughout; R4 and R6 last, because R4 needs R1 and R6 needs R2. The review is in the repository's `reports/` folder when it is kept there.

### R6. A network designed for the loop

A reinforcement-learning loop calls the physics model at every step of every episode: millions of evaluations, each for a new state and action, often with a small update in between. A general network recomputes everything at each call. This bet is an architecture, built together with the engine, in which almost nothing has to be recomputed.

$$
u_\theta(x, t;\, z) \;=\; \sum_{j=1}^{m} c_j(z)\, \phi_j(x, t), \qquad c(z) = A\, h(z) + b
$$

- **$z$**: what changes at every step of the loop (the state, the action, the operating point, the estimated parameters).
- **Trunk $\phi_j(x,t)$**: features of space and time and their derivative streams. They do not depend on $z$, so they are computed once and cached.
- **Head $c(z)$**: a small map from the loop's variables to the coefficients. It is all that runs per step.
- **Linear in $c$**: for a fixed trunk the residual is linear or low-order in the coefficients, so an update is a least-squares solve (R2), not a gradient-descent run.

| Property | How the design gets it | Done when |
|---|---|---|
| A query costs microseconds | Cached trunk streams; the head is one small matrix product; batched over parallel environments in one call | One core serves a stated number of queries per second, with $\partial u / \partial z$ included |
| The model adapts inside the loop | Recursive least squares on $c$ as new sensor data arrive, one rank update per measurement | The twin tracks a drifting parameter at every environment step |
| The policy can differentiate through it | $\partial u / \partial z$ in closed form from the head, exact, no graph | A policy gradient through the model matches finite differences |
| Rollouts stay stable | The network advances the state over one control interval, with the physics residual and a bound on its growth built in | A long rollout does not drift from the reference solver |
| It runs outside Python | Fixed widths and block sizes chosen for the cache; exported as C with the engine's export | The exported model reproduces the Python one to rounding |

- **Recursive least squares**: updating a least-squares solution when one more data point arrives, without solving again from the start.
- **Rollout**: applying the model to its own output step after step, as a simulator does.
- **Control interval**: the time between two actions of the controller.

Baseline to beat: the same task with a general 4 × 64 network retrained by Adam, and with the reference solver as the environment. It fails if a policy trained on this model does not match one trained on the solver, or if the loop is not at least ten times faster than with the general network. It builds on R2 (the linear last layer) and feeds [`pinnrl.gym`](#v010-pinnrlgym-process-environments-and-control), where the PINN world model is planned.

## v0.9: the engine, everywhere

In progress. Already built in this release ([history](history.md#unreleased-v09-so-far)): the compiled kernel is the default engine; a benchmark against PyTorch, JAX and DeepXDE on nine equations ([I17](benchmarks.md#the-step-against-the-alternatives-experiment-i17)); a step compiled per equation, which covers forward and inverse problems, scaled networks and layer normalization (1.3–1.4× the general kernel with normalization on). The aim of what remains is that the engine's lead holds on every problem the package solves and on the machines most users have, and that everything later (control, design, discovery) is built on it.

- **The rest of the step in the compiled call.** The boundary and initial points, the loss reduction and Adam inside the same call (0.5 ms of heat's 3.5); L-BFGS on the compiled gradients in float64, since L-BFGS is what does the learning on the stiff process problems (J13); steady problems; unknown functions of a field (a closure $f(u)$), which still run on autograd. *Gain: one call per step on every problem, and second-order training at engine speed.*
- **Beyond one core, beyond one machine.** The step's matrix products are single-threaded on Apple's matrix unit, where threads do not add. On Linux and x86 the blocks of the compiled step are independent and can run on a core each. *Gain: the lead held on the machines most users have; timed there first, since I17's table is one Apple M4.*
- **Custom architectures.** A declarative network description (blocks, widths, activations, normalization, input embeddings, skip connections) that both the autograd path and the kernel run; activations beyond `tanh` (`sin` for SIREN, `gelu`, `silu`) by their Taylor recurrences; Fourier features. *Gain: a user's architecture is not a reason to fall back.*
- **Many networks in one call.** Ensembles, the swarm of `pinnrl.uq`, restarts and hyperparameter sweeps trained as a batch of networks through one threaded kernel call. *Gain: a calibrated ensemble for the price of a few single runs; `train_with_restarts` in parallel.*
- **Inference outside Python.** Export a trained twin to a dependency-free C source or shared library (values and derivatives), and to ONNX; a throughput benchmark (evaluations per second per core). *Gain: a soft sensor that runs in a PLC gateway, a historian calculation or a spreadsheet plug-in at plant rate.*
- **Prebuilt wheels.** Platform wheels with the kernels compiled (macOS arm64/x86, Linux x86/aarch64), so no compiler is needed. *Gain: `pip install pinnrl` is enough.*
- *Stretch:* a Metal/CUDA backend for large batches; the kernel is a CPU design and small networks are where it wins.

## Every release from v0.9: tests that protect the architecture

A release must not break what exists. The suite today has about 900 tests and covers 96 % of the lines it measures, but it measures neither the dashboard, the launcher nor the training command line, and line coverage says little about whether a new component is caught when it breaks a contract.

- **Contract test**: one test, written once, that every implementation of an extension point must pass (every equation, every network, every sampler, every process unit).
- **Integration test**: a test that runs several parts together, as a user does (train, save, reload, analyse, display).
- **Dead code**: code that no test, no entry point and no documented use reaches.

| Step | What is built | Done when |
|---|---|---|
| T1. Contracts for the extension points | A registry-driven suite: a new equation, architecture, sampler, optimizer or process unit is picked up automatically and must satisfy the same contract (shapes; the exact solution has zero residual; the compiled step equals autograd; save and reload give the same output) | Adding a component without touching the tests still tests it |
| T2. Engine equivalence by generation | Random equations (orders, mixed terms, fields, parameters, unknown functions) checked compiled step against autograd | A defect planted in the generator's output is caught |
| T3. Pipelines end to end | Train, checkpoint, reload, analyse, and read the run in the dashboard, for the forward, inverse and process paths, through the command line and the Python API | Each documented workflow is one test |
| T4. Compatibility | Run folders and checkpoints written by earlier releases still load | A fixture per released version |
| T5. Coverage that counts everything | The dashboard, the launcher and the training command line come back into the measurement | Coverage of the whole package stays above 75 % (the present gate of 90 % on the measured part stays) |
| T6. Tests judged by what they catch | Mutation testing on the core modules: a test that never fails when the code is altered is rewritten or removed | A mutation score is reported per module |
| T7. Dead code removed | Unreached code found by coverage and a static scan, then deleted or given a test | The scan is clean and runs in CI |

Rule for new tests: no assertion that restates the implementation. A test states a property a user relies on, or a contract a new component must meet.

## The dashboard as an experiment-tracking tool

Today the dashboard launches, monitors and compares runs. The aim is a tool a team can use to track experiments and follow their data from the record to the result.

- **Run**: one training or estimation, with its configuration, metrics, checkpoints and figures.
- **Lineage**: which data, which code version and which settings produced a result.
- **Artifact**: a file a run produced (a model, a figure, a table).

| Step | What a user gains | Done when |
|---|---|---|
| D1. A run store with a schema | Every run indexed: configuration, metrics, tags, code version, data fingerprint; search and filter over thousands of runs | A query over 5,000 runs returns in under a second |
| D2. Compare and diff | Curves overlaid, configurations diffed, error against wall-clock time, seeds grouped with median and spread | Two runs can be explained apart from one screen |
| D3. Lineage and artifacts | From a figure back to the run, the data and the commit; an artifact browser; export of a run as a report | Every number in a report links to its run |
| D4. Live operation | Launch, queue, stop and resume runs; live loss, residual maps and engine status; alerts when the misfit test fails | A study of 100 runs is managed without a terminal |
| D5. The process view, grown | Sensors, estimated states with intervals, fouling monitors and the line's variance budget, on recorded or replayed data | An engineer reads a shift from one page |
| D6. Design and reliability | A design system (layout, colour, type), keyboard access, tested callbacks, a separate data layer under the pages | The pages are covered by T3 and usable on a laptop screen |
| D7. Working with other tools | Import and export to MLflow and Weights & Biases formats | A run moves in and out without loss of its metrics |

## v0.10: `pinnrl.gym`, process environments and control

The simulators become environments with the Gymnasium interface, so any RL library can train on them, and the package ships the baselines that an RL result must beat.

- **The environments.** `Evaporator-v0` (actions: steam temperature, feed flow; hidden: fouling), `SprayDryer-v0` and `CounterCurrentDryer-v0` (actions: inlet air temperature, feed rate; constraint: the stickiness margin), `Line-v0` (the passes and the dryer together). Observations are the plant's sensors with their noise, delays and lab assays that arrive hourly, so the problems are partially observed as a plant is. Disturbances come from a model or from a historian export replayed. *Done when each environment passes Gymnasium's checker and a random policy runs at a stated number of steps per second.*
- **Tasks with money in them.** Setpoint tracking of the powder moisture; energy per kilogram of powder under a moisture specification and a stickiness limit (a constrained problem); **when to clean** (the fouling monitor's time-to-threshold turned into an optimal-stopping decision against production lost); start-up and product changeover. Each task has a reward in physical units and a documented constraint.
- **Throughput.** Vectorised environments on the compiled integrators; the dryer's steady and transient solves in C++ as the evaporator's are (I16). *Gain: millions of environment steps an hour on a laptop, which is what RL needs and what a CFD-based environment cannot give.*
- **Baselines first.** PID loops, linear MPC on the linearised model and nonlinear MPC through the reference solver, each with the EKF or MHE as its state estimator. *Rule, as for the estimators: an RL policy is reported against MPC with the same information, and where MPC wins the documentation says so.*
- **Agents.** Adapters for Stable-Baselines3 and CleanRL, and one reference PPO and SAC in the package for reproducibility; the existing DQN becomes one agent behind the same interface.
- **The twin in the loop.** Policies that act on the estimator's state and its interval (the EKF's fouling and band, the powder soft sensor), and training under the uncertainty the inverse tools quantify: domain randomisation over the equivalence region of `pinnrl.uq` instead of over arbitrary ranges. *Gain: a policy robust to exactly the model errors the data cannot rule out, the sim-to-real gap stated in the units of the identifiability analysis.*
- **Safe and offline RL.** Constrained policies for the stickiness margin (a shield from the twin's interval); offline RL from historian data with the simulator as the evaluation ground truth. *Gain: a path to a plant, where exploration on the equipment is not allowed.*
- **PINN world models.** Where an environment has no cheap solve, a parametric PINN as the differentiable model inside model-based RL, with policy gradients through the surrogate. *This is the PINN's case restated for control, and it is tested where it belongs: against the solver-based environment on a unit that has one.*

## v0.11: decisions that serve the inverse problem

The inverse problem contains decisions that are made greedily or by hand today. They become policies, each with its greedy baseline already in the package.

- **Sequential experiment design.** Which operating point to run next and when to pull a lab sample, to identify a closure or a fouling law fastest. Baseline: `analysis.greedy_design` (J09, J12), which is myopic and linear-Gaussian; candidates: Bayesian optimal design and an RL policy on the design environment. *Gain: a campaign plan for a pilot tower in fewer runs, with the expected interval before the first run is made.*
- **Sensor placement and maintenance.** Where to put a thermocouple, which instrument to recalibrate, as a budgeted decision under the same information currency.
- **A training gym (`PINNTraining-v0`).** The PINN's own training as an environment: actions are the sampling distribution, the loss weights, the learning rate and when to restart; the reward is the error gained per residual evaluation. The bandit, RAD, the adaptive weights and the schedules become policies in one interface, and classical bandits (UCB, Thompson, EXP3; C07) and a policy-gradient sampler (C09) join them. *Gain: the sampling study (B01–B08) becomes a leaderboard anyone can add to, and a learned training controller is tested for transfer across equations (C10).*
- **Acceptance and restarts as a policy.** `train_with_restarts` chooses the next seed blindly; a policy that reads the misfit and the loss curve chooses whether to continue, restart or switch optimizer.

## v0.12: discovering unknown physics

The package identifies parameters and free-form functions. The step beyond is to say what the function is, whether the model's form is wrong, and how the plant is being run.

- **From a learned closure to an equation.** Sparse regression of a fitted `UnknownFunction` on a library of candidate terms (the SINDy idea applied to the closure, not the whole equation), with the candidates' identifiability checked first and the selected form refitted through the reference solver and judged by the misfit test. *Gain: a drying-rate law an engineer can read and publish, with an interval on every coefficient.*
- **Model-form error, named.** A structured discrepancy term next to the known physics, constrained so it cannot absorb what the parameters explain (F08); a report that separates "the parameters are uncertain" from "the equations are wrong". Model selection between candidate closures by Laplace evidence and held-out misfit.
- **Hybrid models.** Known balances with learned terms where the physics is missing (wall deposition, agglomeration), using the anchor and monotonicity constraints so the learned part stays physical.
- **Units without a method-of-lines form.** A droplet size distribution (population balance) and wall deposition in the dryer, the first models where the PINN twin has no cheap classical rival. *This is the open item of J13: the place the PINN's case can still be made.*
- **Ways of working.** From historian data: operating regimes and transitions (changepoints and clustering on the twin's states, not on raw tags), the differences between shifts or recipes expressed as differences in estimated parameters, and drift that precedes a quality excursion. *Gain: the twin explains how the line is actually run, which is the question a plant asks first.*
- **Amortised identification.** A network trained on simulated records that maps a new record to parameters and intervals in one pass (simulation-based inference), checked for calibration as J06 checked the filters. *Gain: identification in milliseconds for a fleet of units.*

## v0.13: simulation at scale

- **Parametric surrogates.** One network across operating points, geometries and parameters (inputs beyond `x, t`), trained against the reference solver where one exists and by the residual where none does. *Gain: what-if studies and design sweeps without a solve per point, and the world model v0.10 needs.*
- **Hard constraints and decomposition.** Exact initial and boundary conditions by output transform (D09); domain decomposition and time marching for long horizons and sharp fronts; separable networks for 2-D and 3-D (E09).
- **A unit library with a plugin interface.** Wiped-film evaporators (P22), plate heat exchangers, membrane filtration, fluidised-bed drying, the cleaning cycle itself; a documented base class so a user's unit gets the reference solver, the PINN form, the estimators and a gym environment from one definition. *Gain: the method leaves food powders for pharmaceuticals, batteries and chemicals without a rewrite.*
- **Pretraining on The Well.** The dataset loader is in place; operator backbones pretrained on simulation data and fine-tuned with a residual, tested on whether pretraining buys accuracy at equal compute.

## v1.0: a stable platform

The 0.x releases may still move the API. 1.0 is the promise that they stop.

- **A stable API.** A deprecation policy, semantic versioning that means it, and one documented public surface for the trainer, the inverse tools, the process units and the gym.
- **The software paper (P06).** The package described and benchmarked in one citable place, with the head-to-head of v0.9 and the control baselines of v0.10.
- **Reproducibility.** Every figure in the documentation regenerated by one command from a pinned environment.

## v2.0 to v6.0: the platform before the plant

The plant is not next. A twin that advises a real line has to be right when nobody can check it, so the platform first has to be far more capable, and proven on simulated plants where the truth is known. The major versions are the stages of that; their content will be set as the 0.x work shows what holds up.

| Release | Stage | Done when |
|---|---|---|
| v2.0 | **Control that earns its place.** `pinnrl.gym` as a benchmark suite: tasks, baselines and a leaderboard for constrained, partially observed process control | A learned policy beats nonlinear MPC on at least one task at equal information, or the suite documents why none does |
| v3.0 | **Discovery as a workflow.** Records in; parameters, closures as equations, model-form error and operating regimes out, with intervals | The workflow recovers planted physics on every unit of the library, blind |
| v4.0 | **Scale.** Parametric and operator surrogates, 2-D and 3-D units, flowsheets of many units, a GPU backend where the batch justifies it | A whole line's what-if study runs interactively |
| v5.0 | **The autonomous twin.** A twin that identifies itself, tests its own misfit, redesigns its measurements, retrains and says when it should not be trusted, unattended | It runs for simulated months through drift, faults and recipe changes without a silent failure |
| v6.0 | **The plant rehearsal.** The whole stack against a simulated plant behind real interfaces (OPC UA, MQTT, a historian), with public pilot-scale data wherever it exists | Advice in shadow mode is scored against the simulated truth, and the failures are reported |

## v7.0: the plant

Deliberately far out: nothing here starts until v6.0's rehearsal has passed.

- **Live data.** OPC UA and MQTT connectors next to the historian adapter; the monitor as a service that runs for weeks, with alarms on the misfit test.
- **Advice with intervals (P25).** Setpoint recommendations and cleaning schedules from the controllers, shown in the dashboard with what they are expected to gain and how sure the twin is.
- **Validation on a real line (P26).** The simulated-plant results repeated on plant data, with the failures reported.

## Not planned

A hosted service and monetization (until the research results justify it); a custom PDE-builder UI; a general-purpose RL library (the agents come from the libraries that exist; `pinnrl` supplies the environments, the baselines and the twin); CFD-scale solvers.
