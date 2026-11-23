# Research plan: an optimizer made for physics-informed losses

**Aim.** The error a PINN reaches per second on one CPU core, far below what Adam and L-BFGS give. Not one more variant of Newton's method: a set of moves that use what a physics-informed loss is, and a rule for choosing among them.

**Measure.** $\varepsilon(T) = \lVert u_\theta - u \rVert_2 / \lVert u \rVert_2$ after wall-clock time $T$. Ten seeds, median and spread, same machine and precision for every method.

## What a physics-informed loss gives an optimizer

$$
L(\theta) = \tfrac12 \lVert r(\theta) \rVert^2, \qquad r(\theta) = \big(\mathcal{N}[u_\theta](x_i),\; u_\theta(x_j) - g_j\big), \qquad J = \frac{\partial r}{\partial \theta}
$$

A generic optimizer sees a scalar and its gradient. Five facts are thrown away:

1. **It is a sum of squares**, and every row belongs to one point in space and time.
2. **The rows are infinite.** A fresh row costs one residual evaluation, anywhere, at any moment.
3. **$r$ comes from a differential operator $\mathcal{N}$** that classical numerics already knows how to invert, coarsely and cheaply.
4. **The unknown is a function** $u_\theta$, linear in the last layer. The weights are a coordinate system, not the object.
5. **Time is a coordinate.** Information flows forward in it; the problem at small $t$ does not depend on the one at large $t$.

Each idea below uses at least one of these. Some exist in the literature in other settings (marked); the plan is to test them on equal terms, keep what survives, and combine the survivors.

## The moves

### M1. Residual flow, integrated like an ODE (facts 1, 4)

Prescribe how the residual should die, then find the weights that make it so:

$$
\dot r = -r \;\;\Longrightarrow\;\; J\,\dot\theta = -r \;\;\Longrightarrow\;\; \dot\theta = -J^{+} r
$$

- **$J^{+}$**: the pseudo-inverse of $J$. Levenberg–Marquardt is one damped Euler step of this flow.
- **The idea**: integrate the flow with the tools of stiff ODEs instead of a trust region: an embedded pair for step control, an exponential or Rosenbrock step, a step size chosen from an error estimate.
- **Test**: against Nielsen's damping rule, same linear algebra. **Kill** if no gain in $\varepsilon(T)$ on two of three problems.

### M2. The classical solver as the preconditioner (fact 3)

Let a coarse finite-difference solve say where the solution should move, and let the network only represent it:

$$
e_k = \mathcal{A}_h^{-1}\, r_k, \qquad \theta_{k+1} = \arg\min_\theta \lVert u_\theta - (u_{\theta_k} - e_k) \rVert^2
$$

- **$\mathcal{A}_h$**: the equation linearized about $u_{\theta_k}$ on a coarse grid. Its inverse costs milliseconds.
- **Why**: gradient methods on PINNs are slow in proportion to the conditioning of the operator (up to the fourth power of the highest frequency for a second-order equation). $\mathcal{A}_h^{-1}$ removes exactly that factor. The fit that remains is a regression, which is easy.
- **Known**: operator preconditioning as an analysis; defect correction in numerical analysis. **Not found**: a classical coarse solve inside a PINN optimizer loop.
- **Test**: heat and Burgers. **Kill** if the coarse grid needed approaches the accuracy wanted, since then the solver alone is the answer.

### M3. Optimization as filtering (facts 1, 2, 5)

Never visit a point twice. Stream fresh rows, in time order, through a recursive estimator that carries the uncertainty of the weights:

$$
K_k = P_k J_k^\top (J_k P_k J_k^\top + R)^{-1}, \qquad \theta_{k+1} = \theta_k - K_k r_k, \qquad P_{k+1} = (I - K_k J_k) P_k + Q
$$

- **$P$**: the covariance of the weights, kept in low rank. **$K$**: the gain. **$Q$**: a forgetting term.
- **Why**: this is Gauss–Newton with memory, so a batch of 64 rows moves the weights as a batch of thousands would. It gives the step and a posterior on the weights in one object, which is the Laplace approximation the package computes afterwards today.
- **Known**: extended Kalman filter training of networks (1990s), recursive least squares. **Not found**: on PINN residual streams with causal ordering and a low-rank covariance.
- **Test**: rows per second and $\varepsilon(T)$ against M1 at equal time. **Kill** if the covariance needs full rank to work.

### M4. Sample and solve: the sampler is the optimizer (facts 1, 2)

Pick a small block of rows and satisfy it exactly:

$$
\theta_{k+1} = \theta_k - J_B^\top (J_B J_B^\top + \lambda I)^{-1} r_B, \qquad B \sim p(\text{rows})
$$

- **$B$**: a block of tens of rows. The solve is a tiny dense system.
- **Why**: randomized block Kaczmarz converges at a rate set by the block choice. Choosing $B$ by residual, by leverage or by a learned policy makes the sampling decision the optimization decision, which is where a reinforcement-learning policy has something real to control.
- **Known**: sketch-and-project, Kaczmarz. **Not found**: as the main PINN optimizer with adaptive blocks.
- **Test**: block size 16 to 256 against M1. **Kill** if it never beats a full Gauss–Newton step per second.

### M5. Solve what is linear, optimize the rest (fact 4)

$$
u_\theta = \sum_j c_j\, \phi_j(x; \vartheta), \qquad c^\star(\vartheta) = \arg\min_c \lVert r(c, \vartheta) \rVert^2, \qquad \min_\vartheta \lVert r(c^\star(\vartheta), \vartheta) \rVert^2
$$

- **Variable projection**: the last layer is solved at every step and never trained.
- **Beyond it**: grow the basis. Add the neuron that best correlates with the current residual (a matching-pursuit step), so the architecture is an output of the optimizer.
- **Known**: variable projection; greedy and Galerkin neural networks. **Not found**: combined with M1 on trained hidden weights in a compiled engine.
- **Test**: heat first (linear), then Burgers. **Kill** if no gain on the linear problem.

### M6. Follow the solution, do not search for it (facts 3, 5)

Start from a problem that is easy and move the physics, predicting each step from the last:

$$
\frac{d\theta}{d\mu} = -\big(J^\top J\big)^{-1} J^\top \frac{\partial r}{\partial \mu}, \qquad \mu:\; \text{easy} \to \text{target}
$$

- **$\mu$**: a physical parameter (viscosity, interface width, the end time). The formula is the implicit-function tangent the package already computes for sensitivities.
- **Why**: Burgers at small viscosity and Allen–Cahn fail from a cold start and are easy from a neighbour. Predictor and corrector along $\mu$ never leaves the basin.
- **Known**: numerical continuation; curriculum training of PINNs. **Not found**: with an exact tangent predictor and step control.
- **Test**: Burgers from $\nu = 0.1$ to $0.01/\pi$; Allen–Cahn. **Kill** if a cold Gauss–Newton run is as good.

### M7. Let the equation's own time do the training (facts 4, 5)

For an evolution equation $u_t = F[u]$, do not minimize over the space–time box at all. Evolve the weights:

$$
M(\theta)\,\dot\theta = f(\theta), \qquad M = J_u^\top J_u, \quad f = J_u^\top F[u_\theta], \quad J_u = \frac{\partial u_\theta}{\partial \theta}
$$

- **Why**: the training loop becomes an ODE in the weights, integrated once with step control. Causality holds by construction.
- **Known**: neural Galerkin schemes and evolutional networks. **Not found**: on a compiled CPU engine against the same equation's space–time PINN at equal time.
- **Test**: heat, Burgers, KdV. **Kill** if the weight ODE is too stiff to integrate faster than M1 trains.

### M8. Constraints as multipliers, not weights (fact 1)

$$
\min_\theta \max_\lambda \;\tfrac12 \lVert r_{\text{pde}} \rVert^2 + \lambda^\top c(\theta) + \tfrac{\rho}{2} \lVert c(\theta) \rVert^2, \qquad c = \text{initial and boundary rows}
$$

- **Why**: the loss weights 10 and 10 are guesses. Multipliers find the balance themselves and remove a tuning knob.
- **Known**: augmented-Lagrangian PINNs. **Test**: inside M1's linear algebra, where the saddle-point system is a small change. **Kill** if hard constraints (roadmap R3) make it unnecessary.

### M9. An agent that chooses the move (all five)

The moves above cost different amounts and help at different moments. Choosing is a sequential decision:

$$
a_k \in \{\text{M1 step},\ \text{M3 stream},\ \text{M4 block},\ \text{M5 solve},\ \text{grow},\ \text{M6 advance},\ \text{restart}\}, \qquad \text{reward} = \frac{\log \varepsilon_k - \log \varepsilon_{k+1}}{\text{seconds spent}}
$$

- **State**: the residual's size and its spectrum by frequency, the gain ratio of the last step, the cost of each move so far.
- **First version**: a bandit over the moves. **Then**: a policy trained across equations and tested on equations it has not seen.
- **Baseline**: the best single move, and a fixed schedule.

## Order of work and gates

| Stage | Work | Gate to pass |
|---|---|---|
| 0 | Levenberg–Marquardt with dense $J$ in PyTorch (`run.py`), against Adam, L-BFGS and their sequence | It reaches the published Gauss–Newton errors (about $10^{-6}$ on heat) on one core |
| 1 | M1, M4, M5 as variants of stage 0 in the same script | Each beats stage 0 on two of three problems, or is dropped |
| 2 | M6 first, then M5b (the last-layer solve on the equation linearized about the current solution), M3, M8 | The same |
| 3 | M2, M7 (each needs a second code path) | The same, and against the classical solver where one exists |
| 4 | The survivors in the compiled engine: $Jv$ and $J^\top w$ as generated passes, single precision | The prototype's $\varepsilon(T)$ at a fraction of the time |
| 5 | M9, then the nine equations of I17 and the public suites | Roadmap bets R4 and R5 |

Rules: every arm is charged its full wall-clock time; nothing is tuned per problem that a user would not tune; a dropped idea is written up with its numbers.

## Results

Stages 0 to 2 are measured in [README.md](README.md) (heat, Burgers, KdV; 60 s and 180 s; 3 seeds).

| Move | Outcome |
|---|---|
| Stage 0 | Passed: $6 \times 10^{-7}$ on heat, 120 and 220 times below Adam then L-BFGS on heat and KdV, 3 times on Burgers |
| M1 | Dropped. The flow with a fixed cut-off does not move ($J$ has numerical rank 93 of 2,241); with the damping set from the linearization error it equals Levenberg–Marquardt |
| M4 | Dropped as the main optimizer: the fastest method in the first seconds on KdV, then it stalls. Kept as an opening move for M9 |
| M5 | Kept: $5 \times 10^{-9}$ on heat, 127 times below stage 0. Level with it on Burgers and KdV, and only after a warm start |
| M5b | No arm needed: it is the Gauss–Newton sweep on the last layer that M5 already runs |
| M3 | **Kept, the lead candidate.** At 180 s: $2 \times 10^{-5}$ on Burgers (65 times below stage 0), $1.5 \times 10^{-9}$ on heat (73 times), $10^{-7}$ on KdV (10 times). Slower than stage 0 for the first minute. Needs a cap on the covariance and a check on each update, or it diverges |
| M3, hybrid | Kept: 20 s of stage 0 first gives stage 0's error at 30 s and the filter's at 180 s. The restart of $P$ costs about 40 s of flat error |
| M3, covariance per layer | Kept with a caveat: level on heat, 24 times better on KdV, 20 times worse on Burgers than the full covariance |
| M3, covariance per neuron | Dropped: behind stage 0 on heat and KdV, diverges on Burgers |
| M3, forgetting 0.95 | Kept: ahead of stage 0 from the first reading on heat and KdV (280 and 55 times at 30 s), and 7 and 5 times below forgetting 0.995 at the end. On Burgers the long memory ends 3 times lower |
| M3, low rank (64 directions) | Kept for size: level with stage 0 at 12,737 weights and 600 s, 17 times ahead on heat at 60 s; stops at stage 0's level where the full filter goes 10 to 70 times lower |
| M6 | Dropped at this viscosity: no better than a cold start with 4, 8 or 16 stages |
| M8 | Dropped: worse than the same penalty without multipliers. Loss weights of 1 beat the weights of 10 used so far |

What the results change:

1. **The filter is the method to build on.** It is the only move ahead of Levenberg–Marquardt on all three equations, and the only one that moves Burgers.
2. **Its two weaknesses set the next work.**
   - **The slow first minute.** Open with Levenberg–Marquardt or blocks (M4), then hand the weights to the filter. The switch is a decision for M9.
   - **The full covariance.** $P$ costs the square of the number of weights. Test a low-rank plus diagonal form, a Kronecker-factored form per layer, and a square-root update, at 2,241 weights first and then at the 12,000 of I17's network.
3. **Combine the two survivors.** M5 solves the last layer exactly; M3 filters the rest. On heat they reach the same error by different routes.
4. **Order the stream in time** (fact 5). The filter here draws points uniformly; the plan's causal ordering is untested.
5. **A harder equation.** Continuation was not needed at $\nu = 0.01/\pi$. Allen–Cahn and Burgers at a smaller viscosity are where it, and the filter, should be tried next.
6. **Baselines to add**: SOAP and self-scaled Broyden, and every baseline at 180 s.
7. **Basis growth** (the second half of M5) is still untested.

Stage 2b, measured so far (item 2):

- **The slow first minute is solved by a hybrid**, at the price of a flat stretch after the switch. Starting $P$ at $(J^\top J + I)^{-1}$ on a full sample at the switch was tried and is no better, slightly worse: dropped.
- **The covariance can lose the coupling between layers on smooth solutions, not on a front.** Coupling within a neuron alone is not enough. At I17's 4×64 network (12,737 weights) the per-layer form is too slow (40 to 50 times behind stage 0). A low-rank information form with 64 directions keeps up with stage 0 but loses the full filter's late gains: the long memory lives in the weak directions it drops. Keeping the dropped information on a diagonal and 256 directions in place of 64 were tried (one seed) and do not bring it back.
- **The forgetting factor is the filter's pace.** 0.95 in place of 0.995 removes the slow first minute on heat and KdV; 0.999 is 10,000 times worse on heat. Burgers prefers 0.995 late. Next: a forgetting that rises along the run, or is set from the innovation $r^\top (J P J^\top + I)^{-1} r$ (adaptive forgetting, as in recursive least squares), and forgetting 0.95 at 12,737 weights.

- **Against Adam then L-BFGS at 180 s** the filter is 35,000 (heat), 350 (Burgers) and 9,800 (KdV) times lower: the 100 times of bet R1 on all three, at this size.

Still open in stage 2b: items 3, 4, 5 and the rest of 6 (SOAP, self-scaled Broyden). Stage 3 (M2, M7) and the engine work of stage 4 follow as planned.
