# Concepts

This page defines the ideas behind each component of `pinnrl`, one equation and one line at a time. Read it if you are new to physics-informed neural networks (PINNs); derivations are in the [cited papers](#references).

## The PINN loss

A PINN is a network $u_\theta(x, t)$ trained to satisfy a differential equation and its conditions (Raissi, Perdikaris & Karniadakis, 2019):

$$
\mathcal{N}[u] = 0, \qquad r_\theta(x, t) = \mathcal{N}[u_\theta](x, t)
$$

- **$u_\theta$**: the network; its weights $\theta$ are the unknowns.
- **$\mathcal{N}$**: the differential operator of the equation, for example $\mathcal{N}[u] = u_t - \alpha u_{xx}$.
- **Residual $r_\theta$**: what is left when the network is put into the equation. It is zero for an exact solution.
- **Collocation point**: a point $(x_j, t_j)$ inside the domain where the residual is evaluated.

Training minimizes a weighted sum of four terms by gradient descent:

$$
\mathcal{L}(\theta) = w_r\,\mathcal{L}_r + w_b\,\mathcal{L}_b + w_i\,\mathcal{L}_i + w_d\,\mathcal{L}_d
$$

| Term | Definition | Evaluated at |
|---|---|---|
| Residual | $\mathcal{L}_r = \operatorname{mean}_j \lvert r_\theta(x_j, t_j) \rvert^2$ | collocation points |
| Boundary | $\mathcal{L}_b = \operatorname{mean} \lvert \mathcal{B}[u_\theta] - g \rvert^2$ | every condition $\mathcal{B}[u] = g$ on every boundary face |
| Initial | $\mathcal{L}_i = \operatorname{mean} \lvert u_\theta(x, t_0) - u_0(x) \rvert^2$ | $t = t_0$ |
| Data | $\mathcal{L}_d = \operatorname{mean} \lvert u_\theta(x_\text{obs}, t_\text{obs}) - u_\text{obs} \rvert^2$ | observations; zero unless data is attached (inverse and data modes) |

- **Loss weights $w_r, w_b, w_i, w_d$**: fixed numbers under `loss_weights` (15, 20, 10, 10 in `config.yaml`), or [adapted during training](#loss-weighting).
- **Mean squared error**: the form of each term with the default `loss_function: mse`.
- **Mesh-free**: nothing ties the network to a grid, so the same code handles irregular domains and higher dimensions.

Accuracy depends on three knobs, covered below:

1. Where the residual is evaluated (sampling).
2. How the terms are balanced (weighting).
3. What the network can represent (architecture).

### Automatic differentiation

The derivatives in $\mathcal{N}[u_\theta]$ come from the chain rule applied to the network itself:

$$
u_x = \frac{\partial u_\theta}{\partial x}, \qquad u_{xx} = \frac{\partial}{\partial x}\!\left(\frac{\partial u_\theta}{\partial x}\right)
$$

- **Automatic differentiation**: exact derivatives by the chain rule, with no finite-difference step.
- **Engine**: the code that computes them. The default, `derivatives="auto"` in `build_trainer` and `pinnrl-train`, is a compiled kernel wherever it applies; `"autograd"` switches it off. See [The native engine](engine.md).

### Forward and inverse problems

$$
\text{forward: } \min_\theta \mathcal{L}(\theta; \lambda) \qquad\qquad \text{inverse: } \min_{\theta,\,\lambda} \mathcal{L}(\theta, \lambda)
$$

- **$\lambda$**: the physical parameters of the equation, such as a diffusivity.
- **Forward problem**: $\lambda$ is known; find $u$.
- **Inverse problem**: $\lambda$ is unknown; find it and $u$ from measurements ([below](#inverse-problems)).

## Boundary and initial conditions

| Kind | Enforces | Config |
|---|---|---|
| Dirichlet | $u = g$ on each face | `dirichlet: {value: 0}`, `{left: a, right: b}`, or `{type: exact}` (uses the exact solution) |
| Neumann | $\partial u/\partial x_i = g$ on both faces of axis $i$ (the axis derivative, not the outward normal, so a nonzero $g$ has the same sign on both faces) | `neumann: {value: 0}` or `{type: zero}` |
| Periodic | $u$ and $\partial u/\partial x_i$ match across opposite faces | `periodic: {}` |

- Every spatial axis gets the condition on both faces.
- **Second order in time** (wave, pendulum): the initial loss adds the initial velocity,

$$
\mathcal{L}_i = \operatorname{mean}\lvert u_\theta(x, t_0) - u_0 \rvert^2 + \operatorname{mean}\lvert \partial_t u_\theta(x, t_0) - v_0 \rvert^2
$$

- **$v_0$**: `initial_condition.velocity`, default 0. Without it the problem is not well posed.
- An unknown condition type raises an error; it is not silently ignored.

## Collocation sampling

The residual is only enforced where it is evaluated:

$$
\mathcal{L}_r = \frac{1}{N} \sum_{j=1}^{N} \lvert r_\theta(x_j, t_j) \rvert^2, \qquad (x_j, t_j) \sim p
$$

- **Sampler**: the rule $p$ that draws the $N$ collocation points of a batch.
- **Sharp feature**: a shock, interface or boundary layer. It fills a small part of the domain, so uniform points can under-resolve the place where the error is.

| Strategy | Rule | Idea | Cost |
|---|---|---|---|
| `uniform` | $z_j = z_j^\text{grid} + 0.01\,L\,\xi_j$ | A jittered space-time grid, redrawn every batch. Unbiased, and implicitly regularizing. | none |
| `stratified` | $z_j = a + (\pi(j) + U_j)\,\dfrac{b - a}{N}$ | Latin-hypercube points: one per stratum, so coverage is even. | none |
| `residual_based` (**RAD**) | $p(x) \propto \dfrac{\lvert r(x)\rvert^k}{\operatorname{mean}\lvert r\rvert^k} + c$ | Draw from a uniform candidate pool with probability $p$ (Wu et al., 2023). | one residual evaluation per pool point |
| `adaptive` (**RL bandit**, experimental) | $\pi = (1 - \epsilon)\operatorname{softmax}(Q / T) + \dfrac{\epsilon}{N_\text{cells}}$ | A bandit learns which cells have high residual, and mixes its picks 50/50 with uniform points. 1-D only. | 256 residual evaluations (32 cells × 8 points) per update, every `probe_every` batches (10 in `config.yaml`) |

- **Jitter**: Gaussian noise $\xi_j \sim \mathcal{N}(0, 1)$ scaled to 1 % of the axis length $L$, added to a grid node.
- **Stratum**: one of $N$ equal slices of an axis $[a, b]$; $\pi$ is a random permutation and $U_j$ a uniform draw in $[0, 1)$.
- **Pool**: the RAD candidates, `pool_factor` = 4 times the batch.
- **$k$, $c$**: $k$ sharpens the focus on high residuals; $c$ keeps a uniform floor. The trainer uses $k = 1$, $c = 0$; other values are arguments of `generate_collocation_points` and of the sampling study.
- **Cell**: one of 16 × 8 space-time boxes; each is one action of the bandit.
- **Contextual bandit**: a DQN with discount 0. Its value $Q$ regresses onto the reward, a cell's mean $\lvert r \rvert$ from probes relative to the other probed cells.

Adaptive sampling is not free: it has to pay for its overhead and beat a strong uniform baseline at equal budget. [Benchmarks](benchmarks.md) reports how the strategies compare.

## Architectures

A plain network learns the smooth part of a solution first:

$$
u(z) = \sum_k a_k\, e^{\mathrm{i} k \cdot z}: \quad \text{small } \lvert k \rvert \text{ is fitted first, large } \lvert k \rvert \text{ last}
$$

- **MLP**: multilayer perceptron, a stack of layers $h \mapsto \sigma(W h + b)$ on the input $z = (x, t)$.
- **Spectral bias**: MLPs learn low-frequency components first and struggle with high-frequency or multiscale solutions (Rahaman et al., 2019). Most of the alternatives exist to counter it.

| Key | Layer | Idea | Use it for |
|---|---|---|---|
| `feedforward` | $\tanh(W h + b)$ | Tanh MLP, the standard PINN baseline | smooth solutions; the first thing to try |
| `resnet` | $\sigma\big(h + F(h)\big)$ | MLP blocks $F$ with skip connections | deeper networks that still train stably |
| `fourier` | $[\sin(zB), \cos(zB)]$ | Random Fourier features before the MLP (Tancik et al., 2020); one hidden layer per `hidden_dims` entry | oscillatory or high-frequency solutions |
| `siren` | $\sin\big(\omega_0 (W h + b)\big)$ | Sine activations with frequency `omega_0` (Sitzmann et al., 2020) | wave-like solutions; sensitive to `omega_0` |
| `fno` | $\sigma\big(\operatorname{LN}(K h + W h + h)\big)$ | Fourier-neural-operator-style spectral layers $K$ on point-wise inputs (Li et al., 2021) | experimentation |
| `attention` | $\operatorname{LN}\big(h + A(h)\big)$ | Self-attention blocks $A$ over a learned embedding | experimentation |
| `autoencoder` | $z \to \text{latent} \to u$ | An encoder–latent–decoder MLP | experimentation |

- **Skip connection**: the input $h$ of a block added back to its output.
- **$B$**: a fixed random matrix, $B_{ij} \sim \mathcal{N}(0, \sigma^2)$; $\sigma$ = `scale` sets the frequency band.
- **LN**: layer normalization.
- **`fno` is not operator learning**: the FFT runs over each point's hidden features, and the network maps $(x, t)$ to $u$, not functions to functions.

## Optimizers

$$
\theta \leftarrow \theta - \eta\, \nabla_\theta \mathcal{L}(\theta)
$$

- **Learning rate $\eta$**: the step size. Schedules: `cosine`, `reduce_lr` (on plateau) or `constant`, with optional linear warmup.
- **Adam**: the robust default.
- **L-BFGS**: quasi-Newton with a strong-Wolfe line search. It often sharpens the final convergence of smooth problems but can stall early. Each step works on one sampled batch, so it behaves best with a single batch per epoch (`batch_size` = `num_collocation_points`).
- **`adam_lbfgs`**: Adam for a fraction of the epochs (`adam_lbfgs_switch_ratio`), then L-BFGS. A common recipe in the PINN literature.

## Loss weighting

The residual, boundary and initial terms can differ by orders of magnitude in value and in gradient size, and gradient descent then ignores the small ones. Fixed weights (`loss_weights`) are the simplest fix. Two strategies rebalance during training.

**`lrw`: gradient-norm balancing**

$$
\bar g_i \leftarrow \alpha\, \bar g_i + (1 - \alpha)\, \lVert \nabla_\theta \mathcal{L}_i \rVert, \qquad w_i = \frac{1/(\bar g_i + \epsilon)}{\sum_j 1/(\bar g_j + \epsilon)}
$$

- **$\bar g_i$**: a running average of the gradient norm of term $i$, with retention factor $\alpha$ = `alpha`.
- The weights sum to 1, so no term dominates the update.
- This is a simplified form of the learning-rate annealing of Wang, Teng & Perdikaris (2021), which scales each term by the ratio of the residual's maximum gradient to that term's mean gradient.

**`rbw`: relative loss balancing**

$$
\bar L_i \leftarrow \alpha\, \bar L_i + (1 - \alpha)\, \mathcal{L}_i, \qquad w_i \leftarrow \alpha\, w_i + (1 - \alpha)\, \frac{\bar L_i}{\sum_j \bar L_j + \epsilon}
$$

- **$\bar L_i$**: a running average of the loss of term $i$.
- The weights sum to 1 and are smoothed again over steps. The terms with the largest loss get the most weight.

Both strategies:

- replace the `loss_weights` of the residual, boundary and initial terms; the data weight stays fixed;
- are switched off during L-BFGS, where the three terms then have weight 1.

**Loss function** (`loss_function`), for an error $e$:

| Value | Term |
|---|---|
| `mse` (default) | $\operatorname{mean}\, e^2$ |
| `mae` | $\operatorname{mean}\, \lvert e \rvert$ |
| `huber` | $e^2/2$ for $\lvert e \rvert \le \delta$, else $\delta\,(\lvert e \rvert - \delta/2)$, with $\delta$ = `huber_delta` |

- **Robust loss**: `mae` and `huber` reduce the influence of outliers, for example noisy observations.

## Inverse problems

$$
\min_{\theta,\,\lambda}\; w_r\,\mathcal{L}_r(\theta, \lambda) + w_b\,\mathcal{L}_b + w_i\,\mathcal{L}_i + w_d \operatorname{mean} \lvert u_\theta(x_\text{obs}, t_\text{obs}) - u_\text{obs} \rvert^2
$$

1. Mark the PDE parameters $\lambda$ as trainable. They become `torch.nn.Parameter`s.
2. Supply observations. They add the data term.
3. Train. The network and the parameters are optimized together.

- **Physics terms**: pin $\lambda$ to values under which the network solves the PDE. $\lambda$ enters through the residual and, for some initial conditions such as the Allen–Cahn and KdV profiles, through the initial loss.
- **Data term**: pulls the solution toward the measurements.

See [Inverse problems](inverse_problems.md).

## Training modes

| `mode` | Loss terms | Use |
|---|---|---|
| `forward` | residual + BC + IC | Solve a known PDE |
| `inverse` | residual + BC + IC + data, with trainable parameters | Identify parameters from data |
| `data_augmented` | residual + BC + IC + data | Fit reference data while respecting the physics |
| `data_only` | data | A neural surrogate of a dataset (e.g. The Well) |

- **BC, IC**: boundary and initial conditions.

## Code map

| Module | Role |
|---|---|
| `pinnrl.pdes` | `PDEBase` (derivatives, BC/IC, sampling, loss, validation), the ten PDE configurations, and the registry (`create_pde`, `get_pde_spec`) |
| `pinnrl.neural_networks` | `PINNModel` and the seven architectures |
| `pinnrl.training` | `PDETrainer` (`train`, `train_steps`), adaptive weights, and the `pinnrl-train` CLI (`build_model`, `load_run`) |
| `pinnrl.rl` | The experimental residual-bandit sampler |
| `pinnrl.benchmarks` | Finite-difference baselines, numerical reference solutions, the sampling study, and the `pinnrl-benchmark` CLI |
| `pinnrl.datasets` | The Well registry and loader |
| `pinnrl.plots`, `pinnrl.dashboard` | Plotting and the Dash app |

## References

- Raissi, Perdikaris & Karniadakis (2019). Physics-informed neural networks: a deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *J. Comput. Phys.* 378, 686–707.
- Wu, Zhu, Tan, Kartha & Lu (2023). A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks. *Comput. Methods Appl. Mech. Eng.* 403, 115671.
- Wang, Teng & Perdikaris (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5), A3055–A3081.
- Rahaman et al. (2019). On the spectral bias of neural networks. *ICML*.
- Tancik et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS*.
- Sitzmann et al. (2020). Implicit neural representations with periodic activation functions. *NeurIPS*.
- Li et al. (2021). Fourier neural operator for parametric partial differential equations. *ICLR*.
- Krishnapriyan, Gholami, Zhe, Kirby & Mahoney (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*.
- Basdevant et al. (1986). Spectral and finite difference solutions of the Burgers equation. *Comput. Fluids* 14(1), 23–41. (Source of the Burgers reference value used in `pinnrl.benchmarks.reference`.)
