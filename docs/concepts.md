# Concepts

This page covers the ideas behind each component of `pinnrl`: what it does and why it is there. Derivations are in the cited papers.

## The PINN loss

A network $u_\theta(x, t)$ is trained to satisfy the PDE $\mathcal{N}[u] = 0$ together with its boundary and initial conditions (Raissi, Perdikaris & Karniadakis, 2019). Derivatives are computed exactly by automatic differentiation, and the loss is a weighted sum:

$$
\mathcal{L}(\theta) = w_r \underbrace{\operatorname{mean}\,\lvert \mathcal{N}[u_\theta] \rvert^2}_{\text{collocation points}}
+ w_b\,\mathcal{L}_\text{boundary} + w_i\,\mathcal{L}_\text{initial} + w_d\,\mathcal{L}_\text{data}
$$

$\mathcal{L}_\text{boundary}$ averages over every condition on every boundary face, $\mathcal{L}_\text{initial}$ is taken at $t = t_0$, and $\mathcal{L}_\text{data}$ is zero unless observations are attached (inverse and data modes). Each term is a mean squared error with the default `loss_function: mse`.

Training is ordinary gradient descent on $\mathcal{L}$. Because nothing ties the network to a mesh, the same code handles irregular domains and higher dimensions. The flip side is that accuracy depends on where the residual is evaluated (sampling), how the terms are balanced (weighting), and what the network can represent (architecture). Those are the three knobs below.

## Boundary and initial conditions

| Kind | Enforces | Config |
|---|---|---|
| Dirichlet | $u = g$ on each face | `dirichlet: {value: 0}`, `{left: a, right: b}`, or `{type: exact}` (uses the exact solution) |
| Neumann | $\partial u/\partial x_i = g$ on both faces of axis $i$ (the axis derivative, not the outward normal, so a nonzero $g$ has the same sign on both faces) | `neumann: {value: 0}` or `{type: zero}` |
| Periodic | $u$ and $\partial u/\partial x_i$ match across opposite faces | `periodic: {}` |

Every spatial axis gets the condition on both faces. For equations that are second order in time (wave, pendulum), the initial loss also enforces the initial velocity $u_t(x, t_0)$ (`initial_condition.velocity`, default 0). Without it the problem is not well posed. Unknown condition types raise an error rather than being silently ignored.

## Collocation sampling

The residual is only enforced where it is evaluated. Sharp features such as shocks, interfaces and boundary layers occupy a small part of the domain, so uniform points can under-resolve exactly where the error lives.

| Strategy | Idea | Cost |
|---|---|---|
| `uniform` | A jittered space-time grid, redrawn every batch. Unbiased, and implicitly regularizing. | none |
| `stratified` | Latin-hypercube points: one per stratum, so coverage is even. | none |
| `residual_based` (**RAD**) | Draw from a uniform candidate pool (`pool_factor` = 4 times the batch) with probability $p(x) \propto \lvert r(x)\rvert^k / \operatorname{mean}\lvert r\rvert^k + c$ (Wu et al., 2023). $k$ sharpens the focus; $c$ keeps a uniform floor. The trainer uses $k = 1$, $c = 0$; other values are arguments of `generate_collocation_points` and of the sampling study. | one residual evaluation per pool point |
| `adaptive` (**RL bandit**, experimental) | A DQN with discount 0, i.e. a contextual bandit, learns which of 16 × 8 space-time cells have high residual from probes, and mixes its picks 50/50 with uniform points. 1-D only. | 256 residual evaluations (32 cells × 8 points) per update, every `probe_every` batches (10 in `config.yaml`) |

Adaptive sampling is not free: it has to pay for its overhead and beat a strong uniform baseline at equal budget. [Benchmarks](benchmarks.md) reports how the strategies compare.

## Architectures

Plain MLPs suffer from *spectral bias*: they learn low-frequency components first and struggle with high-frequency or multiscale solutions (Rahaman et al., 2019). Most of the alternatives exist to counter it.

| Key | Idea | Use it for |
|---|---|---|
| `feedforward` | Tanh MLP, the standard PINN baseline | smooth solutions; the first thing to try |
| `resnet` | MLP blocks with skip connections | deeper networks that still train stably |
| `fourier` | Random Fourier features $[\sin(Bz), \cos(Bz)]$ of $z = (x, t)$ before the MLP, with a fixed $B_{ij} \sim \mathcal{N}(0, \sigma^2)$ and $\sigma$ = `scale`, which sets the frequency band (Tancik et al., 2020); one hidden layer per `hidden_dims` entry | oscillatory or high-frequency solutions |
| `siren` | Layers $\sin\big(\omega_0 (W z + b)\big)$ with frequency `omega_0` (Sitzmann et al., 2020) | wave-like solutions; sensitive to `omega_0` |
| `fno` | Fourier-neural-operator-style spectral layers applied to point-wise inputs (Li et al., 2021): the FFT runs over each point's hidden features. This is not operator learning: it maps $(x, t)$ to $u$, not functions to functions. | experimentation |
| `attention` | Self-attention blocks over a learned embedding | experimentation |
| `autoencoder` | An encoder–latent–decoder MLP | experimentation |

## Optimizers

- **Adam** is the robust default.
- **L-BFGS** is quasi-Newton with a strong-Wolfe line search. Each step works on one sampled batch, so it behaves best with a single batch per epoch (`batch_size` = `num_collocation_points`). It often sharpens the final convergence of smooth problems but can stall early.
- **`adam_lbfgs`** runs Adam for a fraction of the epochs (`adam_lbfgs_switch_ratio`) and then switches to L-BFGS, a common recipe in the PINN literature.
- Learning-rate schedules: `cosine`, `reduce_lr` (on plateau) or `constant`, with optional linear warmup.

## Loss weighting

The residual, boundary and initial terms can differ by orders of magnitude in value and in gradient size. When they do, gradient descent effectively ignores the small ones. Fixed weights (`loss_weights`) are the simplest fix. The adaptive strategies rebalance during training:

- **`lrw`**: gradient-norm balancing. With $\bar g_i$ a running average (factor `alpha`) of $\lVert \nabla_\theta \mathcal{L}_i \rVert$, the weights are $w_i \propto 1/(\bar g_i + \epsilon)$, normalized to sum to 1, so no term dominates the update. This is a simplified form of the learning-rate annealing of Wang, Teng & Perdikaris (2021), which scales each term by the ratio of the residual's maximum gradient to that term's mean gradient.
- **`rbw`**: relative loss balancing. The weights are proportional to a running average of each term's loss, normalized to sum to 1 and smoothed again over steps, so the terms with the largest loss get the most weight.

Both replace the `loss_weights` of the residual, boundary and initial terms; the data weight stays fixed. They are switched off during L-BFGS, where the three terms then have weight 1.

The `loss_function` setting can be `mse` (default), `mae` or `huber`. The robust losses reduce the influence of outliers, for example noisy observations.

## Inverse problems

Mark PDE parameters as trainable and supply observations. The parameters become `torch.nn.Parameter`s, and a data term $\mathcal{L}_\text{data} = \operatorname{mean} \lvert u_\theta(x_\text{obs}, t_\text{obs}) - u_\text{obs} \rvert^2$ is added. The parameters enter through the residual (and, for some initial conditions such as the Allen–Cahn and KdV profiles, through the initial loss), so the physics terms pin them to values under which the network solves the PDE, while the data term pulls the solution toward the measurements. Both are optimized together. See [Inverse problems](inverse_problems.md).

## Training modes

| `mode` | Loss terms | Use |
|---|---|---|
| `forward` | residual + BC + IC | Solve a known PDE |
| `inverse` | residual + BC + IC + data, with trainable parameters | Identify parameters from data |
| `data_augmented` | residual + BC + IC + data | Fit reference data while respecting the physics |
| `data_only` | data | A neural surrogate of a dataset (e.g. The Well) |

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
