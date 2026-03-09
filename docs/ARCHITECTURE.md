# Architecture Overview

## High-level system diagram

```
┌──────────────┐
│  config.yaml │  ← Device, architecture hyperparameters, PDE parameters,
└──────┬───────┘    training settings, RL configuration
       │
       ▼
┌──────────────┐      ┌─────────────────────┐
│   PDEConfig  │─────▶│   PDE Instance       │  HeatEquation, BurgersEquation, ...
└──────────────┘      │  (PDEBase subclass)  │  Owns: domain, BCs, ICs, exact solution,
                      └──────────┬──────────┘  residual computation via autograd
                                 │
┌──────────────┐                 │
│  ModelConfig │──┐              │
└──────────────┘  │              │
                  ▼              ▼
              ┌──────────────────────────────┐
              │          PINNModel           │  Wraps one of six architectures:
              │  (neural_networks/__init__)  │  FeedForward / ResNet / SIREN /
              └──────────────┬───────────────┘  Fourier / Attention / Autoencoder
                             │
              ┌──────────────▼───────────────┐
              │          PDETrainer          │  Training loop, loss aggregation,
              │        (trainer.py)          │  LR scheduler, early stopping,
              └──────────────┬───────────────┘  visualization, metrics saving
                             │
              ┌──────────────▼───────────────┐     ┌──────────────────────┐
              │       CollocationRLAgent     │◀────│  Residual reward     │
              │         (rl_agent.py)        │     │  from PDE evaluation │
              │    DQN-based point selector  │     └──────────────────────┘
              └──────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────────┐
              │       experiments/           │  config snapshot, loss curves,
              │  Timestamped output dir      │  solution plots, metrics.json
              └──────────────────────────────┘
```

---

## The PINN training loop

Physics-Informed Neural Networks embed physical laws directly into the loss function. The network `u_θ(x, t)` is trained to minimize a composite loss that penalizes violations of the PDE, its boundary conditions, and its initial condition.

### Loss components

**PDE residual loss** — evaluated at randomly sampled interior collocation points `{(x_i, t_i)}`:

```
L_residual = (1/N_r) * Σ |F[u_θ](x_i, t_i)|²
```

where `F[u_θ]` is the differential operator applied to the network output. Derivatives are computed with PyTorch autograd (`torch.autograd.grad`) so the computation graph flows through the network.

For the heat equation: `F[u_θ] = ∂u_θ/∂t − α ∂²u_θ/∂x²`

**Boundary condition loss** — evaluated at points `{(x_b, t_b)}` on the domain boundary:

```
L_bc = (1/N_b) * Σ |u_θ(x_b, t_b) − g(x_b, t_b)|²
```

where `g` is the prescribed boundary value (Dirichlet) or flux (Neumann). Periodic BCs enforce `u_θ(x_left, t) = u_θ(x_right, t)`.

**Initial condition loss** — evaluated at `t=0`:

```
L_ic = (1/N_i) * Σ |u_θ(x_i, 0) − u_0(x_i)|²
```

**Total loss:**

```
L_total = w_r * L_residual + w_b * L_bc + w_i * L_ic + w_s * L_smooth
```

Default weights: `w_r = 15.0`, `w_b = 20.0`, `w_i = 10.0`, `w_s = 0.1`.

### Optimizer and scheduler

The default optimizer is Adam (`lr=0.005`, `β₁=0.9`, `β₂=0.999`, `weight_decay=0.0005`). The learning rate follows a cosine annealing schedule (`T_max=400`, `η_min=1e-7`). An alternative `reduce_lr` scheduler is available (ReduceLROnPlateau, `factor=0.5`, `patience=50`).

Early stopping monitors validation loss with `patience=100` epochs and `min_delta=1e-7`.

---

## Neural architectures

The six architectures are registered in `src/neural_networks/__init__.py` and selected via the `architecture` field in `ModelConfig` or `config.yaml`.

### FeedForward

**File:** `src/neural_networks/feedforward.py`

A standard multi-layer perceptron with configurable depth and width.

```yaml
architectures:
  feedforward:
    hidden_dims: [128, 128, 128, 128, 128, 128, 128]  # 7 layers
    activation: "tanh"
    dropout: 0.0
    layer_norm: true
```

**When to use:**
- Baseline comparison against more specialized architectures
- Black-Scholes equation (smooth, low-frequency solution)
- Problems where interpretability matters more than accuracy
- Quick prototyping

**Characteristics:** Tanh activation is preferred over ReLU for PDEs because it is smooth (infinitely differentiable), which is required for computing higher-order derivatives via autograd.

---

### ResNet

**File:** `src/neural_networks/resnet.py`

A residual network with skip connections. Each block computes `H(x) = F(x) + x`, helping gradients flow through deep networks.

```yaml
architectures:
  resnet:
    hidden_dim: 512
    num_blocks: 7
    activation: "tanh"
    dropout: 0.0
```

**When to use:**
- Burgers equation (shock formation, steep gradients)
- Cahn-Hilliard equation (higher-order, phase separation)
- Pendulum (nonlinear dynamics)
- Any problem where a plain FeedForward fails to converge

**Characteristics:** Skip connections prevent gradient vanishing in deep networks, making ResNet more expressive than FeedForward for the same parameter budget. The 7-block default with `hidden_dim=512` gives approximately 3.7M parameters.

---

### SIREN (Sinusoidal Representation Networks)

**File:** `src/neural_networks/siren.py`

Replaces standard activations with `sin(ω₀ · Wx + b)`. The frequency parameter `ω₀` controls the bandwidth of representable functions.

```yaml
architectures:
  siren:
    hidden_dims: [124, 124, 124, 124, 124, 124, 124]
    omega_0: 30.0
```

Weights are initialized with a special scheme (`U[-√(6/n)/ω₀, √(6/n)/ω₀]`) that preserves the distribution of activations across layers (Sitzmann et al., 2020).

**When to use:**
- Wave equation (oscillatory, time-periodic solutions)
- KdV equation (soliton dynamics, multi-scale structure)
- Any problem with high-frequency spatial or temporal features
- Implicit neural representations

**Characteristics:** SIREN can represent fine-scale oscillations that tanh networks require exponentially more neurons to approximate. The tradeoff is sensitivity to the `omega_0` hyperparameter — values between 10 and 30 work for most PDEs in this library.

---

### Fourier Features

**File:** `src/neural_networks/fourier.py`

Applies a random Fourier feature embedding before the MLP layers. Input coordinates `(x, t)` are lifted to a high-dimensional space via:

```
φ(x) = [sin(B·x), cos(B·x)]
```

where `B ~ N(0, scale²)` is a fixed random matrix sampled at initialization.

```yaml
architectures:
  fourier:
    mapping_size: 512
    hidden_dims: [512, 512, 512, 512]
    scale: 4.0
    activation: "tanh"
    layer_norm: true
    periodic: true
```

The Fourier transform is implemented with TorchScript (`@torch.jit.script`) for additional inference speed.

**When to use:**
- Heat equation with periodic boundary conditions
- Convection equation (transport, phase shifts)
- Allen-Cahn equation (sharp interfaces)
- Any problem with periodic BCs or spectral-like accuracy requirements

**Characteristics:** The random Fourier embedding is a kernel approximation of the RBF kernel. The `scale` parameter controls the frequency bandwidth — larger values capture higher frequencies but may cause spectral aliasing if set too aggressively. `scale=4.0` is a robust default for most 1D PDEs.

---

### Self-Attention Transformer

**File:** `src/neural_networks/attention.py`

A transformer-style architecture where each layer applies multi-head self-attention over the input sequence, followed by a feed-forward block.

```yaml
architectures:
  attention:
    hidden_dim: 124
    num_layers: 4
    num_heads: 4
    dropout: 0.0
    activation: "gelu"
```

**When to use:**
- Multi-scale problems where long-range spatial correlations matter
- Sequence-like problems (e.g., time series of PDE states)
- High-dimensional PDEs where positional structure is important
- Experimental use — attention is not yet the default recommendation for any PDE in this library, but is available for research comparisons

**Characteristics:** GELU activation is used rather than tanh because it is empirically more stable for attention-based architectures. With `num_heads=4` and `hidden_dim=124`, the model has roughly 600K parameters — lighter than ResNet or Fourier.

---

### Autoencoder

**File:** `src/neural_networks/autoencoder.py`

An encoder-decoder architecture where the input `(x, t)` is compressed to a latent space before being decoded to the solution.

```yaml
architectures:
  autoencoder:
    latent_dim: 64
    hidden_dims: [124, 248, 124]  # encoder → latent → decoder
    activation: "relu"
    dropout: 0.0
    layer_norm: true
```

**When to use:**
- Dimensionality reduction combined with PDE solving
- Problems where a low-dimensional latent representation of the solution is scientifically meaningful
- Multi-query inference where the latent vector is reused across parameter sweeps
- Experimental — recommended for research exploration rather than production accuracy

**Characteristics:** The latent dimension `latent_dim=64` acts as an information bottleneck. This forces the network to learn a compressed representation of the PDE solution manifold. ReLU is used here (rather than tanh) because the encoder-decoder structure provides sufficient smoothness through architecture depth.

---

## RL adaptive sampling

When `rl.enabled: true` in `config.yaml`, the `CollocationRLAgent` (implemented in `src/rl/rl_agent.py`) replaces uniform random collocation sampling with a DQN-based point selection strategy.

### How it works

1. **State:** The current collocation point `(x, t)` is the RL state (dimension = `state_dim`, default 2).

2. **Action:** The agent outputs a scalar sampling probability for each candidate point (dimension = `action_dim`, default 1).

3. **Reward:** After the PINN evaluates the PDE residual at the selected points, the reward is:
   ```
   r = w_r * |F[u_θ](x,t)|² + w_b * |BC error|² + w_i * |IC error|²
   ```
   Points with large residuals yield large rewards, encouraging the agent to focus sampling where the network is most inaccurate.

4. **Exploration:** Epsilon-greedy policy with exponential decay (`ε_start=1.0`, `ε_end=0.01`, `ε_decay=0.995`).

5. **Replay buffer:** Experience tuples `(s, a, r, s')` are stored in a ring buffer of size 10,000. The DQN network is updated every step using a mini-batch of 124 samples.

6. **Target network:** A separate target DQN is updated every 100 steps to stabilize training.

### DQN architecture

The `DQNNetwork` is a 3-layer MLP with LayerNorm and ReLU activations (separate from the PINN architectures):

```
Input(state_dim) → Linear(512) → LayerNorm → ReLU → Dropout(0.1)
                → Linear(512) → LayerNorm → ReLU → Dropout(0.1)
                → Linear(action_dim)
```

### When to enable RL sampling

RL adaptive sampling is most beneficial for:
- PDEs with sharp solution features (Burgers shocks, KdV solitons, Allen-Cahn interfaces)
- Long training runs (>5000 epochs) where uniform sampling wastes compute on smooth regions
- Benchmarking comparisons against RAR (Residual-Adaptive Refinement) sampling

For smooth PDEs (heat, wave, convection with simple ICs), uniform sampling with 5000 points is usually sufficient.

---

## Loss weighting

The default loss weights (`residual=15.0`, `boundary=20.0`, `initial=10.0`) were determined empirically across all nine supported PDEs.

| Weight | Value | Rationale |
|---|---|---|
| `residual` | 15.0 | Interior PDE points are most numerous; weight balances their average contribution |
| `boundary` | 20.0 | Boundary violations propagate inward and corrupt the interior solution; penalized most heavily |
| `initial` | 10.0 | IC sets the "anchor" for the time evolution; slightly lower than BC because it only applies at `t=0` |
| `smoothness` | 0.1 | Mild regularization that discourages spurious high-frequency artifacts; negligible for well-posed PDEs |

### Adaptive weights (optional)

Set `training.adaptive_weights.enabled: true` to use the `AdaptiveLossWeights` module (`src/components/adaptive_weights.py`). Two strategies are available:

- **`lrw` (Learning Rate Weighting):** Weights are updated proportionally to the gradient magnitude of each loss term.
- **`rbw` (Relative Balance Weighting):** Weights are adjusted to keep each loss term roughly equal in magnitude, using an exponential moving average with `alpha=0.7`.

Adaptive weights can improve convergence on stiff PDEs (Cahn-Hilliard, Allen-Cahn) but add overhead and may destabilize training if `alpha` is too low.

---

## Config injection: how `config.yaml` maps to `ModelConfig`

`Config.__init__()` reads `src/config/config.yaml` and builds a `ModelConfig` dataclass that is passed to `PINNModel`. The mapping is:

```yaml
# config.yaml
architectures:
  fourier:
    mapping_size: 512        → ModelConfig.mapping_size
    hidden_dims: [512, ...]  → ModelConfig.hidden_dims
    scale: 4.0               → ModelConfig.scale
    activation: "tanh"       → ModelConfig.activation
    dropout: 0.0             → ModelConfig.dropout
    layer_norm: true         → ModelConfig.layer_norm
    periodic: true           → ModelConfig.periodic
```

`PINNModel.__init__` dispatches on `ModelConfig.architecture` to instantiate the correct class:

```python
ARCH_MAP = {
    "feedforward": FeedForwardNetwork,
    "resnet":      ResNet,
    "siren":       SIREN,
    "fourier":     FourierNetwork,
    "attention":   AttentionNetwork,
    "autoencoder": AutoEncoder,
}
```

All architecture classes inherit from `BaseNetwork` (`src/neural_networks/base_network.py`), which standardizes the `forward(x: Tensor) -> Tensor` interface and provides shared utilities for weight initialization and parameter counting.
