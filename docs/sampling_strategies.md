# Sampling Strategies

PINNs approximate PDE solutions by minimizing a physics-informed loss at a set of
**collocation points** — locations in the spatiotemporal domain where the PDE residual
is evaluated. *Where* you place these points directly affects how fast and how
accurately the network converges.

This project provides four sampling strategies, ordered from simplest to most
sophisticated:

| Strategy | Adaptive? | Needs model? | Needs RL agent? | Point count |
|---|---|---|---|---|
| `uniform` | No | No | No | Approximate |
| `stratified` | No | No | No | Exact |
| `residual_based` | Yes | Yes | No | Exact |
| `adaptive` | Yes | No | Yes | Exact |

---

## 1. Uniform Sampling

**What it does.** Creates a regular grid across space and time, then adds a tiny
amount of random noise (1% of domain width) so the network doesn't memorize grid
positions. In multiple dimensions, a meshgrid is constructed and randomly
subsampled to the requested point count.

**When to use it.** Default baseline. Works well for smooth solutions with no sharp
gradients or localized features (e.g., slow diffusion problems).

**Limitation.** Wastes points in regions where the solution is flat and undersamples
regions where it changes rapidly — like shock fronts in Burgers' equation or steep
initial transients.

### Math

For a 1D spatial domain $[a, b]$ and time domain $[0, T]$, generate an
$n_x \times n_t$ grid where $n_x = n_t = \lfloor\sqrt{N}\rfloor$:

$$x_i = a + \frac{(b - a) \cdot i}{n_x - 1}, \quad i = 0, \ldots, n_x - 1$$

$$t_j = \frac{T \cdot j}{n_t - 1}, \quad j = 0, \ldots, n_t - 1$$

Then add jitter:

$$\tilde{x}_{ij} = x_i + \epsilon_x, \quad \epsilon_x \sim \mathcal{N}(0,\, (0.01 \cdot (b-a))^2)$$

$$\tilde{t}_{ij} = t_j + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0,\, (0.01 \cdot T)^2)$$

and clamp back to the domain boundaries.

---

## 2. Stratified Sampling

**What it does.** Divides each dimension into $N$ equal bins (where $N$ is the
requested number of points) and places exactly one random sample inside each bin.
The samples are then shuffled independently across dimensions, giving the
Latin Hypercube property: every row and column of the sample matrix contains exactly
one point.

**When to use it.** Better space-filling than uniform for the same number of
points. Guarantees no large empty gaps in any single dimension. Good for
initial exploration before you have a trained model.

**Limitation.** Still static — doesn't concentrate points where the PDE is hard
to satisfy. Marginal improvement over uniform for smooth problems.

### Math

For dimension $d$ with bounds $[l_d, u_d]$, the bin width is:

$$\Delta_d = \frac{u_d - l_d}{N}$$

The $i$-th sample in dimension $d$ is:

$$s_{i,d} = l_d + (i + U_i) \cdot \Delta_d, \quad U_i \sim \text{Uniform}(0, 1)$$

for $i = 0, 1, \ldots, N-1$. After generating all $N$ values for dimension $d$,
apply a random permutation $\pi_d$ independently:

$$\hat{s}_{i,d} = s_{\pi_d(i),\, d}$$

This ensures the Latin Hypercube property: projecting the $N$ points onto any
single axis yields exactly one point per bin.

---

## 3. Residual-Based Adaptive Refinement (RAR)

**What it does.** Uses the current neural network to decide where to place points.
First, generates a large candidate pool (4x the requested count) using uniform
sampling. Then evaluates the PDE residual at every candidate and resamples $N$
points with probability proportional to the residual magnitude. Regions where the
network violates the PDE most get more points.

**When to use it.** Mid-to-late training, once the network has learned the broad
shape of the solution but struggles with localized features (shocks, boundary
layers, sharp gradients). Especially effective for Burgers, KdV, and Allen-Cahn
equations.

**Limitation.** Requires a forward pass through the model at every resampling step
(adds ~25% compute per epoch). On the first epoch there is no trained model, so it
falls back to uniform. Also, it is greedy: it only looks at where the current
residual is high, not where sampling would most improve future training.

### Math

1. Generate a candidate pool $\mathcal{P} = \{(\mathbf{x}_k, t_k)\}_{k=1}^{4N}$ via uniform sampling.

2. Compute the PDE residual at each candidate:

$$r_k = |\mathcal{L}[u_\theta](\mathbf{x}_k, t_k)|$$

where $\mathcal{L}$ is the PDE differential operator and $u_\theta$ is the current
network prediction.

3. Convert to sampling probabilities:

$$p_k = \frac{r_k + \varepsilon}{\sum_{j=1}^{4N}(r_j + \varepsilon)}, \quad \varepsilon = 10^{-8}$$

4. Draw $N$ points from $\mathcal{P}$ with replacement according to $\{p_k\}$.

The small $\varepsilon$ ensures that even zero-residual regions retain a nonzero
probability of being sampled, preventing the network from "forgetting" regions
it has already learned.

---

## 4. RL-Based Adaptive Sampling

**What it does.** A reinforcement learning agent (DQN) learns a sampling policy
over the domain. The agent takes the current spatiotemporal coordinates as state
and outputs a probability for each candidate point. Points are then sampled
according to these learned probabilities.

The key difference from RAR: the RL agent learns *where sampling helps training
converge faster*, not just where the residual is currently high. It can learn to
preemptively sample regions that will become problematic, and it improves its
policy across epochs via experience replay.

**When to use it.** The flagship strategy of this project. Best for problems where
the residual landscape shifts during training (moving fronts, evolving phase
boundaries) and a static or greedy strategy can't keep up.

**Limitation.** Adds the overhead of maintaining and training the RL agent
(replay buffer, target network updates, epsilon scheduling). Requires tuning
RL hyperparameters (state_dim, hidden_dim, learning rate, epsilon decay).
Non-deterministic by nature.

### Math

At each training epoch $e$, the agent operates as follows:

1. Construct a candidate grid $\mathcal{G}$ of size $M = (\min(100, \sqrt{N}))^{d+1}$ over the domain.

2. The agent's Q-network $Q_\phi$ maps each candidate to a score:

$$q_k = Q_\phi(\mathbf{x}_k, t_k), \quad k = 1, \ldots, M$$

3. Convert to sampling probabilities:

$$p_k = \frac{|q_k|}{\sum_{j=1}^{M} |q_j|}$$

4. Draw $N$ points from $\mathcal{G}$ via multinomial sampling with probabilities $\{p_k\}$.

5. Add small Gaussian noise to break grid alignment:

$$(\tilde{\mathbf{x}}_k, \tilde{t}_k) = (\mathbf{x}_k, t_k) + \mathcal{N}(0, \sigma^2 I)$$

where $\sigma$ is the grid cell size.

6. $\varepsilon$-greedy exploration: with probability $\varepsilon_e$ the agent
samples uniformly instead of following its policy. The exploration rate decays:

$$\varepsilon_e = \max(0.1,\; \varepsilon_0 \cdot 0.95^e)$$

The agent's policy improves over training as the replay buffer accumulates
experience about which sampling decisions led to faster loss reduction.

---

## Choosing a Strategy

```
Is this your first experiment with a new PDE?
  YES --> uniform (simple baseline, no setup needed)
  NO  --> Do you have a trained model?
            NO  --> stratified (better coverage, no model needed)
            YES --> Is the residual localized (shocks, boundary layers)?
                      YES --> residual_based (cheap, effective)
                      NO  --> adaptive (RL learns optimal placement)
```

For benchmarking, run all four strategies on the same PDE and compare:

- **Convergence speed**: epochs to reach a target L2 error
- **Final accuracy**: L2 error after fixed epochs
- **Point efficiency**: error per collocation point evaluated
- **Wall-clock time**: total training time including sampling overhead

---

## Usage

```python
from pinnrl.pdes.heat_equation import HeatEquation
from pinnrl.rl.rl_agent import RLAgent

pde = HeatEquation(...)

# 1. Uniform
x, t = pde.generate_collocation_points(1000, strategy="uniform")

# 2. Stratified
x, t = pde.generate_collocation_points(1000, strategy="stratified")

# 3. Residual-based (pass current model)
x, t = pde.generate_collocation_points(1000, strategy="residual_based", model=model)

# 4. RL adaptive (attach agent first)
pde.rl_agent = RLAgent(state_dim=2, action_dim=1, hidden_dim=64, device="cpu")
x, t = pde.generate_collocation_points(1000, strategy="adaptive")
```

---

## Configuration

In `config.yaml`:

```yaml
training:
  collocation_distribution: "uniform"  # or "stratified", "residual_based", "adaptive"
```

The trainer reads this value and passes it to `generate_collocation_points()`. When
an RL agent is attached, the trainer automatically overrides to `"adaptive"`.
