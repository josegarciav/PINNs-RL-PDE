# Troubleshooting

Failure modes the package and its experiments have met. Each has a symptom, a cause and a fix. Experiment codes (`F01`, `J06`, ...) name the source of each number; see [Benchmarks](benchmarks.md).

## Installation and the native engine

### The compiled engine is not used

```python
from pinnrl import native
print(native.available(), native.unavailable_reason())
```

Training still works on autograd, two to eight times slower per step ([I17](benchmarks.md#the-step-against-the-alternatives-experiment-i17)).

| Reason printed | Fix |
|---|---|
| `no C++ compiler found` | macOS: `xcode-select --install`. Linux: install `g++` |
| `compiling jets.cpp failed` on Linux | `apt install libopenblas-dev`, or point `PINNRL_NATIVE_CXXFLAGS` and `PINNRL_NATIVE_LDFLAGS` at OpenBLAS |
| `the native kernels build on macOS (Accelerate) and Linux (OpenBLAS) only` | None on other platforms; autograd runs |

- **`available()` is true but `pde.native_active(model)` is false:** the network or the problem is outside what the engine covers. See [where it does not apply](engine.md#where-it-does-not-apply).

### A pause of a second or two early in a run

- **Cause:** the step for your equations is compiled once per machine, and takes over at evaluation 32.
- **Fix:** none needed; later runs load it from the cache. `PINNRL_NATIVE_SPECIALIZE=off` skips the build.

### A stale cache

- **Where:** `~/.cache/pinnrl/native`.
- **Fix:** delete the directory. It is always safe; libraries are rebuilt on demand.

### `TheWellNotInstalledError`

- **Fix:** `pip install 'pinnrl[well]'`. The registry works without it; loading a slice does not.

## Training

### The loss falls and the solution is wrong

The residual is not the error. They are linked by a stability constant $C$:

$$
\lVert u - u_{\text{exact}} \rVert \le C\, \lVert r \rVert .
$$

- **Evidence:** for heat in $d = 1 \ldots 5$ the residual stays at 2–7 % while the error grows from 2 % to 44 % (K08).
- **Fix:** check against a reference: `pde.validate`, `pinnrl.benchmarks`, or `pinnrl-benchmark fdm`.

### A system of ODEs collapses to a constant

- **Cause:** a trivial solution satisfies the residual almost everywhere (K09).
- **Fix:** time-marching windows rescue SIR (error 1.0 to 0.1 %, [H04](benchmarks.md#time-marching-windows-experiment-h04)). Nothing rescues Lorenz; use the reference solver.

### `adam_lbfgs` diverges

- **Evidence:** an early switch diverged in 3 of 6 runs on Burgers (D01).
- **Fix:** leave more of the budget to Adam (`adam_lbfgs_switch_ratio`, default 0.7), or run L-BFGS as a second `train()` call.

### Results vary more between seeds than between methods

- **Evidence:** on Burgers the error varies six-fold from seed to seed (B01).
- **Fix:** compare with paired seeds and equal budgets, as `pinnrl-benchmark sampling` does.

### A network with `torch.nn.LayerNorm` trains on a wrong gradient

- **Cause:** `torch.nn.LayerNorm` (PyTorch 2.10) returns a wrong weight gradient for a loss with second derivatives of the network. It is 2–3 % off in norm on heat (I12).
- **Fix:** use `pinnrl.neural_networks.ExactLayerNorm`. Same function, same parameters, same checkpoint keys.

### Timings are twice those of the benchmarks page

- **Cause:** a laptop on battery in low power mode runs at about half speed.
- **Fix:** run on mains power with nothing else using the processor.

## Inverse problems

- **Inverse problem:** estimating a parameter $\theta$ of the equation from data.
- **Cramér–Rao bound:** the smallest standard deviation any unbiased estimate can have, $\sigma_\theta \ge \sqrt{(F^{-1})_{\theta\theta}}$.
- **Fisher information** $F = J^\top \Sigma^{-1} J$: how much the data say about $\theta$, from the sensitivities $J = \partial u / \partial \theta$ and the noise covariance $\Sigma$.

### An identified parameter stalls 1–2 % from the truth

- **Cause:** weight decay and gradient clipping act as priors on the network, and the parameter absorbs them.
- **Evidence:** the heat diffusivity stalled at 1.5–2 % with them; the floor is 0.1–0.2 % without (F01, D01).
- **Fix:** `weight_decay=0`, `gradient_clipping=0` (`--weight-decay 0 --gradient-clipping 0`).

### The estimate is confidently wrong

- **Cause:** a PINN returns a number whether or not the data determine it. Field error does not predict parameter error.
- **Evidence:** KdV's coefficients come back 18 % and 25 % low where the bound says 0.6 % and 1.4 % (F02). The Burgers viscosity is 2.5–5 times too large (F03).
- **Fix:** compare with the bound: `pinnrl.analysis.fisher_information`. See [Inverse problems](inverse_problems.md).

### Two parameters cannot be told apart

- **Cause:** some combinations are one unknown for any method.
- **Evidence:** on the evaporator, 0.5 K of steam bias reproduces the records of a 30 % fouling error (J04).
- **Fix:** read the correlations of $F$ before training. Fix one of the pair, or add the measurement that separates them (`pinnrl.analysis.greedy_design`).

### An unknown function collapses to a constant

- **Evidence:** the droplet's drying-rate closure does so from a cold start on every seed (J07).
- **Fix:**
    1. Constrain it: `positive`, `monotone`, an `anchor` where its value is known.
    2. Start it from a pointwise estimate of the record.
    3. Judge it by the misfit through the reference solver ([Process twins](process_twins.md)).

### An interval that does not cover

- **Coverage:** the fraction of repeated data sets whose interval contains the truth. A 90 % interval should cover 90 %.
- **Evidence:** ensembles over initializations covered 13 %; a Laplace interval 77 % with a 2 % bias; the swarm's region 90 % (F06).
- **Fix:** measure it with `pinnrl.uq.coverage` before reporting an interval.

## Process twins

### The twin fails its misfit test

The test runs the fitted model through a reference solver and returns

$$
\chi^2_{\text{red}} = \frac{1}{n} \sum_{i=1}^{n} \frac{(y_i - \hat y_i)^2}{\sigma_i^2},
$$

which is near 1 for a fit.

- **Evidence:** the PINN twin failed on 22 of 36 runs on the evaporator (J06) and on every run on the counter-current dryer (J13).
- **Fix:**
    1. Retry from new seeds: `train_with_restarts` keeps the run with the best misfit.
    2. If no run passes and the model has a reference solver, use the classical estimator: `ExtendedKalmanFilter`, `MovingHorizonEstimator`, or least squares through `reference()`.

### A filter's interval is too narrow

- **Evidence:** a model biased by 2.7 % (11 cells where 51 are needed) cut the coverage to 0.33–0.69. A steam record trusted beyond its 0.3 K accuracy cut it to 0.21–0.52 (J06).
- **Fix:** refine the grid until the estimate stops moving. Estimate an instrument's offset under its prior.

## Tests and development

| Symptom | Cause | Fix |
|---|---|---|
| Eleven tests skipped | They download from Hugging Face | `PINNRL_NETWORK_TESTS=1` (about ten minutes) |
| Kernel tests skipped | The kernel cannot be built | See the first section; `PINNRL_REQUIRE_NATIVE=1` makes it a failure |
| The docs build fails on a link | `mkdocs build --strict` treats broken links and anchors as errors | Fix the link; add a new page to the navigation of `mkdocs.yml` |
| A snippet fails in the release check | `scripts/run_doc_snippets.py` runs every `python` block of `README.md` and `docs/` | Put `<!-- doctest: skip -->` before a block that is slow or needs missing files |
| An API name is undocumented | `test_docs_api_coverage.py` checks every public name | Add a `:::` entry under `docs/api/` |
