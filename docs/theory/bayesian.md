# The Bayesian view and uncertainty

This page reads the PINN loss as a probability statement and lists the ways to put an error bar on an identified parameter. Read it before reporting an interval, or before choosing loss weights.

## The loss is a negative log-posterior

$$
y_i = (\mathcal{H}u)_i + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)
$$

- **$y_i$**, **$\sigma_i$**: measurement $i$ and its noise standard deviation.
- **$\mathcal{H}$**: the sensor operator; $(\mathcal{H}u)_i$ is what sensor $i$ reads from the field $u$.

Add a Gaussian belief that the residual, the boundary mismatch and the initial mismatch are zero, with scales $\sigma_r$, $\sigma_b$, $\sigma_0$. Up to a constant:

$$
-\log p(w, \theta \mid y) = \frac{1}{2\sigma_r^2}\|\mathcal{N}_\theta[u_w]\|^2 + \frac{1}{2\sigma_b^2}\|\mathcal{B}u_w\|^2 + \frac{1}{2\sigma_0^2}\|u_w(\cdot, 0) - u_0\|^2 + \sum_i \frac{(y_i - (\mathcal{H}u_w)_i)^2}{2\sigma_i^2} - \log p(w, \theta)
$$

- **$w$, $\theta$**: the network weights and the unknown parameters of the equation.
- **$u_w$**: the network (the surrogate of the solution).
- **$\mathcal{N}_\theta[u_w]$**: the residual of the equation; **$\mathcal{B}u_w$**: the boundary mismatch; **$u_0$**: the initial condition.
- **$p(w, \theta \mid y)$**, **$p(w, \theta)$**: the posterior and the prior.

This is the PINN loss with weights $1/\sigma^2$ (Kaipio & Somersalo, 2005; Stuart, 2010).

- **Maximum a posteriori (MAP)**: the most probable point of the posterior. Training finds it.
- **The weights are not tuning knobs.** The data weight is the inverse noise variance; the physics weights encode how much model error one admits.
- **A per-sensor noise level is a per-observation weight**, which [`fisher_information`](../api/analysis.md) accepts. [`pinnrl.uq.likelihood_weights`](../api/uq.md) turns a noise level into loss weights.
- **A change of weights is a change of prior.** Results obtained with different weights are not comparable without saying so.
- **K39**: setting the data weight from the measured noise did not beat tuned weights (10 and 100). It was the worst, with bias up to −21 %, because the residual is systematic surrogate error, not Gaussian noise ([benchmarks](../benchmarks.md#calibrated-uncertainty-experiments-f06-k10-k39-k46)). F01 motivated the test: a fixed weight that ignores the noise level biased the estimate at few, noisy points.

**For a PINN:** state the weights with every estimate, and do not expect the noise-model weight to help while the surrogate's own error exceeds the noise.

## Sources of uncertainty

An error bar on $\theta$ must say which of these it covers.

| Source | Where it comes from | How it shows |
|---|---|---|
| Aleatoric | measurement noise $\varepsilon$ | the spread across noise draws; bounded below by $F^{-1}$ |
| Parametric epistemic | finite data, correlated parameters | the posterior width along sloppy directions |
| Surrogate | the network's approximation error and the non-convex training | the bias of F01, the seed spread |
| Model form | the equation is not the process | a residual structure the surrogate cannot remove (F08, planned) |
| Input | a recorded input with an offset (a steam temperature) | a shift of the estimate along the direction the input can be traded against; ignored, the interval is too narrow (J06) |

- **$F$**: the Fisher matrix of [identifiability](identifiability.md); $F^{-1}$ is the Cramér–Rao bound.
- **Sloppy direction**: a parameter combination the data determine poorly.

What separates them:

- The Fisher bound covers the first source only.
- Ensembles over seeds and over noise draws separate the first from the third.
- No method internal to the model detects model-form error. This is why a twin must report its residual against the plant.

**For a PINN:** an interval from the Fisher matrix alone says nothing about the network's own bias or a wrong equation.

## Approximating the posterior

Four approximations, compared broadly on physics-informed models by Psaros et al. (2023).

### Laplace

$$
p(\theta \mid y) \approx \mathcal{N}\big(\hat\theta,\ \Sigma_\theta\big), \qquad \Sigma_\theta \approx \big(J^\top \Sigma^{-1} J\big)^{-1} = F^{-1}
$$

- **$\hat\theta$**: the trained estimate.
- **$J$**: the sensitivities of the observations to $\theta$, evaluated on the surrogate.
- **$\Sigma$**: the noise covariance.

1. Expand the loss to second order at the optimum.
2. The posterior is Gaussian, with covariance the inverse Hessian (MacKay, 1992).
3. Eliminate $w$. The block for $\theta$ is the generalized Gauss–Newton approximation of $F^{-1}$.

- **Generalized Gauss–Newton**: the Hessian with the second derivatives of the model dropped, which leaves $J^\top \Sigma^{-1} J$.
- It reuses the sensitivities of [identifiability](identifiability.md). The function is [`pinnrl.uq.laplace_posterior`](../api/uq.md).
- K41 (planned) asks how much of the network the Hessian must cover.

### Deep ensembles

- The estimates $\hat\theta^{(1)}, \dots, \hat\theta^{(M)}$ from $M$ initializations are treated as posterior samples, and the interval is their percentiles (Lakshminarayanan, Pritzel & Blundell, 2017).
- The spread captures the surrogate's non-convexity, which Laplace does not, and misses nothing that retraining reveals.
- F06 measures coverage; K40 (planned) measures it against the ensemble size. The summary function is [`pinnrl.uq.ensemble_summary`](../api/uq.md).

### Sampling

- **Hamiltonian Monte Carlo**: a sampler that follows the gradient of the log-posterior, here over $(w, \theta)$ (Yang, Meng & Karniadakis, 2021).
- It is the reference posterior for small networks (K42, planned) and too costly for the rest.

### Conformal intervals

$$
\big[\, u_w(x) - q,\ u_w(x) + q \,\big], \qquad q = \text{the calibration quantile of } |y_i - (\mathcal{H}u_w)_i| \text{ on held-out sensors}
$$

- Coverage holds in finite samples with no distributional assumption (Vovk, Gammerman & Shafer, 2005).
- It bounds the *field* prediction rather than $\theta$ (K44, planned).

**For a PINN:** Laplace is the cheapest interval and is valid where the surrogate resolves the solution; an ensemble shows the training scatter that Laplace cannot.

## Calibration

$$
\Pr\big(\theta_{\text{true}} \in I_{90\,\%}\big) = 0.90 \quad \text{over many data sets}
$$

- **Calibrated interval**: the interval $I_{90\,\%}$ reported at the 90 % level contains the truth as often as it claims.

Coverage alone is not enough: an interval that is always the whole real line is calibrated. pinnrl reports three quantities, from F06 and K10 onward ([`pinnrl.uq.metrics`](../api/uq.md)):

- **Empirical coverage**: the fraction of synthetic data sets, with known truth, whose nominal interval contains the truth.
- **Width** (sharpness): it should be as small as coverage allows (Gneiting, Balabdaoui & Raftery, 2007).
- **Continuous ranked probability score (CRPS)**: a proper scoring rule that rewards both at once (Gneiting & Raftery, 2007).

$$
\mathrm{CRPS}(P, y) = \int_{-\infty}^{\infty} \big(P(z) - \mathbb{1}\{z \ge y\}\big)^2\, dz
$$

- **$P$**: the predicted cumulative distribution; **$y$**: the true value. Lower is better.
- **Proper scoring rule**: a score that the true distribution minimizes on average.

A method is kept only where it is calibrated, and the page of each method states the regime in which that was checked.

F06/K10 (heat $\alpha$, 30 data sets, target 90 %): 8-member ensembles over seeds cover 13 % (the members agree to 0.5 % while the estimate varies by 1.75 % across noise draws), Laplace 77 % (the width is right, a 2 % bias costs coverage), the physics region of [equivalence regions](equivalence_regions.md) 90 % ([benchmarks](../benchmarks.md#calibrated-uncertainty-experiments-f06-k10-k39-k46)).

### Intervals for an identified function (J06)

J06 measures, on the evaporator, when the Fisher (Laplace) interval of an identified *function* is calibrated. Each condition is an assumption the interval inherits, not something the data check for it.

| Condition | What J06 measured when it fails |
|---|---|
| **The form assumed for the unknown contains the truth.** | A ramp assumed for a fouling curve that saturates gives a ±2.5 % interval around an estimate 25 % off. |
| **The model's own error is below the interval's width.** | A lumped model biased by 2.7 % supports a ±7 % interval and not a ±2 % one. |
| **Every input the unknown can be traded against is either measured to that accuracy or among the unknowns with its prior.** | A steam record trusted to 0.3 K and taken at face value reduces coverage to 21–52 %. Estimated under its prior, coverage returns at five times the width, and that width is the honest one. |

- The form sets the width as much as the data do: ±2 % for a ramp, ±8 % for six modes, on the same records.
- Only the misfit says when the form is wrong.

**For a PINN:** an interval reported for a process estimate should name the three: the form, the model, and the inputs it conditions on.

## References

- Kaipio, J. & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
- Stuart, A. M. (2010). Inverse problems: a Bayesian perspective. *Acta Numerica* 19, 451–559. [doi:10.1017/S0962492910000061](https://doi.org/10.1017/S0962492910000061)
- MacKay, D. J. C. (1992). A practical Bayesian framework for backpropagation networks. *Neural Computation* 4, 448–472. [doi:10.1162/neco.1992.4.3.448](https://doi.org/10.1162/neco.1992.4.3.448)
- Lakshminarayanan, B., Pritzel, A. & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*.
- Yang, L., Meng, X. & Karniadakis, G. E. (2021). B-PINNs: Bayesian physics-informed neural networks for forward and inverse PDE problems with noisy data. *J. Comput. Phys.* 425, 109913. [doi:10.1016/j.jcp.2020.109913](https://doi.org/10.1016/j.jcp.2020.109913)
- Psaros, A. F., Meng, X., Zou, Z., Guo, L. & Karniadakis, G. E. (2023). Uncertainty quantification in scientific machine learning: methods, metrics, and comparisons. *J. Comput. Phys.* 477, 111902. [doi:10.1016/j.jcp.2022.111902](https://doi.org/10.1016/j.jcp.2022.111902)
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer. [doi:10.1007/b106715](https://doi.org/10.1007/b106715)
- Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *J. R. Stat. Soc. B* 69, 243–268. [doi:10.1111/j.1467-9868.2007.00587.x](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- Gneiting, T. & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *J. Am. Stat. Assoc.* 102, 359–378. [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
