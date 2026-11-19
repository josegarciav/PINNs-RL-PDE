# Well-posedness and stability

This page states when a small change in the data gives a small change in the estimate. Read it before trusting a parameter identified from noisy measurements.

Notation is shared with the [theory index](index.md): state $u$, parameters $\theta$, observations $y = \mathcal{H}u_\theta + \varepsilon$, network $u_w$.

## Hadamard's three conditions

A problem is **well posed** (Hadamard, 1902) when its solution

1. exists,
2. is unique,
3. depends continuously on the data.

- **Forward problem**: given $\theta$ and the initial and boundary data, find $u$. Well posed for the [catalogue](../custom_equations.md#the-catalogue) equations.
- **Inverse problem**: given $y$, find $\theta$ (or an initial state, or a coefficient field). It is usually not well posed, because it inverts a map that smooths.

The heat equation shows why condition 3 fails. Diffusion damps each mode, so reading the initial state from a late measurement multiplies the noise by the inverse factor:

$$
\sin(k\pi x) \;\longmapsto\; e^{-\alpha k^2 \pi^2 t}\,\sin(k\pi x),
\qquad
\text{noise gain at mode } k = e^{\alpha k^2 \pi^2 t}.
$$

- **$k$**, **$\alpha$**, **$t$**: the mode number, the diffusivity and the time of the measurement.

The gain is unbounded in $k$. Uniqueness (condition 2) can also fail; that is the subject of [identifiability](identifiability.md).

**For training:** a PINN that fits the data well can still return a wrong $\theta$ when the inverse problem is ill posed. A small loss does not repair this.

## Conditional stability

Under an a-priori bound on the unknown, continuity returns, with a modulus that is not Lipschitz:

$$
\|\theta\|_{Z} \le M
\quad\Longrightarrow\quad
\|\theta_1 - \theta_2\|_X \le \omega\big(\|y_1 - y_2\|_Y\big), \qquad \omega(\delta) \to 0 \text{ as } \delta \to 0 .
$$

- **A-priori bound**: the restriction $\|\theta\|_Z \le M$ of the unknown to a set bounded in a stronger norm $Z$.
- **$\theta_1, \theta_2$**, **$y_1, y_2$**: two unknowns in that set and their data, measured in the norms $X$ and $Y$.
- **$\omega$**: the modulus of continuity. The well-posed (Lipschitz) case is $\omega(\delta) = C\delta$.

Two forms of $\omega$ occur (Alessandrini et al., 2009; Isakov, 2017):

$$
\omega(\delta) = C\,\delta^{\mu} \quad \text{(Hölder)},
\qquad
\omega(\delta) = C\,|\log\delta|^{-\mu} \quad \text{(logarithmic)}.
$$

- **Hölder stability**, $0 < \mu \le 1$: the good case. With $\mu = 1/2$, four digits of data give two digits of parameter.
- **Logarithmic stability**: the bad case. Examples are the backward heat problem and the Cauchy problem for elliptic equations.

A scalar parameter in an otherwise known equation is the benign case. For the heat diffusivity, F01 found the spread of the estimate across noise draws to follow the Cramér–Rao bound (the smallest standard deviation any unbiased estimator can reach), so the map from data to $\alpha$ is locally Lipschitz, with a constant the Fisher information gives ([identifiability](identifiability.md)).

**For training:** the type of $\omega$ sets the accuracy you can expect from the data, whatever the optimizer does.

## Regularization, and the physics as a regularizer

Classical inverse theory restores continuity by regularization (Engl, Hanke & Neubauer, 1996):

$$
\min_\theta \; \|\mathcal{H}u_\theta - y\|^2 + \lambda\, R(\theta).
$$

- **$R(\theta)$**: the penalty, which encodes the a-priori bound. Among the $\theta$ that fit the data equally, it selects the one the prior prefers.
- **$\lambda$**: the regularization weight, chosen from the noise level.

A PINN inverse problem has the same shape:

$$
\min_{w,\,\theta} \; \|\mathcal{H}u_w - y\|^2 + \lambda\, \|\mathcal{N}_\theta[u_w]\|^2 .
$$

It differs from the classical form in two ways:

1. The residual term $\|\mathcal{N}_\theta[u_w]\|^2$ regularizes the *state*. The network cannot fit noise with a function that violates the equation.
2. The parameter $\theta$ carries no explicit penalty. It is regularized only through the physics.

Consequence measured in F01 ([benchmarks](../benchmarks.md#identification-baseline-experiment-f01)): with a fixed data weight and few, noisy points, the estimate of $\alpha$ is biased low. The residual term prefers a smoother field, and a smoother field implies slower diffusion.

**For training:** set the data weight from the noise level. The [Bayesian view](bayesian.md) turns the weights into noise levels.

## Stability of the estimate under disturbances

For a disturbance $\delta y$ of the data, the first-order shift $\delta\theta$ of a least-squares estimate is

$$
\delta\theta \approx (J^\top \Sigma^{-1} J)^{-1} J^\top \Sigma^{-1}\, \delta y, \qquad J = \frac{\partial \mathcal{H}u_\theta}{\partial\theta}, \qquad F = J^\top\Sigma^{-1}J .
$$

- **$J$**: the sensitivity of the predicted observations to the parameters.
- **$\Sigma$**: the covariance of the measurement noise.
- **$F$**: the Fisher information. $F^{-1}$ is the covariance of the random errors; $F^{-1}J^\top\Sigma^{-1}$ is the gain on systematic errors (bias, drift, a stuck sensor).

The disturbance suite (item 4 of the v0.5 list in the [history](../history.md), experiments K05–K07, [benchmarks](../benchmarks.md#disturbance-and-stability-experiments-k05k07)) measures the observed $\delta\theta$ against this prediction:

| Experiment | Disturbance | Function |
|---|---|---|
| K05 | noise, bias, dropout and delay in the data | [`perturb_observations`](../api/analysis.md#analysis.disturbance.perturb_observations), [`linear_response`](../api/analysis.md#analysis.disturbance.linear_response) |
| K06 | perturbations of the problem: initial data, boundary data, a coefficient | [`field_response`](../api/analysis.md#analysis.disturbance.field_response) |
| K07 | error against residual | [`stability_ratio`](../api/analysis.md#analysis.disturbance.stability_ratio) |

- **Empirical stability constant**: the ratio of the observed shift to the predicted one. In K05 it is 0.78–1.06 for bias and delay.

**For training:** the shift a sensor fault causes can be computed from $J$ before any training.

## What a PINN adds and what it does not

The network adds its own failure of continuity: training is non-convex, and the surrogate (the trained network $u_w$ used in place of the exact solution) reaches a residual that is small but not zero. Two consequences matter for an estimate:

| Consequence | Evidence |
|---|---|
| The parameters absorb the surrogate's approximation error. More observations do not remove this bias. | F01: a floor of 0.1–0.2 % on clean data without weight decay and gradient clipping, about 2 % with them (attributed to the regularizers by D01). |
| With the same data, different seeds can give different estimates when the problem is ill-conditioned. | The seed spread is reported next to the noise spread. F04 checks that the estimate does not depend on where it started. |

Both are properties of the solver, not of the inverse problem, and the [numerical-analysis bridge](numerical_analysis.md) treats them as such.

**For training:** for an inverse run, start from `weight_decay=0` and `gradient_clipping=0` (F01), and report the seed spread.

## References

- Hadamard, J. (1902). Sur les problèmes aux dérivées partielles et leur signification physique. *Princeton University Bulletin* 13, 49–52.
- Engl, H. W., Hanke, M. & Neubauer, A. (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
- Alessandrini, G., Rondi, L., Rosset, E. & Vessella, S. (2009). The stability for the Cauchy problem for elliptic equations. *Inverse Problems* 25, 123004. [doi:10.1088/0266-5611/25/12/123004](https://doi.org/10.1088/0266-5611/25/12/123004)
- Isakov, V. (2017). *Inverse Problems for Partial Differential Equations*, 3rd ed. Springer. [doi:10.1007/978-3-319-51658-5](https://doi.org/10.1007/978-3-319-51658-5)
- Kaipio, J. & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
