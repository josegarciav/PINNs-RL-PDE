# Well-posedness and stability

This page states when a small change in the data gives a small change in the estimate. Read it before trusting a parameter identified from noisy measurements.

Notation is shared with the [theory index](index.md): state $u$, parameters $\theta$, observations $y = \mathcal{H}u_\theta + \varepsilon$, network $u_w$.

## Hadamard's three conditions

A problem is **well posed** (Hadamard, 1902) when its solution

1. exists,
2. is unique,
3. depends continuously on the data.

- **Forward problem**: given $\theta$ and the initial and boundary data, find $u$. For the equations in the [catalogue](../custom_equations.md#the-catalogue) it is well posed: nearby data give nearby solutions.
- **Inverse problem**: given $y$, find $\theta$ (or an initial state, or a coefficient field). It is usually not well posed, because it inverts a map that smooths.

The heat equation shows why condition 3 fails. Diffusion multiplies each mode of the initial state by a damping factor, so reading the initial state from a late measurement multiplies the noise by the inverse:

$$
\sin(k\pi x) \;\longmapsto\; e^{-\alpha k^2 \pi^2 t}\,\sin(k\pi x),
\qquad
\text{noise gain at mode } k = e^{\alpha k^2 \pi^2 t}.
$$

- **$k$**: the mode number (spatial frequency).
- **$\alpha$**: the diffusivity.
- **$t$**: the time of the measurement.

The gain is unbounded in $k$, so continuity fails. Uniqueness (condition 2) can also fail; that is the subject of [identifiability](identifiability.md).

**For training:** a PINN that fits the data well can still return a wrong $\theta$ when the inverse problem is ill posed. A small loss does not repair this.

## Conditional stability

Ill-posedness is rarely total. Under an a-priori bound on the unknown, continuity returns, with a modulus that is not Lipschitz:

$$
\|\theta\|_{Z} \le M
\quad\Longrightarrow\quad
\|\theta_1 - \theta_2\|_X \le \omega\big(\|y_1 - y_2\|_Y\big), \qquad \omega(\delta) \to 0 \text{ as } \delta \to 0 .
$$

- **A-priori bound**: the restriction $\|\theta\|_Z \le M$ of the unknown to a set bounded in a stronger norm $Z$.
- **$\theta_1, \theta_2$**: two unknowns in that set; **$y_1, y_2$**: their data.
- **$X$, $Y$**: the norms in which the unknown and the data are measured.
- **$\omega$**: the modulus of continuity, the bound on the change of the estimate for a data change of size $\delta$.
- **Lipschitz**: $\omega(\delta) = C\delta$, the well-posed case.

Two forms of $\omega$ occur (Alessandrini et al., 2009; Isakov, 2017):

$$
\omega(\delta) = C\,\delta^{\mu} \quad \text{(Hölder)},
\qquad
\omega(\delta) = C\,|\log\delta|^{-\mu} \quad \text{(logarithmic)}.
$$

- **$\mu$**: the exponent, with $0 < \mu \le 1$ in the Hölder case.
- **Hölder stability**: the good case.
- **Logarithmic stability**: the bad case. Examples are the backward heat problem and the Cauchy problem for elliptic equations.

The exponent tells how many digits of the data survive into the estimate. With $\mu = 1/2$, four digits of data give two digits of parameter.

A scalar parameter in an otherwise known equation is the benign case:

- For the heat diffusivity, F01 found the spread of the estimate across noise draws to follow the Cramér–Rao bound.
- The map from data to $\alpha$ is therefore locally Lipschitz, with a constant the Fisher information gives. See [identifiability](identifiability.md).
- **Cramér–Rao bound**: the smallest standard deviation any unbiased estimator can reach from the given data.

**For training:** the type of $\omega$ sets the accuracy you can expect from the data, whatever the optimizer does.

## Regularization, and the physics as a regularizer

Classical inverse theory restores continuity by regularization (Engl, Hanke & Neubauer, 1996):

$$
\min_\theta \; \|\mathcal{H}u_\theta - y\|^2 + \lambda\, R(\theta).
$$

- **Data misfit**: the first term, the distance between predicted and measured observations.
- **$R(\theta)$**: the penalty, which encodes the a-priori bound.
- **$\lambda$**: the regularization weight, chosen from the noise level.

Among the many $\theta$ that fit the data equally, the penalty selects the one the prior prefers.

A PINN inverse problem has the same shape:

$$
\min_{w,\,\theta} \; \|\mathcal{H}u_w - y\|^2 + \lambda\, \|\mathcal{N}_\theta[u_w]\|^2 .
$$

- **Residual term**: $\|\mathcal{N}_\theta[u_w]\|^2$, the violation of the equation by the network.

It differs from the classical form in two ways:

1. The residual term regularizes the *state*. The network cannot fit noise with a function that violates the equation.
2. The parameter $\theta$ carries no explicit penalty. It is regularized only through the physics.

Consequence measured in F01 ([benchmarks](../benchmarks.md#identification-baseline-experiment-f01)):

- With a fixed data weight and few, noisy points, the estimate of $\alpha$ is biased low.
- Cause: the residual term prefers a smoother field, and a smoother field implies slower diffusion.

**For training:** set the data weight from the noise level. The [Bayesian view](bayesian.md) turns the weights into noise levels, which is the principled way to set $\lambda$.

## Stability of the estimate under disturbances

For a disturbance $\delta y$ of the data, the first-order response of a least-squares estimate is

$$
\delta\theta \approx (J^\top \Sigma^{-1} J)^{-1} J^\top \Sigma^{-1}\, \delta y, \qquad J = \frac{\partial \mathcal{H}u_\theta}{\partial\theta}, \qquad F = J^\top\Sigma^{-1}J .
$$

- **$\delta y$**: the disturbance of the data.
- **$\delta\theta$**: the resulting shift of the estimate.
- **$J$**: the sensitivity Jacobian, the derivative of the predicted observations with respect to the parameters.
- **$\Sigma$**: the covariance of the measurement noise.
- **$F$**: the Fisher information.

$F$ plays two roles:

- $F^{-1}$ is the covariance of the random errors.
- $F^{-1}J^\top\Sigma^{-1}$ is the gain on systematic errors (bias, drift, a stuck sensor).

The disturbance suite (item 4 of the v0.5 list in the [history](../history.md), experiments K05–K07, [benchmarks](../benchmarks.md#disturbance-and-stability-experiments-k05k07)) measures the observed $\delta\theta$ against this prediction:

| Experiment | Disturbance | Function |
|---|---|---|
| K05 | noise, bias, dropout and delay in the data | [`perturb_observations`](../api/analysis.md#analysis.disturbance.perturb_observations), [`linear_response`](../api/analysis.md#analysis.disturbance.linear_response) |
| K06 | perturbations of the problem: initial data, boundary data, a coefficient | [`field_response`](../api/analysis.md#analysis.disturbance.field_response) |
| K07 | error against residual | [`stability_ratio`](../api/analysis.md#analysis.disturbance.stability_ratio) |

- **Empirical stability constant**: the ratio of the observed shift to the predicted one.
- K05 result: for bias and delay the observed shift is 0.78–1.06 of the prediction.

**For training:** the shift a sensor fault causes can be computed from $J$ before any training.

## What a PINN adds and what it does not

The network adds its own failure of continuity:

- Training is non-convex.
- The surrogate reaches a residual that is small but not zero.

Two consequences matter for an estimate:

| Consequence | Evidence |
|---|---|
| The parameters absorb the surrogate's approximation error. More observations do not remove this bias. | F01: a floor of 0.1–0.2 % on clean data without weight decay and gradient clipping. With the packaged regularizers the floor was about 2 %, and D01 attributed that to the regularizers. |
| With the same data, different seeds can give different estimates when the problem is ill-conditioned. | The seed spread is reported next to the noise spread. F04 checks that the estimate does not depend on where it started. |

- **Surrogate**: the trained network $u_w$ used in place of the exact solution.
- **Seed spread**: the spread of the estimate across network initializations.

Neither is a property of the inverse problem. Both are properties of the solver, and the [numerical-analysis bridge](numerical_analysis.md) treats them as such.

**For training:** for an inverse run, start from `weight_decay=0` and `gradient_clipping=0` (F01), and report the seed spread.

## References

- Hadamard, J. (1902). Sur les problèmes aux dérivées partielles et leur signification physique. *Princeton University Bulletin* 13, 49–52.
- Engl, H. W., Hanke, M. & Neubauer, A. (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
- Alessandrini, G., Rondi, L., Rosset, E. & Vessella, S. (2009). The stability for the Cauchy problem for elliptic equations. *Inverse Problems* 25, 123004. [doi:10.1088/0266-5611/25/12/123004](https://doi.org/10.1088/0266-5611/25/12/123004)
- Isakov, V. (2017). *Inverse Problems for Partial Differential Equations*, 3rd ed. Springer. [doi:10.1007/978-3-319-51658-5](https://doi.org/10.1007/978-3-319-51658-5)
- Kaipio, J. & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
