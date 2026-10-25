# Well-posedness and stability

## Hadamard's three conditions

A problem is well posed when a solution exists, is unique, and depends continuously on the data (Hadamard, 1902). Forward problems for the equations in the [catalogue](../custom_equations.md#the-catalogue) are well posed on their domains: given $\theta$, the initial and boundary data determine $u$, and nearby data give nearby solutions.

Inverse problems are usually not. Recovering $\theta$ (or an initial state, or a coefficient field) from $y = \mathcal{H}u_\theta + \varepsilon$ inverts a map that smooths. Diffusion damps the mode $\sin k\pi x$ by $e^{-\alpha k^2 \pi^2 t}$, so reading the initial state from a late measurement amplifies noise at mode $k$ by $e^{\alpha k^2 \pi^2 t}$: continuity fails. Uniqueness can also fail, which is the subject of [identifiability](identifiability.md).

## Conditional stability

Ill-posedness is rarely total. Restrict the unknown to a set where it is bounded in a stronger norm (an a-priori bound $\|\theta\|_{Z} \le M$) and continuity returns, but with a modulus that is not Lipschitz:

$$
\|\theta_1 - \theta_2\|_X \le \omega\big(\|y_1 - y_2\|_Y\big), \qquad \omega(\delta) \to 0 \text{ as } \delta \to 0,
$$

with $\omega$ a power $\delta^\mu$ in the good cases (Hölder stability) and a logarithm $|\log\delta|^{-\mu}$ in the bad ones, such as the backward heat problem or the Cauchy problem for elliptic equations (Alessandrini et al., 2009; Isakov, 2017). The exponent tells how many digits of the data survive into the estimate: with Hölder exponent $\mu = 1/2$, four digits of data give two digits of parameter.

A scalar parameter in an otherwise known equation is the benign case. For the heat diffusivity, F01 found the spread of the estimate across noise draws to follow the Cramér–Rao bound, so the map from data to $\alpha$ is locally Lipschitz with a constant the Fisher information gives; see [identifiability](identifiability.md).

## Regularization, and the physics as a regularizer

Classical inverse theory restores continuity by regularization: minimize a data misfit plus a penalty that encodes the a-priori bound, $\|\mathcal{H}u_\theta - y\|^2 + \lambda R(\theta)$, and choose $\lambda$ from the noise level (Engl, Hanke & Neubauer, 1996). The penalty selects, among the many $\theta$ that fit the data equally, the one the prior prefers.

A PINN inverse problem has the same shape, with two differences. The residual term $\|\mathcal{N}_\theta[u_w]\|^2$ regularizes the *state*: the network cannot fit noise with a function that violates the equation. And the parameter $\theta$ carries no explicit penalty, so it is regularized only through the physics. This is why the fixed data weight of F01 produced a bias at few, noisy points: the residual term preferred a smoother field, and the smoother field implied slower diffusion. The [Bayesian view](bayesian.md) turns the weights into noise levels, which is the principled way to set $\lambda$.

## Stability of the estimate under disturbances

For a disturbance $\delta y$ of the data, the first-order response of a least-squares estimate is

$$
\delta\theta \approx (J^\top \Sigma^{-1} J)^{-1} J^\top \Sigma^{-1}\, \delta y, \qquad J = \frac{\partial \mathcal{H}u_\theta}{\partial\theta},
$$

so the Fisher information $F = J^\top\Sigma^{-1}J$ is both the covariance of random errors and the gain on systematic ones (bias, drift, a stuck sensor). The disturbance suite of the roadmap (item 4, experiments K05–K07) measures the observed $\delta\theta$ against this prediction for noise, bias, dropout and delay, and for perturbations of the initial and boundary data; the ratio is an empirical stability constant.

## What a PINN adds and what it does not

The network introduces its own failure to depend continuously on the data: training is non-convex, and the surrogate reaches a residual that is small but not zero. Two consequences matter for an estimate.

- The surrogate's approximation error is absorbed by the parameters. F01 measured this as a bias of about 2 % that no number of observations removes.
- Different seeds, same data, can give different estimates when the problem is ill-conditioned. The seed spread is therefore reported next to the noise spread, and F04 checks that the estimate does not depend on where it started.

Neither is a property of the inverse problem; both are properties of the solver, and the [numerical-analysis bridge](numerical_analysis.md) treats them as such.

## References

- Hadamard, J. (1902). Sur les problèmes aux dérivées partielles et leur signification physique. *Princeton University Bulletin* 13, 49–52.
- Engl, H. W., Hanke, M. & Neubauer, A. (1996). *Regularization of Inverse Problems*. Kluwer. [doi:10.1007/978-94-009-1740-8](https://doi.org/10.1007/978-94-009-1740-8)
- Alessandrini, G., Rondi, L., Rosset, E. & Vessella, S. (2009). The stability for the Cauchy problem for elliptic equations. *Inverse Problems* 25, 123004. [doi:10.1088/0266-5611/25/12/123004](https://doi.org/10.1088/0266-5611/25/12/123004)
- Isakov, V. (2017). *Inverse Problems for Partial Differential Equations*, 3rd ed. Springer. [doi:10.1007/978-3-319-51658-5](https://doi.org/10.1007/978-3-319-51658-5)
- Kaipio, J. & Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer. [doi:10.1007/b138659](https://doi.org/10.1007/b138659)
