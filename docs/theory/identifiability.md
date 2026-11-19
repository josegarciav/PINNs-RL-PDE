# Identifiability and conditioning

This page states when sensor data determine the parameters of an equation, and how precisely. Read it before trusting a parameter a PINN estimated, or before placing sensors.

- **$\theta$**: the $p$ unknown parameters of the equation.
- **$u_\theta$**: the solution for those parameters.
- **$\mathcal{H}$**: the sensor operator; $\mathcal{H}u$ is what the sensors read from the field $u$.
- **Surrogate**: the trained network $u_w$, with weights $w$, that stands in for the solution.

## Structural identifiability

$$
\mathcal{H}u_{\theta_1} = \mathcal{H}u_{\theta_2} \ \text{on the sensor set} \quad\Longrightarrow\quad \theta_1 = \theta_2
$$

- **Structurally identifiable**: the noise-free observations determine $\theta$ uniquely (Bellman & Åström, 1970).
- It is a property of the equation, the sensor operator and the experiment, not of the data quality.

No amount of data identifies:

- two parameters that only ever appear as a product;
- a rate that only acts in a regime the sensors never see.

Small models are decided symbolically (Chiş, Banga & Balsa-Canto, 2011):

1. Differentiate the observations with respect to time.
2. Solve the resulting algebraic relations for $\theta$.

Experiment K11 (planned, not run yet) does this with SymPy for the catalogue entries that have a closed form.

**For a PINN:** no network or optimizer recovers a parameter that fails this test; change the sensors or the experiment.

## Practical identifiability

Identifiable in principle is not identifiable in practice.

$$
F = J^\top \Sigma^{-1} J, \qquad J_{ij} = \frac{\partial (\mathcal{H}u_\theta)_i}{\partial \theta_j}
$$

- **$J$** (sensitivity matrix): how much observation $i$ moves when parameter $j$ moves.
- **$\Sigma$**: the covariance of the noise, $\varepsilon \sim \mathcal{N}(0, \Sigma)$.
- **$F$** (Fisher matrix): the information the observations carry about $\theta$ near the truth.

$$
\operatorname{Cov}(\hat\theta) \succeq F^{-1}
$$

- **Cramér–Rao bound**: no unbiased estimator $\hat\theta$ has a covariance below $F^{-1}$ (Cramér, 1946).
- **Diagonal of $F^{-1}$**: its square root is the smallest standard deviation each parameter can have.
- **Correlations of $F^{-1}$**: which parameters the data confuse.
- **Practically identifiable**: the bound is small compared with the parameter itself, for the sensors and noise one actually has (Raue et al., 2009; Wieland et al., 2021).

[`pinnrl.analysis.closed_form_bound`](../api/analysis.md) computes $J$ by symbolic differentiation of a closed-form solution. It returns $F$, $F^{-1}$, the standard deviations, the correlation matrix and the eigen-decomposition.

Example, the heat diffusivity $\alpha$:

$$
\frac{\partial u}{\partial \alpha} = -\pi^2 t\, u
$$

- The information sits at late times and mid-domain.
- A probe at $x = 0.5$ carries more of it than one near the wall.
- F01: the spread of the PINN estimate across noise draws matched this bound to about 20 % once the noise dominated the surrogate's own error.

**For a PINN:** the bound is the best error bar these sensors allow, and the PINN reaches it only when its own error is below the noise.

## Sloppy directions and conditioning

$$
F = \sum_{k=1}^{p} \lambda_k\, v_k v_k^\top, \qquad \kappa(F) = \frac{\lambda_{\max}}{\lambda_{\min}}
$$

- **$v_k$** (eigenvector): a combination of parameters.
- **$\lambda_k$** (eigenvalue): how well the data determine that combination.
- **Stiff / sloppy direction**: a combination with a large / small $\lambda_k$.
- **$\kappa(F)$** (condition number): how much a relative error in the data can grow into a relative error along the sloppiest direction. It is the first diagnostic to report.

A spectrum spanning many decades, with a few stiff directions and many sloppy ones, is the rule in models with several parameters (Gutenkunst et al., 2007). F02 measures $\kappa(F)$ for the two KdV coefficients: amplitude and width enter through $3/\alpha$ and $\sqrt{\beta}$, so the data correlate them.

Two conditionings appear in a PINN inverse problem and should not be confused:

| Conditioning of | Comes from | What improves it |
|---|---|---|
| the *inverse problem* | $F$ | new sensors or a prior; no solver |
| *training* | the loss Hessian or the neural tangent kernel | optimizers and preconditioning (Wang, Teng & Perdikaris, 2021; Wang, Yu & Perdikaris, 2022) |

- **Loss Hessian**: the second derivatives of the training loss with respect to the weights.
- **Neural tangent kernel**: the matrix that sets how fast each error component decays in gradient training.
- The differential operator drives the training conditioning: high-order derivatives stiffen it.

Roadmap item 3 computes both (`pinnrl.analysis`, released; see [History](../history.md)). The second is the subject of K03 (done) and K04 (planned).

**For a PINN:** slow training and a poorly determined parameter are different problems; check $\kappa(F)$ first, because no optimizer fixes it.

## Profile likelihood

$$
\mathrm{PL}(\theta_j) = \min_{\theta_{k \ne j},\; w} \ \mathcal{L}(w, \theta)
$$

- **$\mathcal{L}$**: the loss.
- **$\mathrm{PL}(\theta_j)$** (profile): the smallest loss reachable with parameter $j$ held fixed.

The Fisher bound is local. When the posterior is far from Gaussian, or a direction is nearly flat, the profile is the honest check (Raue et al., 2009):

1. Fix one parameter on a grid of values.
2. Re-optimize everything else at each value.
3. Plot the minimal loss against the fixed value.

- **Flat profile**: the parameter is not practically identifiable.
- **Asymmetric profile**: the Gaussian error bar is wrong on one side.

F07 does this by retraining the PINN at each grid value.

**For a PINN:** the profile costs one training run per grid value and assumes no parabola.

## Sensitivities without a closed form

Most problems have no closed-form solution, and the surrogate's output does not depend on $\theta$ directly: the parameters enter through the residual. Three routes to $J$ remain, in order of cost.

1. **Finite differences of a numerical reference**, when one exists: F03 does this for Burgers with the Cole–Hopf solution at $\nu(1 \pm 1\,\%)$.
2. **The implicit-function theorem at the trained optimum** (formula below). This is experiment K01, and it also yields the Laplace approximation of the [Bayesian view](bayesian.md).
3. **Retraining** at perturbed $\theta$, the brute-force check of the other two.

$$
\frac{\partial w^*}{\partial \theta} = -H_{ww}^{-1} H_{w\theta}, \qquad J = \frac{\partial (\mathcal{H}u_w)}{\partial w}\, \frac{\partial w^*}{\partial \theta}
$$

- **$w^*(\theta)$**: the weights that minimize the loss for the parameters $\theta$.
- **$H_{ww}$, $H_{w\theta}$**: blocks of the loss Hessian $H$.
- **Hessian-vector product**: $H$ times a vector, computed without forming $H$; both blocks are applied this way.

Each route is validated against the exact layer on a catalogue entry with a closed form before it is trusted elsewhere.

**For a PINN:** route 2 gives $J$ from one trained network through [`pinnrl.analysis.surrogate_sensitivities`](../api/analysis.md).

## Field-valued unknowns

When the unknown is a function, $\alpha(x)$ or $\alpha(t)$, the parameter space is infinite-dimensional and the questions above are asked of its modes. For the heat equation in conservative form, $u_t = (\alpha(x)\, u_x)_x$:

$$
\delta u_t = (\alpha\, \delta u_x)_x + (\delta\alpha\, u_x)_x
$$

- **$\delta\alpha$**: a perturbation of the unknown field.
- **$\delta u$**: the first-order response of the solution.
- **$\delta\alpha\, u_x$**: the flux through which the perturbation acts.
- **Mode**: one component of the field in a basis.

Consequences:

- $\alpha$ is invisible wherever $u_x$ vanishes.
- A localized $\delta\alpha$ is diffused before it reaches the sensors.
- Modes of $\alpha$ that are oscillatory, or supported where the gradient is small, lie in the sloppy directions of the (now infinite) Fisher matrix.
- A time-varying $\alpha(t)$ is better placed: it changes the decay rate of every mode at once, and $\int_0^t \alpha$ is read directly from the amplitude of the solution.

The practical test is the one of the previous section with the function in place of the parameter. Compare the solution for the true field with the one for the best constant (or the best low-order polynomial) $\bar\alpha$:

$$
\|u_{\alpha} - u_{\bar\alpha}\| \lesssim \text{the surrogate's own field error} \quad\Longrightarrow\quad \text{the recovered field collapses to } \bar\alpha
$$

- This holds however expressive the network, even when the Fisher information of the data (with an exact forward model) says the modes are determined.
- The forward error, not the sensors, is the limit. The remedy is forward accuracy.

| Field (F09, heat) | Its solution vs the best constant's | Result |
|---|---|---|
| two-layer $\alpha(x)$ | 20 % away | recovered in shape |
| centred bump | reproduced to 4 % | not recovered |

- A smooth network turns a jump in $\alpha(t)$ into a ramp: the finite resolution of a surrogate with a few hundred weights.
- Reduced bases for the field (K48, since run; see [equivalence regions](equivalence_regions.md)) and pointwise intervals (K47, not built) are the next steps.

**For a PINN:** identify a field only when its solution differs from the best constant's by more than the network's forward error.

## References

- Bellman, R. & Åström, K. J. (1970). On structural identifiability. *Mathematical Biosciences* 7, 329–339. [doi:10.1016/0025-5564(70)90132-X](https://doi.org/10.1016/0025-5564(70)90132-X)
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- Chiş, O.-T., Banga, J. R. & Balsa-Canto, E. (2011). Structural identifiability of systems biology models: a critical comparison of methods. *PLoS ONE* 6, e27755. [doi:10.1371/journal.pone.0027755](https://doi.org/10.1371/journal.pone.0027755)
- Raue, A. et al. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25, 1923–1929. [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358)
- Gutenkunst, R. N. et al. (2007). Universally sloppy parameter sensitivities in systems biology models. *PLoS Computational Biology* 3, e189. [doi:10.1371/journal.pcbi.0030189](https://doi.org/10.1371/journal.pcbi.0030189)
- Wieland, F.-G., Hauber, A. L., Rosenblatt, M., Tönsing, C. & Timmer, J. (2021). On structural and practical identifiability. *Current Opinion in Systems Biology* 25, 60–69. [doi:10.1016/j.coisb.2021.03.005](https://doi.org/10.1016/j.coisb.2021.03.005)
- Wang, S., Teng, Y. & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43, A3055–A3081. [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043)
- Wang, S., Yu, X. & Perdikaris, P. (2022). When and why PINNs fail to train: a neural tangent kernel perspective. *J. Comput. Phys.* 449, 110768. [doi:10.1016/j.jcp.2021.110768](https://doi.org/10.1016/j.jcp.2021.110768)
