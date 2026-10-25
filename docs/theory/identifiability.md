# Identifiability and conditioning

## Structural identifiability

Parameters are structurally identifiable when the noise-free observations determine them uniquely: $\mathcal{H}u_{\theta_1} = \mathcal{H}u_{\theta_2}$ on the sensor set implies $\theta_1 = \theta_2$ (Bellman & Åström, 1970). It is a property of the equation, the sensor operator and the experiment, not of the data quality. Two parameters that only ever appear as a product are not structurally identifiable from any amount of data; a rate that only acts in a regime the sensors never see is not identifiable either.

For small models, structural identifiability is decided symbolically (Chiş, Banga & Balsa-Canto, 2011): the observations are differentiated with respect to time and the resulting algebraic relations are solved for $\theta$. Experiment K11 does this with SymPy for the catalogue entries that have a closed form.

## Practical identifiability

Identifiable in principle is not identifiable in practice. With noise $\varepsilon \sim \mathcal{N}(0, \Sigma)$, the information the observations carry about $\theta$ near the truth is the Fisher matrix

$$
F = J^\top \Sigma^{-1} J, \qquad J_{ij} = \frac{\partial (\mathcal{H}u_\theta)_i}{\partial \theta_j},
$$

and the Cramér–Rao inequality bounds the covariance of any unbiased estimator from below by $F^{-1}$ (Cramér, 1946). Its diagonal gives the smallest standard deviation each parameter can have; its correlations say which parameters the data confuse. A parameter is practically identifiable when its bound is small compared with the parameter itself, for the sensors and the noise one actually has (Raue et al., 2009; Wieland et al., 2021).

`pinnrl.analysis.closed_form_bound` computes $J$ by symbolic differentiation of a closed-form solution and returns $F$, $F^{-1}$, the standard deviations, the correlation matrix and the eigen-decomposition. For the heat diffusivity, $\partial u/\partial\alpha = -\pi^2 t\, u$: the information sits at late times and mid-domain, and a probe at $x = 0.5$ carries more of it than one near the wall. F01 found the spread of the PINN estimate across noise draws to match this bound to about 20 % once the noise dominated the surrogate's own error.

## Sloppy directions and conditioning

The eigenvectors of $F$ are parameter combinations; the eigenvalues say how well the data determine each. A spectrum spanning many decades, with a few stiff directions and many sloppy ones, is the rule rather than the exception in models with several parameters (Gutenkunst et al., 2007). The condition number $\lambda_{\max}/\lambda_{\min}$ of $F$ is then the first diagnostic to report: it tells how much a relative error in the data can grow into a relative error along the sloppiest direction. Experiment F02 measures it for the two KdV coefficients, whose amplitude and width enter through $3/\alpha$ and $\sqrt{\beta}$ and are therefore correlated by the data.

Two different conditionings appear in a PINN inverse problem and should not be confused:

- the conditioning of the *inverse problem*, from $F$, which no solver can improve without new sensors or a prior;
- the conditioning of *training*, from the loss Hessian or the neural tangent kernel, which the differential operator drives (high-order derivatives stiffen it) and which optimizers and preconditioning can improve (Wang, Teng & Perdikaris, 2021; Wang, Yu & Perdikaris, 2022).

Roadmap item 3 computes both; the second is the subject of experiments K03 and K04.

## Profile likelihood

The Fisher bound is local. When the posterior is far from Gaussian, or when a direction is nearly flat, the profile likelihood is the honest check (Raue et al., 2009): fix one parameter on a grid of values, re-optimize everything else, and plot the minimal loss against the fixed value. A flat profile means the parameter is not practically identifiable; an asymmetric one means the Gaussian error bar is wrong on one side. Experiment F07 does this by retraining the PINN at each grid value.

## Sensitivities without a closed form

Most problems of interest have no closed-form solution, and the surrogate's output does not depend on $\theta$ directly: the parameters enter through the residual. Three routes to $J$ remain, in order of cost.

1. **Finite differences of a numerical reference**, when one exists: F03 does this for Burgers with the Cole–Hopf solution at $\nu(1 \pm 1\,\%)$.
2. **The implicit-function theorem at the trained optimum**: $\partial u_w/\partial\theta = -H_{ww}^{-1} H_{w\theta}$ with $H$ the loss Hessian, computed with Hessian-vector products. This is experiment K01, and it also yields the Laplace approximation of the [Bayesian view](bayesian.md).
3. **Retraining** at perturbed $\theta$, the brute-force check of the other two.

Each is validated against the exact layer on a catalogue entry with a closed form before it is trusted elsewhere.

## References

- Bellman, R. & Åström, K. J. (1970). On structural identifiability. *Mathematical Biosciences* 7, 329–339. [doi:10.1016/0025-5564(70)90132-X](https://doi.org/10.1016/0025-5564(70)90132-X)
- Cramér, H. (1946). *Mathematical Methods of Statistics*. Princeton University Press.
- Chiş, O.-T., Banga, J. R. & Balsa-Canto, E. (2011). Structural identifiability of systems biology models: a critical comparison of methods. *PLoS ONE* 6, e27755. [doi:10.1371/journal.pone.0027755](https://doi.org/10.1371/journal.pone.0027755)
- Raue, A. et al. (2009). Structural and practical identifiability analysis of partially observed dynamical models by exploiting the profile likelihood. *Bioinformatics* 25, 1923–1929. [doi:10.1093/bioinformatics/btp358](https://doi.org/10.1093/bioinformatics/btp358)
- Gutenkunst, R. N. et al. (2007). Universally sloppy parameter sensitivities in systems biology models. *PLoS Computational Biology* 3, e189. [doi:10.1371/journal.pcbi.0030189](https://doi.org/10.1371/journal.pcbi.0030189)
- Wieland, F.-G., Hauber, A. L., Rosenblatt, M., Tönsing, C. & Timmer, J. (2021). On structural and practical identifiability. *Current Opinion in Systems Biology* 25, 60–69. [doi:10.1016/j.coisb.2021.03.005](https://doi.org/10.1016/j.coisb.2021.03.005)
- Wang, S., Teng, Y. & Perdikaris, P. (2021). Understanding and mitigating gradient flow pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43, A3055–A3081. [doi:10.1137/20M1318043](https://doi.org/10.1137/20M1318043)
- Wang, S., Yu, X. & Perdikaris, P. (2022). When and why PINNs fail to train: a neural tangent kernel perspective. *J. Comput. Phys.* 449, 110768. [doi:10.1016/j.jcp.2021.110768](https://doi.org/10.1016/j.jcp.2021.110768)
