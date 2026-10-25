# The Bayesian view and uncertainty

## The loss is a negative log-posterior

Take Gaussian observation noise, $y_i = (\mathcal{H}u)_i + \varepsilon_i$ with $\varepsilon_i \sim \mathcal{N}(0, \sigma_i^2)$, and a Gaussian belief that the residual, the boundary and the initial mismatch are zero with scales $\sigma_r$, $\sigma_b$, $\sigma_0$. Up to a constant, minus the log of the posterior over $(w, \theta)$ is

$$
-\log p(w, \theta \mid y) = \frac{1}{2\sigma_r^2}\|\mathcal{N}_\theta[u_w]\|^2 + \frac{1}{2\sigma_b^2}\|\mathcal{B}u_w\|^2 + \frac{1}{2\sigma_0^2}\|u_w(\cdot, 0) - u_0\|^2 + \sum_i \frac{(y_i - (\mathcal{H}u_w)_i)^2}{2\sigma_i^2} - \log p(w, \theta).
$$

This is the PINN loss, with weights $1/\sigma^2$ (Kaipio & Somersalo, 2005; Stuart, 2010). Training finds the maximum-a-posteriori point, and the weights are not tuning knobs: the data weight is the inverse noise variance, and the physics weights encode how much model error one admits. Experiment K39 tests whether setting the data weight from the measured noise beats tuned weights, which F01 motivates: a fixed weight that ignores the noise level biased the estimate at few, noisy points.

Two things follow for pinnrl. A per-sensor noise level is a per-observation weight, which `fisher_information` already accepts. And a change of weights is a change of prior, so results obtained with different weights are not comparable without saying so.

## Sources of uncertainty

An error bar on $\theta$ must say which of these it covers.

| Source | Where it comes from | How it shows |
|---|---|---|
| Aleatoric | measurement noise $\varepsilon$ | the spread across noise draws; bounded below by $F^{-1}$ |
| Parametric epistemic | finite data, correlated parameters | the posterior width along sloppy directions |
| Surrogate | the network's approximation error and the non-convex training | the bias of F01, the seed spread |
| Model form | the equation is not the process | a residual structure the surrogate cannot remove (F08) |

The Fisher bound covers the first. Ensembles over seeds and over noise draws separate the first from the third. Model-form error is the one that no method internal to the model detects, and it is why a twin must report its residual against the plant.

## Approximating the posterior

- **Laplace.** Expand the loss to second order at the optimum; the posterior is Gaussian with covariance the inverse Hessian (MacKay, 1992). The block for $\theta$, after eliminating $w$, is the generalized Gauss–Newton approximation of $F^{-1}$. It reuses the sensitivities of [identifiability](identifiability.md), which is why the conditioning module comes first on the roadmap. Experiment K41 asks how much of the network the Hessian must cover.
- **Deep ensembles.** Retrain from several initializations and treat the estimates as posterior samples (Lakshminarayanan, Pritzel & Blundell, 2017). The spread captures the surrogate's non-convexity, which Laplace does not, and misses nothing that retraining reveals. Experiments F06 and K40 measure coverage against the ensemble size.
- **Sampling.** Hamiltonian Monte Carlo over $(w, \theta)$ (Yang, Meng & Karniadakis, 2021) is the reference posterior for small networks (K42) and too costly for the rest.
- **Conformal intervals.** Hold out sensors, calibrate the residual quantile on them, and get finite-sample coverage with no distributional assumption (Vovk, Gammerman & Shafer, 2005). It bounds the *field* prediction rather than $\theta$ (K44).

A broad comparison of these on physics-informed models is Psaros et al. (2023).

## Calibration

An interval is calibrated when it contains the truth as often as it claims: a 90 % interval, over many data sets, should cover 90 % of the time. Coverage alone is not enough, since an interval that is always the whole real line is calibrated. The quantities pinnrl reports, from F06 and K10 onward, are:

- **empirical coverage** of the nominal interval across synthetic data sets with known truth;
- **width**, the sharpness, which should be as small as coverage allows (Gneiting, Balabdaoui & Raftery, 2007);
- the **continuous ranked probability score**, a proper scoring rule that rewards both at once (Gneiting & Raftery, 2007).

A method is kept only where it is calibrated, and the page of each method states the regime in which that was checked.

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
