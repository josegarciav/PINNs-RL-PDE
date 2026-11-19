# Nonlinear uncertainty: equivalence regions and swarm sampling

This page defines the set of all parameters that fit the data within the noise, and the particle-swarm method that samples it. Read it when a Gaussian error bar is not enough: several parameters, a nonlinear model, or a best fit that may not be the truth.

- The Fisher bound and the Laplace approximation of the [Bayesian view](bayesian.md) are *linear* statements: an ellipse around one estimate.
- Fernández-Martínez, García-Gonzalo and co-workers (University of Oviedo), working on geophysical inverse problems, argue that the object to report is the **region of equivalence**, and that it is sampled rather than linearized. Their nine papers shape the design of [`pinnrl.uq`](../api/uq.md).

## The region of equivalence

$$
d_{\text{obs}} = F(m_{\text{true}}) + \delta d, \qquad
M_{\mathrm{tol}} = \{\, m : \|F(m) - d_{\text{obs}}\| \le \mathrm{tol} \,\}
$$

- **$m$**: the model, that is the vector of unknown parameters ($\theta$ on the other pages).
- **$F$** (forward map): takes parameters to predicted observations. On this page $F$ is not the Fisher matrix.
- **$d_{\text{obs}}$**, **$\delta d$**: the observations and their noise.
- **Misfit**: the distance $\|F(m) - d_{\text{obs}}\|$ between prediction and data.
- **$M_{\mathrm{tol}}$** (region of equivalence): every model whose misfit is within a tolerance $\mathrm{tol}$ set by the noise.

The solution of an inverse problem is this set, not its minimum-misfit member (Pallero et al., 2018, after Tarantola). Three facts carry over to PINN identification unchanged.

### 1. The best-fit point is not the truth

$$
\nabla c(m_0) = 2 J^\top \delta d \ne 0
$$

- **$m_0$**: the minimizer of the noisy misfit.
- **$c$**: the noise-free misfit.
- **$J$**: the Jacobian of $F$ (the sensitivity matrix of [identifiability](identifiability.md)).

The true model lies inside $M_{\mathrm{tol}}$ with a *higher* misfit than the best fit. Reporting the minimizer alone reports the noise.

### 2. The linearized region is only local

- **Linearized region**: the ellipse from $J$ at $m_0$, the Fisher ellipse of [identifiability](identifiability.md). It matches $M_{\mathrm{tol}}$ near $m_0$ and nowhere else.
- For a nonlinear forward map the true region is a curved valley (the "banana"), elongated along the small singular values of $J$, possibly disconnected.
- The ellipse both includes models outside the region and misses models inside it.
- F02 is such a case in `pinnrl`: the PINN's optimum for the two KdV coefficients sits 20 % from the truth with a tight Fisher ellipse around it.

### 3. Noise and regularization deform the region

- **Noise** shrinks the low-misfit part of the valley and may move the best fit into a different basin (Fernández-Martínez et al., 2014a, b).
- **Tikhonov term**: a penalty on the distance to a reference model. It bounds the region anisotropically, most along the directions of highest uncertainty, and can create local minima.
- **For a PINN the physics weights play the role of the regularization.** F07 found that the soft physics term flattens the data-misfit profile 2.5–12×, which is the same deformation seen from inside the loss.

**For a PINN:** report the set of parameters that fit, with the trained estimate as one member; the estimate alone carries the noise and the physics weight.

## Sampling while optimizing

Particle swarms in their *exploratory* form sample $M_{\mathrm{tol}}$. Each particle moves under

$$
v_i^{k+1} = \omega\, v_i^k + \phi_1 (g^k - x_i^k) + \phi_2 (l_i^k - x_i^k), \qquad x_i^{k+1} = x_i^k + v_i^{k+1}
$$

- **$x_i^k$**, **$v_i^k$**: position (a parameter vector) and velocity of particle $i$ at iteration $k$.
- **$g^k$**, **$l_i^k$**: the best position found by the swarm and by particle $i$.
- **$\phi_1, \phi_2$**: random accelerations; $\phi = \phi_1 + \phi_2$ and $\bar\phi$ is its mean.
- **$\omega$**: the inertia.
- **Sample**: every evaluated position with misfit below the tolerance is kept.

The method needs only forward solves: no gradient, no inverse operator, no regularization and no reference model.

### PSO is a damped mass–spring system

$$
x^{k+1} + (\phi - \omega - 1)\, x^k + \omega\, x^{k-1} = \phi_1 g^k + \phi_2 l^k
\quad\longleftrightarrow\quad
x'' + (1 - \omega)\, x' + \phi\, (x - o) = 0
$$

- **Left**: the stochastic second-order difference equation each particle obeys.
- **Right**: the continuous system it discretizes, with damping $1 - \omega$ and stiffness $\phi$ (Fernández-Martínez & García-Gonzalo, 2008, 2011).
- **$o$**: the stochastic centre of attraction, between $g$ and $l$.
- **Generalized PSO**: the same discretization with a time step $\Delta t \ne 1$. Larger steps explore, smaller steps freeze the swarm.

The step `pinnrl.uq` takes ([`sample_equivalence_region`](../api/uq.md)), with an inertia $\omega_i$ and a time step $\Delta t_i$ per particle (the "cloud", below):

$$
v_i^{k+1} = \big(1 - (1 - \omega_i)\Delta t_i\big)\, v_i^k + \Delta t_i \big[\phi_1 (g^k - x_i^k) + \phi_2 (l_i^k - x_i^k)\big], \qquad x_i^{k+1} = x_i^k + \Delta t_i\, v_i^{k+1}
$$

### A family from the discretization

Centering or delaying the velocity and acceleration differences gives CC-PSO, CP-PSO, PP-PSO and RR-PSO (Fernández-Martínez & García-Gonzalo, 2009, 2012; García-Gonzalo et al., 2014). All are linearly isomorphic for the same delay. They differ in convergence rate and exploration.

| Variant | Behaviour |
|---|---|
| PSO, CC-PSO | converge fastest |
| CP-PSO, PP-PSO | the most exploratory; the recommended samplers |
| RR-PSO | best on benchmarks, with its good parameters on the line $\phi = 3(\omega - 3/2)$ |

### Stability regions replace tuning

$$
|\omega| < 1, \qquad 0 < \bar\phi < 2(\omega + 1)
$$

- **First-order stability**: the mean trajectory of a particle stays bounded inside this region.
- **Second-order stability**: the variance stays bounded. Its region is smaller.
- **USL curve**: the upper border of the second-order region. It is given analytically for any distribution of $(\omega, \phi_1, \phi_2)$ with finite first moments (García-Gonzalo & Fernández-Martínez, 2014).

For the classical PSO with uniformly distributed accelerations, the border that [`second_order_stable`](../api/uq.md) checks is

$$
\bar\phi < \frac{12\,(1 - \omega^2)}{7 - 5\,\omega}
$$

- Good parameters lie just inside that border, where the variance is large and successive positions are uncorrelated: *high variance and temporal uncorrelation improve exploration of ill-posed problems* (2008).
- **Cloud versions**: every particle gets its own parameters drawn near the border, and its own time step. Nothing is tuned by hand.
- Most published PSO variants are unnecessary once the stability analysis is understood (García-Gonzalo & Fernández-Martínez, 2012; Fernández-Martínez & García-Gonzalo, 2013).

### Collapse is the failure mode

$$
D^k = \frac{\operatorname{median}_i \|x_i^k - \bar{x}^k\|}{\operatorname{median}_i \|x_i^0 - \bar{x}^0\|}
$$

- **$D^k$** (dispersion): the median distance of the particles to their centre $\bar{x}^k$, relative to its initial value. It is monitored every iteration.
- **Collapse**: $D^k$ falls toward zero and the swarm stops exploring.

When it collapses, one of three things is done:

1. The sampling stops.
2. Exploration is raised (larger $\Delta t$, repulsive forces).
3. The collapsed particles are counted as one sample.

The sampled set is an *approximation* of the posterior, not a posterior sample. Whether and when PSO performs importance sampling is stated as the open question (2013).

### Posterior statistics by counting

From the retained samples (Pallero et al., 2018):

- min–max ranges per parameter;
- a mean or median model with its interquartile range;
- percentiles and empirical CDFs of any derived quantity;
- histograms that reveal multimodality.

Their example is a 300-parameter gravity inversion with 40,000 forward solves and 25,429 retained models.

### Model reduction before sampling

- An isotropic search space cannot be sampled efficiently beyond about ten dimensions.
- **Reduced basis**: a small set of modes in which a field-valued unknown is expanded. The modes are principal components of prior scenarios, or the singular vectors of a model already in the low-misfit region, plus one high-frequency term.
- The swarm moves in the coefficients (Pallero et al., 2018; Fernández-Martínez, 2015).

**For a PINN:** the swarm needs thousands of forward solves, so it needs a forward map far cheaper than a training run.

## What this changes in `pinnrl.uq`

The UQ module was planned as a likelihood-based data term, deep ensembles, a Laplace approximation and calibration metrics. The module is released (item 5 in [History](../history.md)), with the swarm sampler and the parametric surrogate next to the Laplace approximation.

### 1. Report regions, not points

$$
\mathrm{tol}^2 = (1 + \varepsilon)\, N \sigma^2
$$

- **$N$**: the number of observations; **$\sigma$**: the noise standard deviation (Gaussian noise).
- **$\varepsilon$**: the margin above the expected misfit of the truth, $N\sigma^2$ ([`misfit_tolerance`](../api/uq.md), default 0.1).

`sample_equivalence_region` uses the likelihood-ratio region when no tolerance is given:

$$
\{\, \theta : c(\theta) \le c_{\min} + \sigma^2\, \chi^2_p(\text{level}) \,\}
$$

- **$c(\theta)$**: the sum of squared residuals; **$c_{\min}$**: the best misfit found.
- **$\chi^2_p(\text{level})$**: the chi-square quantile with $p$ degrees of freedom, $p$ the number of parameters (tabulated for $p \le 5$).
- For Gaussian noise this is the asymptotic confidence region at that level, directly comparable with the Fisher ellipse.

The Fisher ellipse and the Laplace Gaussian are reported next to the set as the linear approximation, so the two can disagree visibly (the F02 case).

### 2. A swarm sampler over the parameters

For $p \lesssim 10$ parameters a cloud CP-PSO over $\theta$ is the sampler. It needs a cheap forward map for each $\theta$, and for a PINN a forward solve is a training run. Two routes:

| Route | What it is |
|---|---|
| **Parametric surrogate** $u_w(x, t;\, \theta)$ | $\theta$ is a network input; trained once over the prior range and evaluated in microseconds. Built first. |
| **Warm-started retraining** | a few hundred steps from the nearest particle's weights (the F07 profile, generalized to $p$ dimensions and to a moving swarm) |

The test cases are the KdV pair and the Burgers viscosity, since their regions are known to be displaced from the truth.

- **K45**: one parametric surrogate is accurate to 0.2–0.3 % over the range on heat and Burgers and 0.2–2 % on KdV, at a quarter to a half of the cost of five dedicated networks, and the swarm on its misfit reproduces the closed-form region ([benchmarks](../benchmarks.md#parametric-surrogate-as-the-forward-emulator-experiment-k45)).
- **K46**: on closed-form misfits (heat, KdV, saturation) the regions are ellipses at these noise levels, and the swarm under-samples their edges (width 0.6–0.8 of the ellipse). On the PINN's own misfit the region is the same as on the closed form ([benchmarks](../benchmarks.md#calibrated-uncertainty-experiments-f06-k10-k39-k46)).

### 3. A tolerance from the noise model, and the physics weight as regularization

- The tolerance is set by $\sigma$ (K39).
- The dependence of the region on the physics weight is measured and documented, since it plays the role of the Tikhonov term.
- **Too strong**: the region shrinks anisotropically toward what the surrogate can represent.
- **Too weak**: the data are fitted by a field that violates the equation.

### 4. Reduced bases for fields

- When the unknown is a coefficient field or a closure (roadmap v0.6, released), it is parametrized by fewer than ten modes and the swarm samples the coefficients.
- **Karhunen–Loève modes**: the principal directions of a random field's covariance; an alternative to principal components of prior scenarios.
- This is also the bridge to the measurement-design work: the modes the data do not constrain are the ones a new sensor should.
- **K48** (the F09 fields in a cosine basis of 8 modes): the Fisher spectrum from the reference solver says 200 observations at noise 0.01 determine every mode to < 0.006, so the bump's collapse in F09 is the surrogate's limit, not the data's. A 4-mode swarm under-samples the likelihood region (0.2–0.7 of the Fisher half-widths), and the truncated basis's best fit sits 10 Fisher half-widths from the truth's projection ([benchmarks](../benchmarks.md#reduced-bases-for-field-unknowns-experiment-k48)).

### 5. Ensembles are a swarm without dynamics

- Deep ensembles over seeds are a crude sample of $M_{\mathrm{tol}}$. The papers' dispersion diagnostic and the "collapsed particles count once" rule apply to them ([`ensemble_summary`](../api/uq.md)).
- K10 (with F06): ensembles over seeds cover 13 %, Laplace 77 %, the physics region 90 %. The reference posterior by HMC (K42) is not run yet.
- The open question of when a swarm sample is a posterior sample is answered empirically in K47, by weighting retained samples:

$$
w_s \propto \exp\!\left(-\frac{\text{misfit}_s}{2\sigma^2}\right)
$$

- **K47** (heat $\alpha$, 30 data sets; pendulum $(k, b)$, 10), against the exact posterior on a grid: the weighted set is centred right but 0.5–0.7 as wide, with coverage 80–87 %; the accepted set covers 90–100 % ([benchmarks](../benchmarks.md#the-swarm-as-a-posterior-sample-experiment-k47)). This holds only where the cloud covers the support.

### 6. Optimizer stability from the same mathematics

- The second-order (variance) stability regions of a stochastically forced damped oscillator are the same analysis that sets a safe step for a momentum optimizer under gradient noise.
- D01/D02 can use it to choose the learning rate and momentum from the Hessian spectrum of K03 rather than by trial.

**For a PINN:** at the noise levels tested the region is the Fisher ellipse; the swarm confirms it but under-samples its edges, and the weighted set is too narrow to serve as a posterior.

## References

- Pallero, J. L. G., Fernández-Muñiz, M. Z., Cernea, A., Álvarez-Machancoses, Ó., Pedruelo-González, L. M., Bonvalot, S. & Fernández-Martínez, J. L. (2018). Particle swarm optimization and uncertainty assessment in inverse problems. *Entropy* 20, 96. [doi:10.3390/e20020096](https://doi.org/10.3390/e20020096)
- Fernández-Martínez, J. L. & García-Gonzalo, E. (2008). The generalized PSO: a new door to PSO evolution. *Journal of Artificial Evolution and Applications* 2008, 861275. [doi:10.1155/2008/861275](https://doi.org/10.1155/2008/861275)
- Fernández-Martínez, J. L. & García-Gonzalo, E. (2009). The PSO family: deduction, stochastic analysis and comparison. *Swarm Intelligence* 3, 245–273. [doi:10.1007/s11721-009-0034-8](https://doi.org/10.1007/s11721-009-0034-8)
- Fernández-Martínez, J. L. & García-Gonzalo, E. (2011). Stochastic stability analysis of the linear continuous and discrete PSO models. *IEEE Transactions on Evolutionary Computation* 15, 405–423. [doi:10.1109/TEVC.2010.2053935](https://doi.org/10.1109/TEVC.2010.2053935)
- Fernández-Martínez, J. L. & García-Gonzalo, E. (2012). Stochastic stability and numerical analysis of two novel algorithms of the PSO family: PP-GPSO and RR-GPSO. *International Journal on Artificial Intelligence Tools* 21, 1240011. [doi:10.1142/S0218213012400118](https://doi.org/10.1142/S0218213012400118)
- García-Gonzalo, E. & Fernández-Martínez, J. L. (2012). A brief historical review of particle swarm optimization (PSO). *Journal of Bioinformatics and Intelligent Control* 1, 3–16. [doi:10.1166/jbic.2012.1002](https://doi.org/10.1166/jbic.2012.1002)
- Fernández-Martínez, J. L. & García-Gonzalo, E. (2013). Particle swarm optimisation: time for uniformisation. *International Journal of Computing Science and Mathematics* 4, 16–33. [doi:10.1504/IJCSM.2013.054671](https://doi.org/10.1504/IJCSM.2013.054671)
- García-Gonzalo, E., Fernández-Martínez, J. L. & Cernea, A. (2014). Four-points particle swarm optimization algorithms. *Journal of Multiple-Valued Logic and Soft Computing* 22, 239–266.
- García-Gonzalo, E. & Fernández-Martínez, J. L. (2014). Convergence and stochastic stability analysis of particle swarm optimization variants with generic parameter distributions. *Applied Mathematics and Computation* 249, 286–302. [doi:10.1016/j.amc.2014.10.066](https://doi.org/10.1016/j.amc.2014.10.066)
- Fernández-Martínez, J. L., Pallero, J. L. G., Fernández-Muñiz, Z. & Pedruelo-González, L. M. (2014a, b). The effect of noise and Tikhonov's regularization in inverse problems. Parts I and II. *Journal of Applied Geophysics* 108, 176–185 and 186–193. (Cited through Pallero et al., 2018.)
- Fernández-Martínez, J. L. (2015). Model reduction and uncertainty analysis in inverse problems. *The Leading Edge* 34, 1006–1016. (Cited through Pallero et al., 2018.)
