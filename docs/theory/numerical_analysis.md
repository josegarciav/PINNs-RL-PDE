# The numerical-analysis bridge

This page carries consistency, stability and convergence over from classical schemes to PINNs. Read it to understand how far the training residual can stand in for the error.

Notation is shared with the [theory index](index.md): exact solution $u$, network $u_w$, parameters $\theta$.

## Consistency, stability, convergence

For a linear well-posed evolution problem (Lax & Richtmyer, 1956):

$$
\text{consistency} + \text{stability} \iff \text{convergence},
\qquad
\|u - u_h\| \le C_{\text{stab}}\, \|\text{residual of } u_h\| .
$$

- **$u_h$**: the discrete solution on a mesh of size $h$.
- **Consistency**: the scheme's residual on the exact solution vanishes as the mesh is refined.
- **Stability**: the discrete solution operator is bounded uniformly in the mesh.
- **Convergence**: $u_h \to u$ as the mesh is refined.
- **$C_{\text{stab}}$**: the stability bound, which becomes the constant of the error estimate.

For explicit time-stepping, stability is the CFL condition (Courant, Friedrichs & Lewy, 1928). The explicit heat scheme in `pinnrl-benchmark fdm` needs

$$
\frac{\alpha\,\Delta t}{\Delta x^2} \le \frac{1}{2} .
$$

- **$\alpha$**: the diffusivity.
- **$\Delta t$, $\Delta x$**: the time step and the grid spacing.
- **CFL condition**: a bound on the time step in terms of the grid spacing.
- **Von Neumann analysis**: requires that no Fourier mode grow. It gives the same number.
- **Energy estimate**: a bound on a norm of the solution that the equation preserves or decreases. For nonlinear problems it plays the role of the stability operator.

**For training:** a small residual implies a small error only through a stability constant, and that constant belongs to the equation.

## The same statement for a PINN

A PINN has no mesh, but the structure survives. For a class of forward problems (Mishra & Molinaro, 2022):

$$
\|u - u_w\| \;\le\; C_{\text{stab}}\,\big(\,\mathcal{E}_{\text{train}} + \mathcal{E}_{\text{quad}}\,\big).
$$

- **Collocation point**: a point where the equation's residual is evaluated.
- **$\mathcal{E}_{\text{train}}$**: the residual and the boundary mismatch on the collocation points. Training minimizes it.
- **$\mathcal{E}_{\text{quad}}$**: the quadrature error of replacing the integral of the residual by a sample mean.
- **$C_{\text{stab}}$**: the conditional-stability constant of the PDE.

Related results:

- **Inverse problems** (Mishra & Molinaro, 2022b): the data misfit is added to the right-hand side. The constant is that of the inverse problem, the modulus $\omega$ of [well-posedness](well_posedness.md).
- **Survey**: De Ryck & Mishra (2024).
- **Convergence for linear second-order equations**: Shin, Darbon & Karniadakis (2020).

**For training:** the loss bounds the error only with the boundary and initial terms included, and only up to $C_{\text{stab}}$.

Three things the package can measure follow.

### 1. The training residual is not the error

$$
C = \frac{\text{relative } L^2 \text{ error}}{\text{RMS residual}} \;\approx\; C_{\text{stab}} .
$$

- **$C$**: the empirical stability constant, computed on fresh points by [`stability_ratio`](../api/analysis.md#analysis.disturbance.stability_ratio).

Measured in K07 ([benchmarks](../benchmarks.md#disturbance-and-stability-experiments-k05k07)); K06 measures the response of the trained field to a perturbed problem:

| Equation | $C$ (K07) | Reading |
|---|---|---|
| heat, convection, wave, Klein–Gordon, heat 2-D | 0.8–1.3 | the residual on fresh points is a usable error estimate |
| Poisson | 0.2–0.6 | the residual carries the scale of the source term |
| sine-Gordon | 0.2 | the residual carries a large nonlinear term |
| Burgers travelling wave | 0.4–0.5 | the error still tracks the residual |
| KdV soliton | 1.3–3.8 | the error still tracks the residual |
| Euler–Bernoulli beam | 0.8 to 3.9 | the error stalls at 4.5 % while the residual falls; the boundary and initial terms bind |

- Allen–Cahn is not in K07. A constant in the hundreds for Allen–Cahn, against one near 1 for heat, would explain the B01 finding: the Allen–Cahn error stayed above 0.5 for every sampler at a small residual.
- Where the conditional-stability theory gives a constant, the measured one is compared with it.

**For training:** evaluate the residual on points the optimizer has not seen, and read it next to the boundary and initial mismatches.

### 2. Quadrature error

$$
\mathcal{E}_{\text{quad}} = \mathcal{O}\big(N^{-1/2}\big) \quad \text{in any dimension}.
$$

- **$N$**: the number of collocation points in the Monte Carlo sample mean.
- The constant grows with the variance of the residual over the domain, and the residual of a trained network is rough.
- **Quadrature gap**: the relative difference between the mean-squared residual on a training pool and on a dense sample.

Measured in K08 ([benchmarks](../benchmarks.md#dimension-scaling-experiment-k08)), which sweeps the dimension and reports both terms separately:

| Points | Quadrature gap (K08) |
|---|---|
| 4,096, $d = 1 \ldots 5$ | 1–4 % in every dimension |
| 36 and 216 | 10–14 % |

The gap does not grow with dimension. A dimension costs through the terms listed under [High dimensions](#high-dimensions), as for the five-dimensional heat equation in the gallery of [custom equations](../custom_equations.md).

**For training:** the number of collocation points need not grow with the dimension; a few thousand keep the quadrature term at a few percent.

### 3. Causality

- **Causality**: information in an evolution problem moves forward in time. Time-marching schemes respect this.
- A PINN trained on all of space-time at once does not. It can settle on a wrong solution with a small residual (Krishnapriyan et al., 2021).
- The metastable Allen–Cahn solution of B01 is such a case.
- Causal weighting and time-marching (H03, H04) are the PINN equivalents of marching in time. H04 is in the [benchmarks](../benchmarks.md#time-marching-windows-experiment-h04); H03 is not run yet.

**For training:** when the residual is small and the solution is wrong, train in time windows with `PDETrainer.train_windows` ([API](../api/training.md)).

## Linearized stability of coupled systems

For a system of balances, linearize at an operating point $u^\ast$:

$$
u_t = f(u; \theta), \qquad A = \frac{\partial f}{\partial u}\bigg|_{u^\ast}, \qquad \delta u_t = A\,\delta u, \qquad \text{stiffness} = \frac{\max_i |\operatorname{Re}\lambda_i|}{\min_i |\operatorname{Re}\lambda_i|} .
$$

- **$f$**: the right-hand side. Examples: the ODE entries, the reaction terms of Gray–Scott or the Brusselator, a process model with several balances.
- **$A$**: the Jacobian at the operating point.
- **$\delta u$**: a perturbation of the state.
- **$\lambda_i$**: the eigenvalues of $A$. Perturbations decay when every $\operatorname{Re}\lambda_i < 0$.
- **Stiffness**: the ratio of the fastest to the slowest rate. It bounds the time step of an explicit method, and it stiffens the PINN loss in the same way a high-order derivative does.

Experiment K09 computes the spectrum for the catalogue systems and for the evaporator model ([benchmarks](../benchmarks.md#linearized-stability-experiment-k09)). Two cautions from it:

1. **The transport operator is non-normal.** An upwind scheme has one defective eigenvalue $-v/\Delta z$, so the computed spectrum is a pseudospectrum. The stiffness it suggests is the grid's, which a PINN never sees.
2. **On ODEs the PINN failed by the trivial solution, not by stiffness.** The failures (pendulum, SIR, Lorenz) were trivial solutions close to the initial condition.

- **Non-normal operator**: one whose eigenvectors are not orthogonal, so its eigenvalues are sensitive to rounding.
- **Pseudospectrum**: the eigenvalues of slightly perturbed copies of the operator, which is what the computation returns for a defective matrix.
- **$v$, $\Delta z$**: the transport velocity and the grid spacing along the unit.
- **Trivial solution**: a solution that satisfies the equation and ignores the initial condition, such as $u \equiv 0$.

The process twins therefore report the physical rates (residence, source) and the number of fast units in the window, not the eigenvalues of the discretised operator.

**For training:** before training on an ODE system, check whether a trivial solution lies near the initial condition; stiffness alone does not predict failure.

## High dimensions

K08 measures the terms of the error estimate in $d = 1 \ldots 5$ space dimensions at a fixed budget ([benchmarks](../benchmarks.md#dimension-scaling-experiment-k08)).

| Quantity (heat, K08) | $d = 1$ | $d = 5$ |
|---|---|---|
| Empirical constant $C$ | 0.8 | 8.4 |
| Initial error | 0.009 | 0.076 |
| Boundary error | 0.010 | 0.049 |
| Quadrature gap | 0.030 | 0.032 |
| Time per step | 14 ms | 122 ms |

- **The stability constant grows.** For heat, $C = 0.8, 1.0, 1.3, 2.0, 8.4$ for $d = 1 \ldots 5$. The interior residual stays at 2–7 %.
- **Cause 1, the conditions.** The boundary grows relative to the interior, and 512 fixed points cover the initial slab and the faces more thinly.
- **Cause 2, normalisation.** The norm of the product solution shrinks as $2^{-d/2}$, which inflates the relative error (a factor 4 at $d = 5$).
- **The quadrature term does not grow** at 4,096 points. The cost of a dimension is per point: one second-derivative pass per axis.
- **Convection fails from $d = 2$ at this budget.** At $d = 4, 5$ the network converges to $u \approx 0$, with a small residual and an unpaid initial condition, and $C$ is meaningless there (5,947).
- Paper P12 (planned) treats the scaling with dimension and with coupled fields.

The maintainer's rule, for every system with many state variables:

1. Report the spectrum of the linearized dynamics.
2. Report the two error terms, with the initial and boundary errors next to them.
3. Only then trust a surrogate in that dimension.

**For training:** in many dimensions, spend points and loss weight on the initial and boundary conditions, not on more interior collocation points.

## References

- Lax, P. D. & Richtmyer, R. D. (1956). Survey of the stability of linear finite difference equations. *Comm. Pure Appl. Math.* 9, 267–293. [doi:10.1002/cpa.3160090206](https://doi.org/10.1002/cpa.3160090206)
- Courant, R., Friedrichs, K. & Lewy, H. (1928). Über die partiellen Differenzengleichungen der mathematischen Physik. *Mathematische Annalen* 100, 32–74. [doi:10.1007/BF01448839](https://doi.org/10.1007/BF01448839)
- Mishra, S. & Molinaro, R. (2022). Estimates on the generalization error of physics-informed neural networks for approximating PDEs. *IMA J. Numer. Anal.* 43, 1–43. [doi:10.1093/imanum/drab093](https://doi.org/10.1093/imanum/drab093)
- Mishra, S. & Molinaro, R. (2022b). Estimates on the generalization error of physics-informed neural networks for approximating a class of inverse problems for PDEs. *IMA J. Numer. Anal.* 42, 981–1022. [doi:10.1093/imanum/drab032](https://doi.org/10.1093/imanum/drab032)
- De Ryck, T. & Mishra, S. (2024). Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning. *Acta Numerica* 33, 633–713. [doi:10.1017/S0962492923000089](https://doi.org/10.1017/S0962492923000089)
- Shin, Y., Darbon, J. & Karniadakis, G. E. (2020). On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type PDEs. *Commun. Comput. Phys.* 28, 2042–2074. [doi:10.4208/cicp.OA-2020-0193](https://doi.org/10.4208/cicp.OA-2020-0193)
- Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R. & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*.
