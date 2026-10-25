# The numerical-analysis bridge

## Consistency, stability, convergence

For a linear well-posed evolution problem, a consistent discretization converges if and only if it is stable (Lax & Richtmyer, 1956). Consistency says the scheme's residual on the exact solution vanishes with the mesh; stability says the discrete solution operator is bounded uniformly in the mesh; convergence follows with the bound as the constant:

$$
\|u - u_h\| \le C_{\text{stab}}\, \|\text{residual of } u_h\|.
$$

For explicit time-stepping, stability is the CFL condition (Courant, Friedrichs & Lewy, 1928): the explicit heat scheme in `pinnrl-benchmark fdm` needs $\alpha\,\Delta t/\Delta x^2 \le 1/2$, and von Neumann analysis gives the same number by asking that no Fourier mode grow. For nonlinear problems the role of the stability operator is played by an energy estimate, a bound on a norm of the solution that the equation preserves or decreases.

## The same statement for a PINN

A PINN has no mesh, but the structure survives. Mishra & Molinaro (2022) prove, for a class of forward problems, an estimate of the form

$$
\|u - u_w\| \;\le\; C_{\text{stab}}\,\big(\,\mathcal{E}_{\text{train}} + \mathcal{E}_{\text{quad}}\,\big),
$$

with $\mathcal{E}_{\text{train}}$ the residual and boundary mismatch on the collocation points (what training minimizes), $\mathcal{E}_{\text{quad}}$ the quadrature error of replacing the integral of the residual by a sample mean, and $C_{\text{stab}}$ the conditional-stability constant of the PDE. A companion result covers a class of inverse problems (Mishra & Molinaro, 2022b), with the data misfit added to the right-hand side and the stability constant now that of the inverse problem, the modulus $\omega$ of [well-posedness](well_posedness.md). De Ryck & Mishra (2024) survey the estimates; Shin, Darbon & Karniadakis (2020) give the convergence statement for linear second-order equations.

Three things the package can measure follow.

1. **The training residual is not the error.** The ratio of the two is an estimate of $C_{\text{stab}}$. Experiments K06 and K07 record it across the catalogue, and compare it with the constants the conditional-stability theory gives where one is known. A constant in the hundreds for Allen–Cahn, and near one for the heat equation, would explain the B01 finding that the Allen–Cahn error stayed above 0.5 for every sampler at a small residual.
2. **Quadrature error scales with dimension.** A Monte Carlo sample mean of $N$ points has error $\mathcal{O}(N^{-1/2})$ in any dimension, but the constant grows with the variance of the residual over the domain, and the residual of a trained network is rough. In five dimensions the gallery of [custom equations](../custom_equations.md) needed four times the points for a worse result than in one, which is the quadrature term at work. Experiment K08 sweeps the dimension and reports both terms separately.
3. **Causality.** Time-marching schemes respect the direction of information; a PINN trained on all of space-time at once does not, and can settle on a wrong solution that has a small residual (Krishnapriyan et al., 2021). The metastable Allen–Cahn solution of B01 is such a case. Causal weighting and time-marching (H03, H04) are the PINN equivalents of marching in time.

## Linearized stability of coupled systems

For a system $u_t = f(u; \theta)$ (the ODE entries, the reaction terms of Gray–Scott or the Brusselator, a process model with several balances), the Jacobian $\partial f/\partial u$ at an operating point gives the linearized dynamics. Its eigenvalues decide whether perturbations decay, and their ratio is the stiffness, which bounds the time step of an explicit method and stiffens the PINN loss in the same way a high-order derivative does. Experiment K09 computes the spectrum for the catalogue systems and for the evaporator model, and the roadmap's process twins report it as a standard diagnostic.

## High dimensions

Both constants degrade with dimension: the quadrature constant through the variance of the residual, and the stability constant through the growth of the domain's boundary relative to its interior. Experiment K08 and paper P12 measure how. The maintainer's rule, applied to every system with many state variables, is to report the spectrum of the linearized dynamics and the two error terms before trusting a surrogate in that dimension.

## References

- Lax, P. D. & Richtmyer, R. D. (1956). Survey of the stability of linear finite difference equations. *Comm. Pure Appl. Math.* 9, 267–293. [doi:10.1002/cpa.3160090206](https://doi.org/10.1002/cpa.3160090206)
- Courant, R., Friedrichs, K. & Lewy, H. (1928). Über die partiellen Differenzengleichungen der mathematischen Physik. *Mathematische Annalen* 100, 32–74. [doi:10.1007/BF01448839](https://doi.org/10.1007/BF01448839)
- Mishra, S. & Molinaro, R. (2022). Estimates on the generalization error of physics-informed neural networks for approximating PDEs. *IMA J. Numer. Anal.* 43, 1–43. [doi:10.1093/imanum/drab093](https://doi.org/10.1093/imanum/drab093)
- Mishra, S. & Molinaro, R. (2022b). Estimates on the generalization error of physics-informed neural networks for approximating a class of inverse problems for PDEs. *IMA J. Numer. Anal.* 42, 981–1022. [doi:10.1093/imanum/drab032](https://doi.org/10.1093/imanum/drab032)
- De Ryck, T. & Mishra, S. (2024). Numerical analysis of physics-informed neural networks and related models in physics-informed machine learning. *Acta Numerica* 33, 633–713. [doi:10.1017/S0962492923000089](https://doi.org/10.1017/S0962492923000089)
- Shin, Y., Darbon, J. & Karniadakis, G. E. (2020). On the convergence of physics informed neural networks for linear second-order elliptic and parabolic type PDEs. *Commun. Comput. Phys.* 28, 2042–2074. [doi:10.4208/cicp.OA-2020-0193](https://doi.org/10.4208/cicp.OA-2020-0193)
- Krishnapriyan, A., Gholami, A., Zhe, S., Kirby, R. & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. *NeurIPS*.
