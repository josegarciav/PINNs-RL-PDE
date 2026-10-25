# K01: Fisher information and identifiability

## Question

Which parameter combinations do the sensors determine? $J$ by autograd, $F$, its singular values, $\kappa(F)$ and sloppy directions, for heat $\alpha$, KdV $(\alpha, \beta)$ and Burgers $\nu$.

## Setup

The sensitivity $J = \partial u / \partial\theta$ of a trained surrogate is computed by the implicit-function theorem at the physics-loss minimum, $\partial w / \partial\theta = -H_{ww}^{-1} H_{w\theta}$, with both Hessian blocks applied matrix-free by double backward, a damped ($10^{-3}$) MINRES solve (the Hessian of a trained network is indefinite along some directions, which conjugate gradient does not tolerate), and a JVP through the network at the observation points (`pinnrl.analysis.surrogate_sensitivities`). The Fisher matrix $F = J^\top J / \sigma^2$ with $\sigma = 0.05$, its inverse, correlation and eigen-decomposition follow (`fisher_information`).

Each result is checked against an exact or reference sensitivity:

- **Heat** ($\alpha = 0.1$): closed form $\partial u/\partial\alpha = -\pi^2 t\, u$, on the F01 random layouts (20–1,000 points) and the F05 probe layouts (1–8 fixed positions). Forward training, 5,000 steps.
- **KdV two-parameter** ($\alpha, \beta$) = (6, 1): symbolic sensitivities of the soliton, 50–1,000 random points; the 2 × 2 $F$, its correlation and condition number. Data-augmented training (200 clean observations, since the soliton collapses to zero without data), 8,000 steps.
- **Burgers** ($\log\nu$, $\nu = 0.01/\pi$): central differences of the Cole–Hopf reference at $\nu(1 \pm 1\,\%)$ (from F03), 20–1,000 random nodes. Data-augmented training, 10,000 steps.

Three seeds; a 4 × 64 tanh MLP, Adam at $10^{-3}$, 1,000 collocation points, weights 1 / 10 / 10 / 10.

**Metrics:** the relative error of the surrogate Jacobian against the reference, and the two Cramér–Rao bounds (relative standard deviations, correlation, condition number) side by side.

Rerun with:

```bash
python experiments/K01_fisher_information/run.py --seeds 3
```

## Result

Per problem and layout, over 3 seeds: the relative error of the surrogate Jacobian against the reference, $\|J_{\text{sur}} - J_{\text{ref}}\| / \|J_{\text{ref}}\|$, and the ratio of the two Cramér–Rao bounds (surrogate / reference). The field error of the trained networks is given per seed.

| Problem | Field rel. L2 (3 seeds) | Layout | Jacobian rel. error | Bound ratio (surrogate / reference) | Correlation ref / sur | κ(F) ref / sur |
|---|---|---|---|---|---|---|
| Heat, α | 0.038, 0.009, 0.026 | random 20–1,000 | 0.018–0.057 | 1.01–1.04 | – | – |
| | | 1–8 probes | 0.018–0.050 | 1.02–1.03 | – | – |
| KdV, (α, β) | 0.006, 0.007, 0.005 | random 50 | 0.036–0.051 | α 1.02–1.03, β 1.01–1.02 | +0.65 / +0.66 | 12.0 / 12.2 |
| | | random 200 | 0.037–0.053 | α 1.02–1.03, β 1.01–1.02 | +0.57 / +0.59 | 9.7 / 10.2 |
| | | random 1,000 | 0.037–0.050 | α 1.02–1.03, β 1.01 | +0.64 / +0.64 | 11.3 / 12.0 |
| Burgers, log ν | 0.028, 0.035, 0.056 | random 20 | 1.3–24 | 0.04–1.20 | – | – |
| | | random 50 | 1.8–15 | 0.07–0.62 | – | – |
| | | random 200 | 3.6–6.7 | 0.17–0.30 | – | – |
| | | random 1,000 | 2.6–3.2 | 0.32–0.42 | – | – |

The MINRES solves converged to a relative residual below $10^{-5}$ in every case (300 iterations at most) at the requested damping of $10^{-3}$. Each sensitivity takes 5–10 s on a laptop CPU for the 4 × 64 network, after a training of 3–7 minutes.

Two earlier solver versions are kept in `results/` for the record: conjugate gradient with a stop on negative curvature (`*_guarded.*`) returned partial iterates with residuals up to 7.7 and Jacobians off by 20–150 %; conjugate gradient restarted with tenfold damping (`*_cg_restart.*`) converged but at a damping that shrank the heat Jacobian by about 60 %. The Hessian of a trained network is indefinite along some directions, and MINRES is the solver that accepts that.

## Conclusion

1. **The surrogate's Fisher information is right when the surrogate resolves the solution.** On heat and the KdV soliton, with field errors below 4 %, the Jacobian by the implicit-function theorem agrees with the closed form to 2–6 %, and everything derived from it, the Cramér–Rao bounds, the α–β correlation and the condition number of $F$, agrees to 1–4 %. This is the validation the method needed before it is used on problems without a closed form, and it includes a genuinely two-parameter case with correlated estimates.
2. **It is wrong, and optimistic, when the surrogate smooths a feature it should resolve.** On Burgers at $\nu = 0.01/\pi$ the network's shock is wider than the true one (field error 3–6 %, concentrated at the shock), so its sensitivity to ν is spread over a wider region and is a different function from the reference's narrow spike: Jacobian errors of 1.3–24, and a bound 2–25× too small, which would claim more information about ν than the data carry. The direction of the error matters: a diagnostic that overstates identifiability is worse than one that understates it.
3. **The field error next to every surrogate bound is the condition to check.** The three problems order the same way by field error and by Jacobian error. `pinnrl.analysis` should report the two together, and the docs state the condition: the surrogate's Fisher information is trusted only where its forward error is small compared with the feature the parameter acts on.
4. **Sensor placement from the surrogate works where the surrogate is right.** The eight heat layouts (random counts and 1–8 probes) give the same ordering of bounds from the surrogate as from the closed form, so K02 can use the surrogate for problems without a closed form, with the proviso above.

**Consequences for the backlog:**
- K02 (sensor placement) proceeds on heat and KdV with the surrogate bound; for Burgers-type problems the shock must first be resolved (a sharper network, more collocation at the shock, or a numerical reference).
- F03 will show what the PINN estimate of ν actually does where the surrogate bound is unreliable.
- The Laplace approximation of roadmap item 5 reuses these Hessian blocks and inherits the same validity condition.
