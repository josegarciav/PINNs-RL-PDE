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

_Pending._

## Conclusion

_Pending._
