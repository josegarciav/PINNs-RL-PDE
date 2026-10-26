# K46: swarm sampling of the region of equivalence

## Question

Does an exploratory swarm recover the nonlinear region of equivalence that the Fisher ellipse misses?

## Setup

- **Forward maps with a closed form**, so the region sampled is the one the physics implies for a noisy data set, independent of any surrogate: heat $\alpha$ (200 points, noise 0.05); the two-parameter KdV soliton $(\alpha, \beta)$ (200 points, noise 0.01); and the saturation model $y = a(1 - e^{-bx})$ at 40 points with noise 0.3, the nonlinear example of Pallero et al. (2018) whose region is banana-shaped.
- **Sampler:** `pinnrl.uq.sample_equivalence_region`, a cloud generalized PSO (30–40 particles, 100–150 iterations, per-particle inertia, acceleration and time step drawn inside the second-order stability region, dispersion monitored). The accepted set is the likelihood-ratio region, positions within $\sigma^2 \chi^2_p(0.9)$ of the best misfit found.
- **Comparison:** the Fisher ellipse at the 90 % level centred on the best fit; the fraction of accepted samples inside it (1 for a linear problem, less where the region is curved); the width of the swarm's 90 % interval over the ellipse's; the correlation of the accepted samples against the ellipse's $\rho$; whether the truth lies in the swarm interval. 10 data sets per problem.
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/K46_swarm_equivalence/run.py --datasets 10
```

## Result

_Pending._

## Conclusion

_Pending._
