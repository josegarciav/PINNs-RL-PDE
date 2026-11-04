# K47: is the swarm sample a posterior sample?

## Question

Under which conditions does sampling-while-optimizing perform importance sampling?

## Setup

A cloud particle swarm (`pinnrl.uq.sample_equivalence_region`, 30 particles) evaluates the misfit at every position it visits. Two readings of that run are compared with the **exact posterior** $\propto e^{-\text{misfit}/2\sigma^2}$, computed on a grid with the same forward model:

- the **weighted** set: every evaluated position with weight $e^{-(\text{misfit} - \min)/2\sigma^2}$, the importance-sampling reading;
- the **accepted** set: the positions inside the 90 % likelihood-ratio region, unweighted, as K46 used them.

Problems: heat $\alpha$ (closed form), 200 observations, noise 0.05, 30 data sets, 100 iterations; the damped pendulum $\theta'' + b\theta' + k\sin\theta = 0$ with $(k, b) = (9.81, 0.1)$ (only $g/L$ is identifiable, so $k$ replaces $(g, L)$), $\theta$ observed at 50 times on $[0, 10]$ with noise 0.02, 10 data sets, 80 iterations, the forward model an ODE solve; its posterior is 1,000 times narrower than the prior box, so the grid is centred on the swarm's best and spans ±8 standard deviations from the misfit's curvature. Metrics per parameter: the ratio of the sample's standard deviation to the exact one, the shift of the mean in exact standard deviations, the coverage of the truth by the 90 % intervals, the Kolmogorov distance of the marginal CDFs, and the effective sample size of the weights. `run.py`, `results/runs.json`.

## Result

| Problem, parameter | ESS / evaluations | Std ratio, weighted | Std ratio, accepted | Mean shift (exact std) | Coverage: exact / weighted / accepted | KS: weighted / accepted |
|---|---|---|---|---|---|---|
| heat $\alpha$ | 229 / 900 | 0.72 [0.68, 0.76] | 0.74 | 0.02 | 0.93 / 0.87 / 0.90 | 0.138 / 0.105 |
| pendulum $k$ | 90 / 930 | 0.69 [0.63, 0.78] | 0.87 | 0.05 | 1.00 / 0.90 / 1.00 | 0.180 / 0.131 |
| pendulum $b$ | 90 / 930 | 0.53 [0.48, 0.56] | 0.65 | 0.03 | 1.00 / 0.80 / 0.90 | 0.238 / 0.203 |

Median over data sets; brackets are the interquartile range.

## Conclusion

1. **The weighted swarm is centred right and too narrow.** Its mean sits within 0.05 standard deviations of the exact posterior mean on every parameter, but its standard deviation is 0.5–0.7 of the exact one and its 90 % intervals cover the truth 80–87 % of the time. Importance weights can only reweight the positions that exist, and an exploratory swarm concentrates its evaluations near the optimum: the tails of the posterior are not visited, so no weighting recovers them. Sampling-while-optimizing is a posterior sample only where the swarm's cloud covers the posterior's support, which it does for the centre and not for the tails. The effective sample size (25 % of the evaluations on heat, 10 % on the pendulum) says how few evaluations carry the weight.
2. **The accepted set is the better reading, and still narrow.** The likelihood-ratio acceptance defines a region, not a density; its unweighted samples are wider (0.65–0.87 of the exact standard deviation) and cover 90–100 %, consistent with K46's finding that the swarm under-samples the edges of the region at 0.6–0.8 of the Fisher width. For an interval, use the accepted set and widen it by the K46 factor; for a density, the swarm is not the tool.
3. **A sharply peaked posterior is the hard case.** On the pendulum the posterior occupies $10^{-3}$ of the prior box in each direction; 930 evaluations resolve its centre (shift 0.03–0.05 std) but the damping's marginal is the worst (std ratio 0.53, coverage 0.80): the swarm collapses along the best-determined directions first, and the less-determined one is left under-explored.

**Consequences for the backlog:** the parametric surrogate (K45) makes each evaluation cheap enough to run the swarm with 10× the evaluations, or to follow it with a proper sampler (MCMC from the swarm's cloud, K42) when a density is needed; the accepted-set interval, widened by the K46 factor, is the usable product of a single swarm run.
