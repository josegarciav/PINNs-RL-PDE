# F06 / K10: uncertainty and coverage

## Question

Are the confidence intervals for $\alpha$ calibrated? Ensembles (F06), and ensembles vs Laplace vs the sampled region of equivalence (K10): coverage, width and CRPS.

## Setup

- **Data sets:** 30 synthetic data sets of the F01 cell (heat, 200 observations at uniform random points, noise 0.05), one per seed.
- **Ensembles:** 8 members per data set, trained from different initial weights (4 × 64 tanh MLP, 5,000 Adam steps, weights 1 / 10 / 10 / 10, $\alpha$ started at 5× the truth).
- **Four 90 % intervals per data set:** the empirical central interval of the 8 member estimates (`ensemble`), and its Gaussian version from the ensemble mean and standard deviation (`ensemble_gauss`); a Gaussian at member 0's estimate with the standard deviation of the Laplace posterior, i.e. the surrogate's Fisher information at the trained network (`laplace`, `pinnrl.uq.laplace_posterior`); the same with the closed-form Cramér–Rao bound (`fisher`, what an efficient unbiased estimator would report); and the likelihood-ratio region of the *exact* misfit sampled by the swarm (`swarm`, `pinnrl.uq.sample_equivalence_region`), the nonlinear interval the physics alone implies for that data set.
- **Metrics** (`pinnrl.uq.metrics`): the empirical coverage of the truth over the 30 data sets (90 % is calibrated), the mean width, and the mean CRPS (ensemble as samples, the Gaussians in closed form).
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/F06_coverage/run.py --datasets 30 --members 8 --steps 5000 --workers 4
```

## Result

_Pending._

## Conclusion

_Pending._
