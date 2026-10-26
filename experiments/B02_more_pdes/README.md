# B02: the sampler ranking on other equations

## Question

Does B01's ranking hold on heat, wave, convection and KdV?

## Setup

- **Equations** (catalogue entries with a closed form, now accepted by `pinnrl.benchmarks.sampling_study`): heat, convection, wave and the KdV soliton. Cahn–Hilliard has no closed form in the catalogue and is left out.
- **Samplers:** uniform every step, uniform every 100 steps (the same-cadence control) and RAD ($k = 1$, $c = 1$, pool 20×) every 100 steps.
- **Training:** the B01 protocol, 4 × 64 tanh MLP, 1,000 points, Adam at $10^{-3}$, 10,000 steps, 20 checkpoints; 5 seeds, paired across samplers.
- **Metric:** relative L2 error against the closed form on a 256 × 101 grid.
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/B02_more_pdes/run.py --seeds 5 --steps 10000 --workers 4
```

## Result

_Pending._

## Conclusion

_Pending._
