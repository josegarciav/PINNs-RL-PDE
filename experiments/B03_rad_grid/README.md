# B03: the RAD hyperparameter grid

## Question

Which RAD hyperparameters work, and are they PDE-dependent?

## Setup

- **Grid:** $k \in \{0.5, 1, 2, 4\} \times c \in \{0, 0.5, 1, 2\}$, the grid of Wu et al. (2023), plus the uniform-every-100-steps control ($k = 0$).
- **Problems:** Burgers ($\nu = 0.01/\pi$) and Allen–Cahn, scored against the B01 references.
- **Training:** the B01 protocol at 10,000 steps, 1,000 points resampled every 100 steps; 3 seeds.
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/B03_rad_grid/run.py --seeds 3 --steps 10000 --workers 4
```

## Result

_Pending._

## Conclusion

_Pending._
