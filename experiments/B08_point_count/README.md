# B08: collocation-count scaling

## Question

Error vs number of points for each strategy.

## Setup

- **Counts:** 250, 500, 1,000, 2,000 and 5,000 points per step, for uniform (every 100 steps) and RAD.
- **Problems:** Burgers and Allen–Cahn against the B01 references.
- **Training:** the B01 protocol at 5,000 steps; 3 seeds.
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/B08_point_count/run.py --seeds 3 --steps 5000 --workers 4
```

## Result

_Pending._

## Conclusion

_Pending._
