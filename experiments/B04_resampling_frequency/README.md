# B04: resampling frequency

## Question

How often should collocation points be resampled?

## Setup

- **Intervals:** every step, every 10, 100 and 1,000 steps, and never (one fixed set for the whole run), for uniform and for RAD sampling.
- **Problems:** Burgers and Allen–Cahn against the B01 references.
- **Training:** the B01 protocol at 10,000 steps, 1,000 points; 5 seeds.
- **Code:** `run.py`.

Rerun with:

```bash
python experiments/B04_resampling_frequency/run.py --seeds 5 --steps 10000 --workers 4
```

## Result

_Pending._

## Conclusion

_Pending._
