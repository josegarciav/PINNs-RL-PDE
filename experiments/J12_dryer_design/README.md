# J12: measurement and campaign design for the spray-dryer twin

**Question.** J11 named the droplet size as the nuisance that breaks the dryer-scale estimate and J08 suggested measuring it. With every input of the tower admitted at the accuracy it is known to (as J09 did for the evaporator), which uncertainty limits the drying curve's parameters and the stickiness margin, which remedy pays, and which operating points should a campaign run?

**Short answer.** Measuring the droplet size buys nothing: the tower sees the droplet's drying time against its residence time, so the diameter and the droplet velocity are one unknown, and with both pinned the six balance inputs (flows, inlet conditions, wall loss) jointly cost almost as much. What pays is ordinary: thermocouples good to 0.2 K (critical moisture from ±0.085 to ±0.052), a nine-point campaign (±0.065), and replicate lab assays for the margin (1.2 → 0.6 K). The margin is limited by record noise, not by any input. The best campaign depends on the target: points near the operating limit for the margin, the corners of the window at high load for the curve. This corrects the recommendation of J11 and J08.

## Setup

`SprayDryer` with J11's instruments at each operating point (six air thermocouples 0.5 K, outlet humidity 0.001, powder moisture 0.003). Unknowns: critical moisture and exponent (flat priors). Nuisances (1σ): wall loss 30 %, droplet diameter 10 %, droplet velocity 15 %, air flow 3 %, solids flow 2 %, inlet air temperature 1 K, inlet humidity 0.001, feed solids fraction 0.005. Targets: the two parameters, and the stickiness margin at J08's point at the limit (180 °C, 11 g/s). Linear-Gaussian posteriors from central differences through the reference (`pinnrl.analysis`), campaign points by `greedy_design` over a 5 × 5 window with replicates allowed; validation by 60 simulated campaigns per design fitted by bounded least squares (about 4 s each).

```bash
python experiments/J12_dryer_design/run.py     # ~2.5 min on 6 workers
```

## Results

**What each input's uncertainty costs** (five-point factorial campaign):

| Admitted | $X_{cr}$ | $n$ | Margin (K) |
|---|---|---|---|
| nothing | 0.036 | 0.035 | 1.12 |
| droplet diameter (10 %) | 0.065 | 0.049 | 1.12 |
| droplet velocity (15 %) | 0.065 | 0.049 | 1.12 |
| diameter and velocity together | 0.065 | 0.049 | 1.12 |
| any one of the six balance inputs | 0.037–0.040 | 0.036–0.038 | 1.12–1.18 |
| the six balance inputs together | 0.055 | 0.049 | 1.20 |
| everything (the baseline) | 0.085 | 0.056 | 1.20 |

**Remedies**, from the baseline:

| Remedy | $X_{cr}$ | $n$ | Margin (K) |
|---|---|---|---|
| droplet size by laser diffraction (3 %) | 0.085 | 0.055 | 1.20 |
| the same with the droplet velocity to 5 % | 0.084 | 0.055 | 1.20 |
| air flow, wall loss, feed solids or outlet humidity improved | 0.085 | 0.055 | 1.20 |
| powder sampled at five heights | 0.082 | 0.051 | 1.11 |
| two thermocouples near the atomiser | 0.073 | 0.052 | 1.19 |
| four replicate lab assays | 0.072 | 0.038 | **0.64** |
| nine operating points (factorial) | 0.065 | 0.040 | 0.93 |
| thermocouples good to 0.2 K | **0.052** | 0.035 | 1.14 |

**Which operating points:**

| Campaign | $X_{cr}$ | $n$ | Margin (K) |
|---|---|---|---|
| the base point five times | 0.113 | 0.072 | 1.43 |
| factorial, 5 / 9 points | 0.085 / 0.065 | 0.056 / 0.040 | 1.20 / 0.93 |
| greedy for the margin, 5 / 9 points | 0.099 / 0.080 | 0.067 / 0.052 | 1.03 / 0.77 |
| greedy for $X_{cr}$, 5 points | 0.071 | 0.048 | 1.66 |

The greedy campaign for the margin repeats two points next to the limit (180 °C / 11 g/s and 190 °C / 12 g/s) and adds one or two far ones; the one for $X_{cr}$ alternates the hot and cold corners of the window at the highest load.

**Validation** (60 fitted campaigns; predicted in brackets):

| Design | $X_{cr}$ | $n$ | Margin rmse (K), coverage of the 90 % interval |
|---|---|---|---|
| baseline | 0.084 (0.085) | 0.052 (0.056) | 1.44 (1.20), 0.87 |
| laser diffraction | 0.075 (0.085) | 0.051 (0.055) | 1.42 (1.20), 0.88 |
| laser diffraction + velocity + tower samples | 0.073 (0.080) | 0.047 (0.050) | 1.18 (1.11), 0.88 |

## Conclusion

1. **The droplet's diameter and its velocity are one unknown.** Each alone costs the same as both together: the records depend on how far the droplets dry in the time they have, which a smaller droplet and a slower one change alike. Measuring the diameter to 3 % therefore changes nothing while the residence time is known to 15 %, and little with both pinned, because the next limit is right behind.
2. **That next limit is again joint.** No single flow, inlet condition or the wall loss costs more than 0.004; the six together cost as much as the droplet pair. As on the evaporator (J09) a one-at-a-time study clears every one of them, and no single calibration pays.
3. **What pays is record quality and campaign size.** Thermocouples at 0.2 K instead of 0.5 K bring the critical moisture from ±0.085 to ±0.052, nine points instead of five to ±0.065; for the stickiness margin, four replicate assays halve the error (1.2 → 0.6 K) and nothing else moves it by more than 10 %. The margin is limited by the noise of the records that carry it, not by any input: all eight nuisances together cost it 7 %.
4. **A campaign should be designed for its target.** Five points chosen for the margin beat the factorial on the margin (1.03 against 1.20 K) and lose on the curve (0.099 against 0.085); five chosen for the curve do the reverse (0.071, 1.66 K). Running the base point five times is the worst on both. The factorial is a reasonable compromise when the twin has both jobs.
5. **The linear prediction is good for the parameters and optimistic for the margin** (1.44 against 1.20 K at the baseline, coverage 0.87), the same direction as in J11: plan with a 20 % allowance.

**Corrections to earlier write-ups:** J11 (conclusion 3) and J08 (conclusion 3) pointed to a droplet-size measurement as the remedy; with the residence time and the balance inputs admitted it is not one. Both now refer here.
