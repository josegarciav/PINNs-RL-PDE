# Research program

**Objective:** estimate **unknown physics** (parameters, coefficient fields, closures and hidden states of a real system) from its routine sensors, with uncertainty you can trust. The first application is digital twins of food-powder lines, meaning evaporators and spray dryers (see [Applications](applications.md)).

Inverse modelling is the method. An estimate of unknown physics is only useful if three things hold:

- the problem is **identifiable**: the sensors determine the unknowns;
- it is **well conditioned and stable**: small disturbances in the data give small changes in the estimate;
- the **uncertainty is calibrated**: a 95 % interval contains the truth about 95 % of the time.

The program establishes those conditions on problems with known answers before applying them to plant equipment.

## How it is organized

- **Papers:** [`experiments/PAPERS.md`](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/experiments/PAPERS.md) is the paper backlog: 27 papers in four tiers, each with its question, contribution, prerequisite papers, target venues and status.
- **Experiments:** [`experiments/BACKLOG.md`](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/experiments/BACKLOG.md) lists the 150 experiments behind them, tracks A–K. Track K holds the 50 conditioning, stability and UQ baselines for Tier 1. Each one runs in its own `experiments/<ID>_<slug>/` folder with its question, setup, result and conclusion.
- **Rules for every result:**
  - equal budgets with the method's own overhead counted;
  - paired seeds (five for main claims), reported as median and IQR with a Wilcoxon test;
  - a classical baseline wherever one exists (finite differences, Kalman filters, moving-horizon estimation, adjoint methods);
  - stated validity conditions.

## The four tiers

| Tier | Papers | What it establishes |
|---|---|---|
| **0: foundations** | P01 audit · P02 sampling at equal compute · P03 optimizers and loss balancing · P04 architectures · P05 failure modes · P06 software (JOSS) | Trustworthy defaults and a fair benchmark protocol |
| **1: inverse modelling** | P07 identification baseline · P08 identifiability and conditioning · P09 training ill-conditioning · P10 disturbance and stability · P11 stability constants and error estimates · P12 high-dimensional stability · P13 calibrated UQ · P14 model-form error and fields · P15 PINN vs classical estimators · P16 physics as a data multiplier | When unknown physics can be recovered, how stable the answer is, and how good its error bars are |
| **2: process units** | P17 evaporator simulator · P18 fouling identifiability · P19 online fouling estimation · P20 drying-kinetics closures · P21 moisture soft sensor · P22 wiped-film evaporators | The Tier 1 methods applied to one unit at a time, against simulators with known truth |
| **3: line-level twins** | P23 coupled-line uncertainty · P24 measurement design · P25 predictive advisory · P26 plant validation · P27 position paper | The coupled line, sensor design and validation on real data |

How the tiers depend on each other:

- Tier 0 fixes the defaults every later paper uses.
- Tier 1 supplies the theory and the tools: conditioning diagnostics, stability analysis and uncertainty quantification.
- Tier 2 applies them to one unit at a time.
- Tier 3 starts only after the Tier 2 estimators beat their classical baselines.

## Current focus

1. **P02**, sampling at equal compute. The first result, B01/B06, is in [Benchmarks](benchmarks.md).
2. **P07 → P08**, identification baseline, then identifiability and conditioning. This is the entry point of the inverse-modelling core and depends on the conditioning tools in the v0.5 [roadmap](roadmap.md).
3. **P06**, the software paper, once P01 and P02 are drafted (v0.4.0 is on PyPI).
