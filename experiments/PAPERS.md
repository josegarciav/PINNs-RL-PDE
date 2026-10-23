# Paper backlog

The research program as a list of papers. Each one builds only on papers above it, and each maps to experiments in [BACKLOG.md](BACKLOG.md).

**Objective:** estimate unknown physics (parameters, coefficient fields, closures and hidden states of a real system) from its routine sensors, with uncertainty you can trust. The first application is digital twins of food-powder lines: evaporators and spray dryers.

**Rules for every paper:**
- Equal budgets and paired seeds (5 for main claims, 3 for exploratory ones), with method overhead counted. Report median and IQR, with a Wilcoxon test.
- A classical baseline wherever one exists: finite differences, Kalman filters, moving-horizon estimation, adjoint methods.
- The validity conditions of every method are stated: well-posedness, identifiability, conditioning.
- Code, configs and seeds are in `experiments/<ID>_<slug>/`, so the results reproduce from the repository.

Status: `idea` → `planned` (experiments defined) → `running` → `drafting` → `submitted` → `published`.

---

## Tier 0: foundations

Benchmarks with known answers. They fix the defaults every later paper uses.

| ID | Working title | Question and contribution | Experiments | Needs | Venues | Status |
|---|---|---|---|---|---|---|
| P01 | *Auditing a PINN library: exact solutions, boundary semantics and seed variance* | How much of the published "PINN accuracy" survives a physics audit? A reproducible audit protocol, with the bugs it found, such as 2-D operators silently zero and wrong exact solutions. | A01–A10, H08 | – | SoftwareX, J. Open Research Software | running (A01/A08/A09/H08 done) |
| P02 | *Collocation sampling at equal compute* | Do adaptive samplers (RAD, an RL bandit) beat uniform sampling when their overhead is charged and seeds are paired? Includes resampling frequency as a separate factor. | B01–B10, C01 | P01 | J. Comput. Phys., CMAME, NeurIPS ML4PS workshop | running (B01/B06 done) |
| P03 | *Optimizers and loss balancing under paired budgets* | Adam, L-BFGS, switch points and adaptive weights (LRW/RBW): which helps, and when? | D01–D09 | P01 | J. Sci. Comput., ML4PS | planned |
| P04 | *Architectures at matched parameter count* | Spectral bias, SIREN and Fourier scales, depth vs width, with parameter counts matched. | E01–E08 | P01 | Neural Networks, ML4PS | planned |
| P05 | *Known PINN failure modes at equal budget* | Which published fixes (curriculum, time-marching, causal weighting) survive a fair comparison? | H01–H07, B09 | P02, P03 | J. Comput. Phys. | planned |
| P06 | *pinnrl: a PINN library with a benchmark protocol* | The software, its verification suite and its evaluation protocol. | all Tier 0 | P01, P02 | JOSS | planned (after v0.4.0) |

## Tier 1: inverse modelling with PINNs (the core)

When can unknown physics be recovered, how stable is the answer, and how good are its error bars?

| ID | Working title | Question and contribution | Experiments | Needs | Venues | Status |
|---|---|---|---|---|---|---|
| P07 | *Scalar parameter identification with PINNs: a baseline study* | Accuracy against noise, number of sensors, initial guess and data weight, for heat α, KdV (α, β) and Burgers ν. | F01–F05 | P01, P03 | Inverse Problems in Science and Engineering, J. Comput. Phys. | planned |
| P08 | *Identifiability and conditioning of PINN inverse problems* | Fisher-information spectra, sloppy directions, and structural vs practical identifiability (profile likelihood). Which parameter combinations can the sensors determine? | K01, K02, K11–K17, F07 | P07 | Inverse Problems, SIAM/ASA JUQ | planned |
| P09 | *Ill-conditioning in PINN training and how to precondition it* | Loss-Hessian and NTK spectra driven by the differential operator; the effect of second-order and preconditioned optimizers. | K03, K04, K18–K24, D10 | P03 | SIAM J. Sci. Comput., J. Comput. Phys. | planned |
| P10 | *Disturbance and stability analysis of PINN estimators* | How do identified parameters respond to sensor noise, bias, dropout and delay, and to IC/BC/parameter perturbations? The linear prediction F⁻¹JᵀΣ⁻¹δy is compared with what is observed. | K05, K25–K31, F05 | P08 | Inverse Problems, J. Comput. Phys. | planned |
| P11 | *Stability constants and error estimates for PINNs* | Empirical stability constants (error vs residual) across PDEs, compared with conditional-stability bounds. A bridge to classical numerical analysis: Lax equivalence, energy estimates. | K06, K07, K32–K38, A04 | P10 | IMA J. Numer. Anal., ESAIM: M2AN | planned |
| P12 | *Stability and quadrature error in many dimensions* | How do stability constants and sampling error scale with dimension and with coupled fields? | K08, K49, K50, I10, B10 | P11 | SIAM J. Sci. Comput., J. Sci. Comput. | idea |
| P13 | *Calibrated uncertainty for physics identified by PINNs* | Bayesian view (the loss is a negative log-posterior and weights are 1/σ²), comparing ensembles, Laplace and conformal intervals: their coverage and the conditions under which each holds. | F06, F07, K10, K39–K47 | P08 | SIAM/ASA JUQ | planned |
| P14 | *Model-form error and field-valued unknowns* | Identifying time-varying parameters and coefficient fields α(x, t) when the physics is incomplete. | F08, F09 | P13 | CMAME | idea |
| P15 | *PINNs vs classical estimators on identical tasks* | PINN inverse problems against EKF/UKF, moving-horizon estimation and adjoint methods: accuracy, latency, tuning effort. | F10, J05 | P07, P13 | J. Process Control, Automatica | idea |
| P16 | *How much data does the physics save?* | Sample efficiency of physics plus data vs data alone (The Well). | G01–G03 | P07 | TMLR, ML4PS | idea |

## Tier 2: process units

Simulated plants with known truth first, then data.

| ID | Working title | Question and contribution | Experiments | Needs | Venues | Status |
|---|---|---|---|---|---|---|
| P17 | *An open falling-film evaporator simulator and benchmark* | A grid-converged reference model with fouling, plus plant-like sensors. A data/benchmark paper. | J01, J04, K09 | – | Computers & Chemical Engineering, Data in Brief | idea |
| P18 | *Identifiability of fouling profiles from effect-level sensors* | Which functionals of R_f(z, t) can boundary measurements recover, and what a deposition-kinetics prior adds. | J03, J04, K01-type analysis | P08, P17 | Inverse Problems, J. Process Control | idea |
| P19 | *Online fouling estimation with calibrated time-to-clean* | A PINN/hybrid estimator vs EKF/UKF, with intervals for the time to the next clean-in-place. | J02–J06 | P13, P15, P18 | J. Process Control, Food and Bioproducts Processing | idea |
| P20 | *Learned drying-kinetics closures* | Reaction-engineering-approach kinetics as a constrained (monotone) learned closure, from a single droplet to the dryer. | J07 | P14 | Drying Technology, Chem. Eng. Science | idea |
| P21 | *A residual-moisture soft sensor for spray dryers* | Moisture from routine air temperatures and humidity with delayed lab samples, plus a stickiness margin T − T_g. | J08 | P13, P20 | J. Food Engineering, Computers & Chemical Engineering | idea |
| P22 | *Wiped-film evaporator models and method transfer* | A thin-film model, and how much of P18–P19 transfers. | new | P19 | Chem. Eng. Science | idea |

## Tier 3: line-level twins

Only after the Tier 2 estimators beat their classical baselines.

| ID | Working title | Question and contribution | Experiments | Needs | Venues | Status |
|---|---|---|---|---|---|---|
| P23 | *Uncertainty propagation in a coupled evaporator → dryer twin* | How concentrate uncertainty propagates into powder moisture, and the linearized stability of the coupled system. | J10, K09, K48 | P19, P21 | Computers & Chemical Engineering | idea |
| P24 | *Measurement design for physics-informed twins* | Bayesian optimal design: when to pull a lab sample and where to add a sensor. This is where the bandit and RL work leads. | J09, K02, C track | P08, P13, P23 | SIAM/ASA JUQ, Computers & Chemical Engineering | idea |
| P25 | *Model-predictive advisory with a PINN surrogate* | Setpoint advice that respects stickiness and moisture constraints, on a simulated plant. | new | P23 | J. Process Control | idea |
| P26 | *Plant validation* | P19/P21 estimators on historian data. Needs an industrial partner. | new | P19, P21 | Ind. Eng. Chem. Res., J. Food Engineering | idea |
| P27 | *Position paper: physics-informed digital twins for food processing* | What estimation of unknown physics can and cannot deliver, and under which conditions. | – | P13, P19 | Trends in Food Science & Technology, Comput. Chem. Eng. perspective | idea |

---

## Next three to write

1. **P02** (sampling at equal compute). B01/B06 are done. Still to run: B02 (more PDEs), B03 (the RAD k, c grid), B04 (resampling frequency, which B01 suggests matters most) and B08 (point-count scaling).
2. **P07 → P08** (identification baseline, then identifiability and conditioning). These are the entry point of the inverse-modelling core. They need the v0.5 conditioning module (Fisher information, sensitivity SVD).
3. **P06** (JOSS), once v0.4.0 is on PyPI and P01/P02 are drafted.
