# Theory

These pages state the conditions under which an estimate from pinnrl can be trusted. Read them if you identify parameters from measurements and need to know what a reported number assumes.

The setting, in one line:

$$
\mathcal{N}_\theta[u] = 0, \qquad y = \mathcal{H}u + \varepsilon, \qquad u \approx u_w .
$$

- **$u$**: the state, the field the equation governs.
- **$\theta$**: the parameters to identify.
- **$\mathcal{N}_\theta[u] = 0$**: the governing equation.
- **$y$**: the observations.
- **$\mathcal{H}$**: the sensor operator, which maps the state to what the sensors read.
- **$\varepsilon$**: the measurement noise.
- **$u_w$**: the network with weights $w$.

All five pages share this notation. Each module of the inverse-modelling core cites the condition it relies on.

| Page | Question it answers | Used by |
|---|---|---|
| [Well-posedness and stability](well_posedness.md) | When does a small change in the data give a small change in the estimate, and what does the physics do to an ill-posed problem? | the disturbance suite, the stability constants |
| [Identifiability and conditioning](identifiability.md) | Can the sensors determine the parameters at all, and which combinations do they determine badly? | `pinnrl.analysis`, experiments F01–F05 and K01 |
| [The Bayesian view and uncertainty](bayesian.md) | Why is the PINN loss a negative log-posterior, what does that make the loss weights, and what counts as a calibrated error bar? | `pinnrl.uq`, K39, F06, K10 |
| [The numerical-analysis bridge](numerical_analysis.md) | What do consistency, stability and convergence mean for a PINN, and how does its error relate to the training residual? | the error estimates, K06–K08 |
| [Equivalence regions and swarm sampling](equivalence_regions.md) | Why report the set of parameters that fit the data rather than one estimate, and how to sample it when the problem is nonlinear? | `pinnrl.uq`, K10, K45–K48 |

Each page gives four things:

1. The statement.
2. The condition under which it holds.
3. The consequence for a PINN.
4. The reference where the proof lives.

!!! note "Reference papers"
    The maintainer's reference papers on nonlinear uncertainty (the Oviedo particle-swarm work on sampling equivalence regions) are summarized in [Equivalence regions and swarm sampling](equivalence_regions.md) and cited by DOI. The PDFs stay out of the repository.
