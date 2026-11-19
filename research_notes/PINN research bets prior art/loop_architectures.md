# Neural architectures for physics-informed models queried and updated inside tight loops (RL, MPC, real-time estimation)

Target architecture under assessment: u(x, t; z) = sum_j c_j(z) phi_j(x, t), trunk phi cached, small head z -> c, coefficients updatable by recursive least squares (RLS).

Reading conventions used below:
- [PR] = peer-reviewed venue confirmed in the source; [PP] = preprint (arXiv only, or venue not confirmed in this session).
- "(full text)" = numbers read from the paper body in this session; "(snippet)" = number taken from a search-result summary and NOT checked against the paper body. Treat snippet numbers as leads to verify before quoting.
- Timings from different papers use different hardware, horizons and solvers and are not comparable with each other.

## 1. Trunk-and-head and conditioned architectures: what exists, what is reported, which cache trunk features

### Takeaway
The form u = sum_j c_j(z) phi_j(x,t) is exactly the DeepONet / POD-DeepONet / two-step-DeepONet output layer, and hypernetwork variants (HyperPINN, Hyper-LR-PINN, HyperLoRA, NIF) generalise it by letting z modulate inner weights; so the representation itself is not new. What the literature rarely reports is per-query latency on a CPU with the trunk cached: published timings are mostly GPU, mostly whole-field, and only one source found states a measured gain from precomputing trunk outputs.

### Cited Findings

Physics-informed DeepONet (Wang, Wang, Perdikaris; Science Advances 2021) [PR] (full text via ar5iv of arXiv 2103.10974)
- Relative L2 errors: diffusion-reaction, DeepONet ~1.92% +/- 1.12% vs physics-informed DeepONet ~0.45% +/- 0.16% (paper calls this "~80% improvement"); Burgers best average ~1.38% with the modified MLP (2.80e-02 with the standard MLP), worst test sample 3.31e-02; eikonal (circles) 4.22e-03; antiderivative without paired data 3.25e-03 (s) vs 8.80e-01 for the data-free DeepONet — [arXiv 2103.10974](https://ar5iv.labs.arxiv.org/html/2103.10974)
- Inference: a trained model predicts the full spatio-temporal Burgers solution in "~10ms"; claims O(10^3) PDE solves "in a fraction of a second" and "up to three orders of magnitude" over a Chebfun spectral solver; hardware a single NVIDIA V100 GPU. Exact timings are only in a figure — [arXiv 2103.10974](https://ar5iv.labs.arxiv.org/html/2103.10974)
- Training cost (single V100, hours): diffusion-reaction 1.13 (DeepONet) vs 2.27 (physics-informed); Burgers 7.61 (MLP) and 9.25 (modified MLP) — [arXiv 2103.10974](https://ar5iv.labs.arxiv.org/html/2103.10974)
- Networks: Burgers uses 7 layers x 100 units, tanh, for branch and trunk; latent dimension q not found in the extracted text. The paper does not discuss long-time integration or error accumulation (Burgers solved on t in (0,1]) and does not describe caching the trunk — [arXiv 2103.10974](https://ar5iv.labs.arxiv.org/html/2103.10974)
- Published version — [Science Advances / PMC8480920](https://pmc.ncbi.nlm.nih.gov/articles/PMC8480920)

HyperPINN (de Avila Belbute-Peres, Chen, Sha; NeurIPS 2021 workshop DLDE) [workshop paper, lightly reviewed] (full text)
- Burgers, parameterised by viscosity: HyperPINN main network 6 hidden layers x 8 units, 393 parameters, MSE 1.9e-5, evaluation 86 microseconds; hypernetwork 9385 parameters, 158 microseconds. Baselines: small PINN (401 params) MSE 3.0e-4, 92 microseconds; large PINN (9665 params) MSE 2.3e-5, 158 microseconds. Timings "measured on an NVIDIA Tesla V100 GPU" — [arXiv 2111.01008](https://ar5iv.labs.arxiv.org/html/2111.01008)
- Lorenz: main network 115 parameters, hypernetwork 1406; aggregate squared error 17.5 vs 39.4 (small baseline, 214 params) and 20.6 (large, 3334 params); no timing — [arXiv 2111.01008](https://ar5iv.labs.arxiv.org/html/2111.01008)
- Relevance: the hypernetwork is evaluated once per parameter value and the small main network is then reused, which is the "compute once per z, query many (x,t)" split; this is the opposite caching direction from the target architecture (cache per z, not per (x,t)) — [arXiv 2111.01008](https://ar5iv.labs.arxiv.org/html/2111.01008)

Hyper-LR-PINN (Cho, Lee, Rim, Park; NeurIPS 2023) [PR] (full text)
- Hidden weights factored as U diag(s(mu)) V^T; phase 1 learns U, V and a hypernetwork mu -> s; phase 2 freezes U, V, hidden biases and the hypernetwork and trains only the diagonal coefficients plus first and last linear layers. Parameter counts: about 351 (adaptive rank) vs 10,401 for the vanilla PINN ("about 30x smaller") — [arXiv 2310.09528](https://ar5iv.labs.arxiv.org/html/2310.09528)
- Convection beta = 30 (a PINN failure mode), absolute / relative error: Hyper-LR-PINN adaptive rank 0.0375 / 0.0389; naive LR-PINN 0.5617 / 0.5344; MAML 0.6757 / 0.6294; Reptile 0.5893 / 0.5551. On the combined convection-diffusion-reaction case MAML is better (0.0144 vs 0.0329 relative) — [arXiv 2310.09528](https://ar5iv.labs.arxiv.org/html/2310.09528)
- Adaptation to unseen parameters: reaches absolute L2 error below 0.05 "in only one epoch" for most beta; 2D Helmholtz in 10 epochs vs PINN over 2,000; "more than (x4) speed up in training time". No inference timing reported. Stated limitation: parameters are PDE coefficients, not initial or boundary conditions — [arXiv 2310.09528](https://ar5iv.labs.arxiv.org/html/2310.09528)

HyperLoRA for PDEs (Majumdar et al., 2023) [PP] (snippet)
- Hypernetwork predicts low-rank factors of each layer; reports that LoRA-based hypernetworks can violate the physics unless a physics-informed loss is added; about 8x reduction in predicted parameters on Burgers and Kovasznay flow — [arXiv 2308.09290](https://arxiv.org/abs/2308.09290v1)

Low-rank adaptive physics-informed HyperDeepONets (2025) [PP] (snippet)
- Applies LoRA to the hypernetwork's large output weight matrix and extends to physics-informed training — [arXiv 2507.18346](https://arxiv.org/pdf/2507.18346)

Neural Implicit Flow (Pan, Brunton, Kutz; JMLR 2023) [PR] (snippet)
- Two MLPs: ShapeNet (space) and ParameterNet (time, parameters, sensors), where ParameterNet outputs the weights and biases of ShapeNet; i.e. z modulates all trunk weights instead of only a linear read-out — [arXiv 2204.03216](https://arxiv.org/pdf/2204.03216)

POD-DeepONet (Lu et al., "comprehensive and fair comparison", CMAME 2022) [PR] (snippet)
- Replaces the trunk network by precomputed POD basis functions; reported to outperform vanilla DeepONet on a wide class of problems. DeepONet and FNO comparable on simple settings; FNO degrades on complex geometries — [arXiv 2111.05512](https://arxiv.org/abs/2111.05512v1)
- A later subspace-regression study finds bases extracted from FFNO and DeepONet "worse than global POD in all experiments" — [arXiv 2509.23249](https://arxiv.org/pdf/2509.23249)

Two-step DeepONet training (Lee and Shin; SIAM-published) [PR] (snippet)
- Train the trunk first (no branch), orthonormalise it (Gram-Schmidt / QR; a follow-up uses SVD), then train the branch to output the least-squares coefficients; the authors argue the minimised loss equals the monolithic one — [arXiv 2309.01020](https://arxiv.org/abs/2309.01020v1); [NSF PAR 10596828](https://par.nsf.gov/servlets/purl/10596828)
- Caveat from a later hybrid least-squares/gradient-descent paper: forming the LS system over all branch x trunk input combinations gives a prohibitively large linear problem, so it is split into two sub-problems — [arXiv 2508.15394](https://www.arxiv.org/abs/2508.15394)

Explicit trunk caching and fixed-basis estimation (snippets)
- An active operator-learning paper precomputes trunk outputs per grid location and reports about 0.029 s per problem for full-domain evaluation vs 0.165 s computed naively — [arXiv 2503.03178](https://arxiv.org/pdf/2503.03178)
- Hybrid PBDW + DeepONet state estimation replaces the trunk with pre-computed orthonormal basis functions; training and inference at fixed, optimally selected sensor locations — [arXiv 2512.11834](https://arxiv.org/pdf/2512.11834)
- BLISSNet (2026) states that some components can be precomputed after the first call on a domain, giving low inference cost — [arXiv 2602.24228](https://arxiv.org/html/2602.24228v1)

Multi-step DeepONet for MPC (de Jong, Shukla, Lazar, 2025) [PP] (snippet)
- MS-DeepONet predicts a whole output horizon from an input sequence in one shot, in place of one branch per output; code public — [arXiv 2505.18008](https://arxiv.org/abs/2505.18008)

### Inferences
- Novelty cannot be claimed for the representation. DeepONet with a (x,t) trunk and a branch on z is the same bilinear form; POD-DeepONet and two-step training already treat the trunk as a fixed basis and the branch as a coefficient map fitted by least squares.
- What is thin in the literature is the loop-oriented use of that form: (a) cached trunk values and trunk derivatives at a fixed set of query/collocation points, (b) head latency on a single CPU core, (c) online recursive updates of c. Only one source (arXiv 2503.03178) quantifies a caching gain, and it is a ~5.7x whole-field figure (my arithmetic: 0.165 / 0.029), not a per-step loop number.
- Published "query time" numbers are overwhelmingly GPU and field-level: ~10 ms per Burgers field on a V100 (PI-DeepONet), 86–158 microseconds per evaluation on a V100 (HyperPINN). None of these is a CPU, single-point, batch-1 measurement of the kind a control loop needs.
- Hypernetwork methods cache in the other direction (fix z, reuse the generated network over (x,t)). For a loop where z changes every step and (x,t) query points are fixed, the trunk-cached linear head is the cheaper side of the trade; a hypernetwork must regenerate weights and re-evaluate the main network every step.
- A linear read-out restricts expressiveness to a fixed n-dimensional subspace (Kolmogorov n-width limits for transport-dominated problems); Hyper-LR-PINN and NIF exist precisely because inner-weight modulation handles convection-dominated parameter dependence better. The beta = 30 convection results above are the reference test any fixed-basis design should be measured against.

### Gaps
- DeepONet (Lu, Jin, Pang, Zhang, Karniadakis, Nature Machine Intelligence 3, 218–229, 2021) was not opened in this session; no numbers from it are quoted here.
- FiLM (feature-wise linear modulation) and modulation-based conditioning for neural fields/PINNs: the one search attempted returned nothing on FiLM. No sourced findings; needs a dedicated search (FiLM, modulated SIREN / functa-style latent modulation, DINo, CORAL).
- Meta-learned PINNs (MAML/Reptile for PINNs, meta-auto-decoder, learned loss/optimiser): only seen here as baselines in the Hyper-LR-PINN tables. Primary sources not reviewed.
- HyperDeepONet (Lee et al., ICLR 2023; a hypernetwork maps the input function to all target-network parameters) surfaced only in a search summary; the primary source was not opened, so no sourced finding is listed.
- Latent dimension p (number of basis functions) and CPU per-point latency are not reported in the texts read.
- Reduced-basis-method literature proper (certified RB, EIM/DEIM, a posteriori error bounds, offline/online decomposition) was not searched; it is the classical precedent for "offline basis, online small solve" and should be cited by the report writer from primary sources.

## 2. Linear-in-coefficients models updated online: update costs and stability results

### Takeaway
Online update of a linear head on a learned basis is established practice outside PDE surrogates: meta-learned bases with last-layer adaptation in flight control (Neural-Fly, with an exponential-stability theorem and measured millisecond update cost on a Raspberry Pi), Bayesian last-layer regression (ALPaCA), and EDMD/Koopman models updated by RLS. For PDE solution fields, the closest families are least-squares-in-the-last-layer solvers (PIELM, random-feature methods) and Neural Galerkin / EDNN, which update all parameters by a small linear solve per time step; I found no paper that applies RLS to the coefficients of a physics-informed space-time basis inside a control or RL loop.

### Cited Findings

Learned basis + online linear adaptation (control)
- Neural-Fly (O'Connell et al., Science Robotics 2022) [PR] (full text): basis network 11 -> 50 -> 60 -> 50 -> 4 (ReLU), meta-learned from 12 minutes of flight data; 12 coefficients adapted online (4 per force axis); controller runs at 50 Hz on a Raspberry Pi 4; about 10 ms for the full adaptive control computation, about 3.5 ms for feed-forward inference and about 3.0 ms for the adaptation update — [arXiv 2205.06908](https://ar5iv.arxiv.org/html/2205.06908)
- Neural-Fly stability: Theorem 1, "the position tracking error exponentially converges to the ball" whose size scales with representation error, measurement noise and the rate of change of the wind coefficients; the method "guarantees closed-loop exponential stability and robustness without assuming persistent excitation". Tracking: RMS 9.4 cm at 43.6 km/h vs 23.9 (nonlinear baseline) and 22.7 (L1); average improvements 66% / 42% / 35% over nonlinear / L1 / INDI — [arXiv 2205.06908](https://ar5iv.arxiv.org/html/2205.06908)
- ALPaCA (Harrison, Sharma, Pavone, 2018) [PP as cited; venue not confirmed here] (snippet): network trained offline, Bayesian linear regression on the last layer online, giving confidence intervals; feature network, prior over last-layer weights and noise covariance are meta-learned — [arXiv 1807.08912](https://arxiv.org/pdf/1807.08912.pdf); [arXiv 1912.08866](https://arxiv.org/pdf/1912.08866)
- A rover-terrain paper notes the ALPaCA online update can be done with RLS (Kalman-filter-like), that RLS on basis functions dates to the 1980s, and that the distinguishing step is backpropagating the training loss through the update — [arXiv 2009.10191](https://arxiv.org/pdf/2009.10191)
- Off-road driving (2025) [PP] (snippet): Kalman filter online over a last-layer ensemble, adaptation parameters meta-learned offline — [arXiv 2504.16923](https://arxiv.org/html/2504.16923v1)
- A 2026 preprint adapts the last layer by SGD on one-step predictions for MPC and proposes low-rank second-order updates — [arXiv 2604.04039](https://arxiv.org/pdf/2604.04039v1)
- Meta-learning with differentiable closed-form (ridge) solvers as the adaptation mechanism, ICLR 2019 [PR] (snippet) — [ICLR 2019 poster](https://iclr.cc/virtual/2019/poster/669)

Koopman / EDMD with control
- Korda and Mezic (Automatica 2018) [PR] (full text via ar5iv): lifted linear predictor used in linear MPC. Bilinear motor: N = 103 lifting functions, qpOASES in Matlab on a 2 GHz Intel Core i7 with 8 GB RAM, 6.86 ms per step for Koopman MPC (including the lifting map) vs 103 ms for local-linearisation MPC; one-second prediction RMSE 32.3% vs 135.5% — [arXiv 1611.03537](https://ar5iv.arxiv.org/html/1611.03537)
- Same paper, KdV PDE control: state dimension 128, N = 385 lifting functions, dt = 0.01 s, horizon 10 steps, three bounded inputs, average 0.28 ms per control input with the dense MPC form. Van der Pol prediction: average RMSE 24.4% (N = 102) vs 912.5% for linearisation at 0; RMSE falls 66.5% -> 24.4% as N goes 5 -> 100 — [arXiv 1611.03537](https://ar5iv.arxiv.org/html/1611.03537)
- Koopman MPC for nonlinear flows (Arbabi, Korda, Mezic; Burgers and cavity flow) [PP as seen] (snippet): "sub-millisecond computation time" for the control input in closed loop — [arXiv 1804.05291](https://ar5iv.labs.arxiv.org/html/1804.05291)
- 2026 Koopman tutorial, Duffing closed loop (snippet): mean CPU time per iteration 2.2 ms (linear EDMDc), 3.1 ms (bilinear), 4.7 ms (GeKo), 8.2 ms (KCF); worst cases 6.3 to 29.4 ms — [arXiv 2607.01819](https://arxiv.org/pdf/2607.01819)
- Power-grid Koopman embedding MPC (snippet): about 0.2 s per control step vs 7 s for a trajectory-sensitivity MPC baseline — [arXiv 2206.01272](https://arxiv.org/pdf/2206.01272)
- Online/recursive Koopman (all [PP], snippets): Recursive Koopman Learning combines EDMD, RLS and MPC and states prior RLS-based methods "lack comprehensive convergence and complexity analyses" — [arXiv 2509.08241](https://arxiv.org/html/2509.08241v1); CR-RKL argues recursive Koopman updates have "severe numerical vulnerabilities that are frequently overlooked" and proposes covariance regulation — [arXiv 2606.15317](https://arxiv.org/html/2606.15317); ASACK refines an autoencoder Koopman model online with a contractive adaptation and a closed-form ultimate bound depending on operator drift rate under persistent excitation — [arXiv 2605.09659](https://arxiv.org/pdf/2605.09659); Deep Koopman iterative learning proves closed-loop input-to-state stability with respect to the Koopman approximation error — [arXiv 2601.21230](https://arxiv.org/html/2601.21230); adaptive Koopman MPC with online model correction — [arXiv 2412.02972](https://arxiv.org/html/2412.02972v1), [arXiv 2405.09101](https://arxiv.org/pdf/2405.09101v1.pdf)
- Robust stability and recursive feasibility for Koopman data-driven predictive control with a fixed model — [arXiv 2405.01292](https://arxiv.org/pdf/2405.01292v1.pdf)

Least squares in the last layer for PDEs
- PIELM (Dwivedi and Srinivasan): random fixed hidden layer, physics-constrained least-squares solve for output weights — [arXiv 1907.03507](https://arxiv.org/pdf/1907.03507)
- Dong and collaborators: hidden weights fixed, output layer by linear / nonlinear least squares for forward and inverse parametric PDEs, compared with PINNs (JCP 2023) [PR] — [InversePDE JCP 2023](https://www.math.purdue.edu/~sdong/PDF/InversePDE_JCP2023.pdf)
- RINN (2025) [PP]: orthogonalises last-hidden-layer basis functions to reduce PIELM's initialisation sensitivity, then linear least squares — [arXiv 2506.17654](https://arxiv.org/html/2506.17654v1)

Neural Galerkin and evolutional networks (all parameters evolve; a linear solve per step)
- Neural Galerkin (Bruna, Peherstorfer, Vanden-Eijnden; JCP 2024) [PR] (full text, first 100k characters): parameters follow M(theta) theta_dot = F(t, theta), with M and F estimated by adaptive sampling; explicit integrators need one linear solve with M per step (RK45 used), implicit ones a nonlinear solve; the paper suggests "it may be useful to add a regularizing term" lambda*Id to M. No formal cost per step and no runtimes found; no formal stability or convergence analysis in the text read — [arXiv 2203.01360](https://arxiv.org/html/2203.01360v4)
- Reported accuracy there: Allen-Cahn l2 error "two orders of magnitude lower than that of the linear Galerkin approximation" with 16 degrees of freedom; harmonic-trap mean relative error "roughly 10^-4"; uniform static sampling reaches 100% relative error after a few steps on advection even with 10^5 samples, which is the motivation for adaptive sampling — [arXiv 2203.01360](https://arxiv.org/html/2203.01360v4)
- Randomized Sparse Neural Galerkin (Berman and Peherstorfer; NeurIPS 2023) [PR] (snippet): update a random sparse subset of parameters per step; "up to two orders of magnitude more accurate at a fixed computational budget and up to two orders of magnitude faster at a fixed accuracy" than dense updates; randomisation also limits error accumulation — [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0cb310ed8121549488fea8e8c2056096-Abstract.html); [arXiv 2310.04867](https://ar5iv.labs.arxiv.org/html/2310.04867)
- EDNN (Du and Zaki; Phys. Rev. E 104, 045303, 2021) [PR] (snippet): network fitted to the initial state only, then parameters marched with the governing equations; tested on heat, advection, Burgers, Kuramoto-Sivashinsky, Navier-Stokes — [Phys. Rev. E](https://link.aps.org/doi/10.1103/PhysRevE.104.045303); [arXiv 2103.09959](https://arxiv.org/abs/2103.09959v1)
- Low-rank EDNN (LR-EDNN, 2025) [PP] and MAD-NG (meta-auto-decoder Neural Galerkin for parametric PDEs, 2025) [PP] (snippets) — [arXiv 2509.16395](https://arxiv.org/html/2509.16395v1); [arXiv 2512.21633](https://arxiv.org/html/2512.21633v1)

Reduced-order models inside MPC, with online basis updates (process systems)
- Pitchaiah and Armaou: local POD-Galerkin ROM used in constrained MPC and recursively updated with closed-loop data because the ROM is valid only in a small region of state space — [CDC-ECC 2011 paper](https://skoge.folk.ntnu.no/prost/proceedings/cdc-ecc-2011/data/papers/1977.pdf); [AIChE 2011](https://proceedings.aiche.org/conferences/aiche-annual-meeting/2011/proceeding/paper/473a-predictive-controller-design-spatially-distributed-processes-using-adaptive-model-reduction)
- Robust multi-model MPC for distributed parameter systems that triggers online POD-basis updates when a sufficient stability condition fails — [ADCHEM 2015](https://skoge.folk.ntnu.no/prost/proceedings/adchem2015/media/papers/0153.pdf)
- Discrete adaptive POD (DAPOD) for convection-dominated dissipative PDEs, recursive ROM update as snapshots arrive — [AIChE 2018](https://aiche.org/conferences/aiche-annual-meeting/2018/proceeding/paper/560d-adaptive-model-reduction-dissipative-pde-systems-strong-convective-phenomena)
- A recent POD-ROM MPC paper for Navier-Stokes notes that repeated optimisation with such models "can be prohibitive for real-time implementation" — [arXiv 2511.22123](https://arxiv.org/pdf/2511.22123)
- Deep-learning ROMs for real-time optimal control of parametrised systems — [arXiv 2409.05709](https://arxiv.org/pdf/2409.05709)

### Inferences
- "RLS on a learned basis" is prior art as a mechanism (ALPaCA, Neural-Fly, recursive EDMD, adaptive POD). The distinct element of the target design would be the object being adapted: coefficients of a space-time solution field whose basis was trained with a PDE residual, so that each RLS row can be either a sensor reading or a PDE-residual/boundary row evaluated from cached trunk derivatives. I found no source doing that; this is an absence in my searches, not a proof of absence.
- Stability results transfer only partly. Neural-Fly's guarantee is for a control-affine residual force model with bounded representation error; Koopman results are ISS or ultimate-bound statements under persistent excitation. None covers a PDE field surrogate with RLS-updated coefficients, and the recursive-Koopman preprints explicitly flag covariance wind-up and conditioning as open practical problems. Any RLS claim for the target design needs forgetting-factor / covariance-regulation analysis of its own.
- Cost anchors for "fast enough": 3.0 ms adaptation of 12 coefficients on a Raspberry Pi 4 (Neural-Fly, Python-level stack), 0.28 ms per control move for a 385-dimensional lifted KdV model (Korda-Mezic, 2018 laptop CPU). A compiled RLS update on a few tens of coefficients is O(n^2) and should sit well below both, but that is an estimate, not a published number.
- Neural Galerkin / EDNN are the principled alternative: they do not restrict to a fixed subspace but pay a linear solve with an ill-conditioned M(theta) every step and have no reported runtimes in the text read. They are the baseline to beat on accuracy and the one the target design should beat on step cost.

### Gaps
- One search for recursive / online / sequential least-squares updates of a physics-informed output layer (PIELM-type) returned no hits; this is a single-query absence.
- No update-cost numbers (seconds per step) were found for Neural Galerkin or EDNN in the text read; the Neural Galerkin page was truncated at 100k of 132k characters, so appendices may contain them.
- No a posteriori error or stability theory specific to Neural Galerkin time stepping was found in one targeted search.
- Classical RLS theory (forgetting factor, persistent excitation, covariance resetting) and its guarantees were not sourced here; cite a standard adaptive-control text.
- Whether ALPaCA appeared in a peer-reviewed venue was not verified in this session.
- No timing numbers were found for adaptive POD-ROM MPC in process systems; the papers were identified by snippet only.
- Journal status of arXiv 1804.05291 (Koopman MPC for flows) not checked.

## 3. Physics-informed world models for control: surrogate-trained vs solver-trained policies, and throughput

### Takeaway
Surrogate-vs-true comparisons exist but are few and mixed: PINC matches an ODE/RK model inside NMPC with a modest time saving; a TI-DeepONet policy transfers to a finite-difference plant with 10^-3 to 10^-4 terminal error; but RL agents trained only on surrogates fall short of solver-trained agents (Rayleigh-Benard: Nu 2.97–3.31 vs 2.74; Kuramoto-Sivashinsky: 63% vs 73% reduction), and a plug-flow-reactor NMPC benchmark found PINN surrogates slower than the mechanistic model in every case. No source read reports environment throughput in steps per second.

### Cited Findings

PINC (Antonelo, Camponogara, Seman, de Souza, Jordanou, Hubner) [arXiv 2104.02556; journal version not verified in this session] (full text of arXiv PDF)
- Architecture: PINN with time t, initial state y(0) and control u as inputs, trained for one control interval T and chained in self-loop for long horizons; trained with collocation points only ("not require collecting data from ODE simulators"). Van der Pol: 4 hidden layers x 20 units, T = Ts = 0.5 s; four tanks: 5 x 20, T = Ts = 10 s, prediction and control horizons of 5 steps — [arXiv 2104.02556](https://arxiv.org/pdf/2104.02556)
- Table 1, closed-loop NMPC, total time for the complete control simulation on a desktop (10 repetitions): Van der Pol, PINC RMSE 0.15, IAE 123.6, 3.32 +/- 0.15 s vs ODE/RK RMSE 0.15, IAE 122, 3.41 +/- 0.04 s; four tanks, PINC RMSE 0.811, IAE 876, 10.85 +/- 0.14 s vs ODE/RK RMSE 0.807, IAE 544, 14.15 +/- 0.13 s. The paper calls the four-tank time "23.3% inferior" (i.e. lower) — [arXiv 2104.02556](https://arxiv.org/pdf/2104.02556)
- Stated limitation: PINC "does not have an inherent method to deal with modeling errors and completely reject disturbances"; a steady-state offset is noted and a filtered correction factor suggested — [arXiv 2104.02556](https://arxiv.org/pdf/2104.02556)
- Follow-ups: PINC with skip connections for gas-lifted oil wells, MPC on bottom-hole pressure — [arXiv 2403.02289](https://ar5iv.labs.arxiv.org/html/2403.02289); PINN surrogates for MPC of articulated soft robots — [arXiv 2502.01916](https://arxiv.org/pdf/2502.01916); PINN replacing time integration in NMPC of multi-link manipulators, reported faster with sufficient accuracy — [arXiv 2109.10793](https://arxiv.org/pdf/2109.10793)

PINN / PICNN surrogates inside NMPC for PDE reactors (Elorza Casas, Ricardez-Sandoval, Pulsipher; Univ. of Waterloo) [PP; venue not confirmed] (full text)
- Setup: three plug-flow-reactor benchmarks, direct transcription, Ipopt; embeddings compared: explicit algebraic constraints via OMLT in full space (ECE-FS) or reduced space (ECE-RS), external function evaluation with PyTorch AD (EFE), and shooting. Hardware Intel i9-10980HK @ 2.40 GHz, PyTorch 2.2.1, OMLT 1.1, Pyomo 6.5.0, IPOPT 3.14.9 — [arXiv 2501.06335](https://arxiv.org/pdf/2501.06335)
- Raw model evaluation, benchmark 1: PINN 0.000421 s, PICNN 0.000436 s vs 0.000504 s for direct simulation ("modest speed-up"); benchmark 2 surrogate 0.000253 s ("order-of-magnitude faster"); benchmark 3 0.00162 s ("two orders-of-magnitude faster") — [arXiv 2501.06335](https://arxiv.org/pdf/2501.06335)
- NMPC average wall-clock per step (Table 3). Benchmark 1: mechanistic direct transcription 0.120 s; PINN EFE 0.402 s, ECE-FS 1.180 s, ECE-RS 1.332 s, shooting 4.09 s; PICNN EFE 1.427 s, ECE-FS 5.382 s. Benchmark 2: mechanistic 0.467 s vs PICNN EFE 9.586 s, ECE-FS 20.047 s. Benchmark 3: mechanistic 16.364 s vs PICNN ECE-FS 1995.83 s, EFE 2149.671 s — [arXiv 2501.06335](https://arxiv.org/pdf/2501.06335)
- Conclusion quoted: "the mechanistic NMPC problem is significantly more performant on all benchmarks"; EFE's shortfall is attributed to the modelling language "providing more efficient AD on the mechanistic model than PyTorch can provide for the surrogate" — [arXiv 2501.06335](https://arxiv.org/pdf/2501.06335)
- MPC with PINN models for chemical process setpoint tracking (Patel, Bhartiya, Gudi; ADCHEM 2024) [PR conference] (snippet) — [ADCHEM 2024](https://skoge.folk.ntnu.no/prost/proceedings/adchem-2024/files/0042.pdf)

Operator surrogates in MPC / learned explicit policies
- Laser thermal processing MPC (Halilovic, Zhang, Tartakovsky; arXiv July 2026) [PP] (full text): MS-DeepONet whose branch encodes the future power and scan trajectory and whose trunk encodes the current thermal state (not space-time coordinates); one-shot 5-step prediction. On a 14-core Intel CPU (no GPU, PyTorch 2.7, CasADi 3.7 + IPOPT): surrogate forward pass 0.42 ms vs 515 ms for the five explicit finite-difference steps ("roughly 1.2 x 10^3"); one MPC step 41 ms on average (16 IPOPT iterations on average; 59.6 ms at the 95th percentile, 71.7 ms worst case) — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Same paper, real-time position: control interval 0.125 ms, so the forward pass is 3.4x the interval and one IPOPT solve about 330x; the surrogate "closes three of the roughly four orders of magnitude separating the physics model from real time" and the controller is "presented as an offline design-and-validation tool". Closed loop is simulated against the FDM plant (320 steps, about 50 s per run) — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Same paper, control-readiness: "two surrogates with near-identical offline error behave drastically differently in closed loop"; a two-ensemble data design reduces a 91 K path-corner underprediction to 1.4 K, and a calibrated one-sided margin of 13 K gives zero violations of the true upper bound on all tested paths — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Differentiable predictive control with Time-Integrated DeepONets (Sarkar, Drgona, Goswami; arXiv Nov 2025) [PP] (full text): explicit policy trained offline by backpropagating through the surrogate. Per-step inference, NMPC vs DPC policy: heat 4.3 +/- 1.5 s vs (2.5 +/- 2.2)e-4 s; Burgers 3.9 +/- 13 s vs (8.8 +/- 1.3)e-5 s; Fisher-KPP 7.9 +/- 1.6 s vs (2.4 +/- 2.4)e-4 s; hardware not specified. Tracking MSE: heat (4.2 +/- 6.7)e-3 NMPC vs (7.7 +/- 20)e-3 DPC; Fisher-KPP (1.8 +/- 1.6)e-3 vs (1.4 +/- 1.4)e-3; Burgers policy "approximately one order of magnitude higher loss than NMPC" — [arXiv 2511.08992](https://arxiv.org/html/2511.08992)
- Same paper, surrogate-to-solver transfer: the policy trained on the TI-DeepONet was deployed on the finite-difference solver for heat (terminal errors of order 10^-3) and Fisher-KPP (O(10^-4)); no closed-loop FDM test reported for Burgers; no policy trained directly on the true solver is used as a comparator. Surrogate rollout relative L2: heat (1.64 +/- 1.13)e-2, Burgers 9.24e-2, Fisher-KPP 1.18e-2 — [arXiv 2511.08992](https://arxiv.org/html/2511.08992)
- "Learning to Solve" (2025) [PP] (snippet): neural-operator dynamics plus a learned optimisation proxy, speedups "up to four orders of magnitude" over MPC and adjoint methods — [arXiv 2509.24573](https://arxiv.org/html/2509.24573v1)

Physics-informed model-based RL
- Liu and Wang (Proc. R. Soc. A 477: 20210618, 2021) [PR] (full text): Dyna-style TD3 with a learned transition model (convolutional encoder, LSTM transition, MLP decoder) trained with a physics-informed loss; compared against model-free TD3 on the true environment and a data-only Dyna variant. Cart-pole: reaches return 200 after about 3000 real steps, using "about 45.2% and 9.7%" of the steps needed by the model-based and model-free counterparts. Burgers: "about 65%" of MBRL steps and "46.7%" of MFRL steps. Kuramoto-Sivashinsky: "about 15% of the time steps" of model-free TD3 to reach return -55, where the model-free agent needs more than 400,000 steps — [Proc. R. Soc. A / NSF PAR](https://par.nsf.gov/servlets/purl/10311898); [arXiv 2108.00128](https://arxiv.org/abs/2108.00128v1)
- Same paper, on final performance: for KS, model-free fine-tuning on the real environment is applied once the return exceeds -55 because "model bias becomes the bottleneck", and the curves "will eventually converge to each other"; the authors note that where interactions are cheap and fast, model-free RL "could be more efficient" than Dyna-style MBRL. No wall-clock or steps-per-second figures found — [Proc. R. Soc. A / NSF PAR](https://par.nsf.gov/servlets/purl/10311898)
- Ramesh and Ravindran (L4DC 2023, PMLR 211) [PR] (abstract only): physics-informed dynamics model for rigid-body robots without contact, policy trained by backpropagating through imagined trajectories; better return and sample efficiency in environments sensitive to initial conditions, where "numerical errors accumulate fast"; beats SAC in challenging environments, attributed to analytic policy gradients. No numbers in the abstract — [arXiv 2212.02179](https://arxiv.org/abs/2212.02179v4); [PMLR 211](https://proceedings.mlr.press/v211/ramesh23a.html)
- Werner and Peitz, "Learning a model is paramount for sample efficiency in RL control of PDEs" [PP as seen; venue not confirmed] (full text): KS, L = 22, convolutional autoencoder + ConvLSTM latent model (not physics-informed). "An improvement by a factor of nine" in data vs SAC; but after 50,000 steps the model-based controller reduces dissipation and power by "more than 63%" while "SAC converges to a value of about 73%"; SAC needs about 430,000 samples to break even; no compute times reported — [arXiv 2302.07160](https://ar5iv.labs.arxiv.org/html/2302.07160)
- Plotzki and Peitz, Koopman-type surrogate (linear recurrent autoencoder network) for RL control of Rayleigh-Benard convection (arXiv March 2026) [PP] (full text). Mean Nusselt number (lower is better) and training time: DNS only (400,000 interactions) 2.74, 4 h 11 min; random-action surrogate pretraining + DNS 2.73, 3 h 06 min; policy-aware surrogate + DNS 2.75, 2 h 24 min; surrogate only: random-action 3.31 in 0 h 06 min, policy-aware 2.97 in 0 h 17 min; baselines zero-action 4.00, random 4.05. "surrogate rollouts are on average 25.6 times faster than DNS simulations"; surrogate-only "control performance still remains below that of the DNS-trained agent". Hardware: 20 cores of an Intel Xeon E5-2690v4 plus an NVIDIA Tesla P100 — [arXiv 2603.28074](https://arxiv.org/html/2603.28074v1)
- Physics-informed model and hybrid planning for Dyna-style RL (2024) [PP] (title only) — [arXiv 2407.02217](https://arxiv.org/pdf/2407.02217)

Differentiable simulators and benchmark environments
- SHAC (Xu et al., ICLR 2022) [PR] (snippet): short-horizon actor-critic on a parallel differentiable simulator; truncated windows against exploding/vanishing gradients, a critic to smooth the landscape; better sample efficiency and wall-clock time than model-free baselines — [arXiv 2204.07137](https://arxiv.org/abs/2204.07137v1)
- Analytic policy gradients in differentiable physics — [arXiv 2203.02857](https://arxiv.org/pdf/2203.02857); SAPO / Rewarped for deformables and multiphysics — [arXiv 2412.12089](https://web3.arxiv.org/pdf/2412.12089)
- PDE Control Gym (Bhan et al., L4DC 2024) [PR] (snippet): 1D transport, 1D reaction-diffusion, 2D Navier-Stokes; model-free RL stabilises them at higher cost than PDE backstepping — [arXiv 2405.11401](https://arxiv.org/html/2405.11401v2); [PMLR 242](https://proceedings.mlr.press/v242/bhan24a/bhan24a.pdf)
- controlgym (Zhang et al.) includes PDE environments — [arXiv 2311.18736](https://ar5iv.labs.arxiv.org/html/2311.18736)
- HydroGym: solver-independent RL platform for flow control; PyPI listing states 61 environments across 6 solver backends including two JAX differentiable ones — [arXiv 2512.17534](https://arxiv.org/abs/2512.17534v1); [PyPI](https://pypi.org/project/hydrogym/)
- FluidGym (Becktepe, Franz, Thuerey, Peitz; PMLR 306, ICML 2026) [PR] (snippet): "first standalone, fully differentiable" RL benchmark for active flow control, GPU PICT solver in PyTorch; reports over 16k GPU hours of experiments; no per-step throughput found — [PMLR 306](https://proceedings.mlr.press/v306/becktepe26a.html); [arXiv 2601.15015](https://arxiv.org/html/2601.15015v1)

### Inferences
- Throughput in steps per second is essentially unreported for PDE-control surrogates. The closest quantities found are ratios (25.6x faster rollouts than DNS; 1.2e3x faster forward pass than FDM) and total training wall-clock. A loop-throughput benchmark with fixed hardware, batch size and precision would be a contribution in its own right.
- The honest state of "surrogate-trained vs solver-trained policy": parity is shown for MPC with small ODE systems (PINC) and for an explicit policy on two simple 1D PDEs (TI-DeepONet DPC, without a solver-trained comparator); for RL on chaotic or convective PDEs, surrogate-only training underperforms and the working recipe is surrogate pretraining followed by solver fine-tuning (Liu-Wang, Plotzki-Peitz). A claim of surrogate-only parity on a process PDE would be new but must be tested against a solver-trained policy, not only against NMPC.
- The Waterloo result is a direct warning for the target design: a surrogate that evaluates faster than the PDE can still make the NMPC step 3x to 130x slower (my arithmetic from Table 3: 0.402/0.120 to 2149.671/16.364), because derivative quality and problem structure dominate. A linear-in-c head with cached trunk gives closed-form Jacobians with respect to c, which is the specific property that could avoid the PyTorch-AD penalty identified there; that advantage is plausible but untested in the literature read.
- Where the optimiser dominates (41 ms IPOPT vs 0.42 ms forward pass), making the surrogate faster yields little; gains then come from structure the optimiser can exploit (linearity in lifted coordinates as in Koopman MPC, or an explicit policy as in DPC).
- The laser-MPC finding that equal offline error does not imply equal closed-loop behaviour argues for closed-loop metrics (constraint violations, IAE) as the acceptance test, not field L2 error.

### Gaps
- Journal versions of PINC and of the Waterloo NMPC benchmark were not confirmed; numbers are from the arXiv PDFs.
- Ramesh and Ravindran: only the abstract was read; no quantitative results extracted.
- No source found that reports RL environment steps per second for a PINN or neural-operator surrogate, nor for PDE Control Gym, controlgym, HydroGym or FluidGym (searched; not found in snippets; papers not opened in full).
- No PINN/neural-operator-in-MPC study on spray dryers or evaporators was searched for specifically; the process-systems coverage here is plug-flow reactors, four-tank, gas-lift wells and a generic ADCHEM paper.
- Per-step MPC time for PINC is not reported, only total simulation time; the number of MPC steps per run would be needed to convert.
- The DPC paper does not state its hardware.

## 4. Stability of long rollouts of learned PDE surrogates

### Takeaway
Error accumulation in autoregressive surrogates is well documented and addressed mainly by training tricks (pushforward, temporal bundling, recurrent training), spectral fixes (PDE-Refiner, architectural changes in McCabe et al.) and one-shot horizon prediction; formal guarantees are rare and come mostly from the Koopman and adaptive-control side rather than from PDE surrogates.

### Cited Findings
- Message Passing Neural PDE Solvers (Brandstetter, Worrall, Welling; ICLR 2022) [PR] (snippet): instability described as small errors accumulating over rollouts and as overfitting to the one-step training distribution; pushforward trick = unroll two steps, backpropagate only through the last; linked to zero-stability; temporal bundling predicts several steps at once — [arXiv 2202.03376](https://arxiv.org/pdf/2202.03376)
- PDE-Refiner (Lippe, Veeling, Perdikaris, Turner, Brandstetter; NeurIPS 2023) [PR] (snippet): identifies neglect of non-dominant, typically high-frequency, spatial components as the main limit on stable rollouts; diffusion-style multistep refinement; reported to outperform neural, numerical and hybrid baselines on fluid benchmarks — [arXiv 2308.05732](https://ar5iv.labs.arxiv.org/html/2308.05732); [NeurIPS 2023 paper](https://papers.neurips.cc/paper_files/paper/2023/file/d529b943af3dba734f8a7d49efcb6d09-Paper-Conference.pdf)
- Towards Stability of Autoregressive Neural Operators (McCabe et al.; TMLR per search summary) [PR, venue from snippet]: traces error growth to specific instability-inducing operations and proposes architectural changes; lower long-horizon error with no visible divergence on Navier-Stokes, rotating shallow water and weather — [arXiv 2306.10619](https://arxiv.org/abs/2306.10619v2)
- Recurrent Neural Operators (2025) [PP] (snippet): claims recurrent training reduces worst-case error growth from exponential to linear in the horizon, at higher training cost — [arXiv 2505.20721](https://arxiv.org/pdf/2505.20721)
- Spectral Generator Neural Operator (2026) [PP] (snippet): residual operator with an explicit Fourier-domain update and a "generator constraint" for stable long rollouts of semilinear PDEs — [arXiv 2602.18801](https://arxiv.org/pdf/2602.18801)
- Jacobian-adaptive weighting for stability (2026) [PP] (snippet): frames accumulation as drift from the single-step training distribution; spatially adaptive regularisation — [arXiv 2603.05538](https://arxiv.org/pdf/2603.05538)
- Energy-dissipative evolutionary KANs for PDE systems (2025) [PP] (title only) — [arXiv 2503.01618](https://arxiv.org/pdf/2503.01618)
- One-shot horizon prediction as avoidance: the laser-MPC paper notes recurrent surrogates "compound single-step errors along the horizon" and uses one-shot multi-step output for that reason — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Chained physics-informed surrogates: PINC motivates its design by the observation that plain PINNs cannot "simulate for long-range intervals without serious degradation", and chains short-interval predictions, with the true state reset by measurement at each MPC step — [arXiv 2104.02556](https://arxiv.org/pdf/2104.02556)
- Sequential-in-time training: RSNG reports that randomised sparse updates help prevent error accumulating over sequential-in-time training — [arXiv 2310.04867](https://ar5iv.labs.arxiv.org/html/2310.04867)
- Sensitivity matters by system: Ramesh and Ravindran report model accuracy matters most in environments sensitive to initial conditions "where numerical errors accumulate fast" — [arXiv 2212.02179](https://arxiv.org/abs/2212.02179v4)
- Guarantees on the control side: Neural-Fly exponential convergence to an error ball; deep Koopman ISS; ASACK ultimate bound — [arXiv 2205.06908](https://ar5iv.arxiv.org/html/2205.06908); [arXiv 2601.21230](https://arxiv.org/html/2601.21230); [arXiv 2605.09659](https://arxiv.org/pdf/2605.09659)

### Inferences
- A space-time basis u = sum c_j(z) phi_j(x,t) queried directly at t is not autoregressive inside one window, so it sidesteps accumulation within the window; accumulation re-enters when windows are chained (PINC-style) or when z contains the previous state. Measurement resets (MPC) and RLS correction bound it in practice, but nothing found proves a bound.
- With a linear head, the one-window state-transition map restricted to the basis is a matrix in coefficient space when the PDE is linear in u; its spectral radius can be checked or constrained directly. That is a cheaper route to a contractivity certificate than for general autoregressive operators, and I found no paper exploiting it for PINN-type bases (inference; unverified as novel).
- The field's remedies are almost all empirical; a design that ships with an a posteriori residual indicator (the PDE residual is cheap from cached trunk derivatives) would be reporting something most rollouts papers do not.

### Gaps
- Lyapunov-constrained, energy-preserving (Hamiltonian/Lagrangian/port-Hamiltonian), and contractive architectures were named in the brief but my single search returned no primary sources on them; no sourced findings. Needs a dedicated search (stable deep dynamics models, contraction-constrained RNNs, structure-preserving operator inference).
- No quantitative rollout-horizon numbers (e.g. correlation time gains from pushforward or PDE-Refiner) were extracted; those papers were seen only through search summaries.
- No source found analysing rollout stability of DeepONet-type space-time models when chained across windows.

## 5. Deployment constraints for inference in control loops

### Takeaway
Hard numbers for PINN or neural-operator inference on industrial controllers are almost absent; reported latencies are milliseconds on MCUs and single-board computers, microseconds only on FPGAs or GPUs, and I found no PINN on a PLC nor any hardware-in-the-loop PINN controller. Process-control loop periods (order 100 ms to 1 s) are loose compared with these, so the binding constraint in process plants is more likely determinism, export path and certification than raw latency.

### Cited Findings
- Timing context (from a preprint on OT network security, secondary for this purpose): PLC scan cycles typically 1 to 10 ms, process-control loops 100 ms to 1 s, SCADA polling commonly 1000 ms — [arXiv 2608.16769](https://arxiv.org/pdf/2608.16769)
- Beckhoff TwinCAT Machine Learning: distinguishes model execution inside the TwinCAT real-time runtime (hard real-time) from execution in a server process (near real-time); for the server, total latency = ADS communication + server overhead + inference + delay from cyclic PLC execution, reducible only by shortening the task cycle — [Beckhoff product page](https://www.beckhoff.com/en-nz/products/automation/twincat-3-machine-learning/); [Beckhoff Infosys latency page](https://infosys.beckhoff.com/content/1033/tf3820_tc3_machine_learning_server/19185017611.html)
- Native IEC 61131-3 inference: a study implements regression, kNN and a neural network in Structured Text on a Siemens S7-1516 with acceptable cycle times; ICSML runs inference in Structured Text on any IEC 61131-3 device — [Deusto thesis/paper](https://deustoteka.deusto.es/bitstreams/1afb8ac4-2f84-4d20-81ef-cc1e7c0f0c3d/download); [arXiv 2202.10075](https://arxiv.org/pdf/2202.10075v3.pdf)
- CNNs found too heavy for real-time execution on industrial control hardware without acceleration — [Neural Computing and Applications](https://link.springer.com/article/10.1007/s00521-021-05848-4)
- LSTM soft sensor with 517 parameters running inside SIMATIC PCS 7 on Siemens PLCs; continual learning is offline, no update at inference time (KDMiLe 2026) — [SBC proceedings](https://sol.sbc.org.br/index.php/kdmile/article/download/44988/44751/)
- FPGA soft sensors: inference times from 1.04 to 12.04 microseconds for fluid-flow estimation from level data (not a PINN, not a PLC) — [arXiv 2311.15036](https://arxiv.org/pdf/2311.15036)
- PINN digital twin for binary distillation (2026) [PP]: argues physics constraints keep inference fast enough for online use; no PLC deployment reported in the snippet — [arXiv 2603.24644](https://arxiv.org/html/2603.24644v1)
- PINN monitoring benchmark for offshore wind: 0.381 ms mean GPU inference; embedded validation left to future work — [arXiv 2606.24176](https://arxiv.org/pdf/2606.24176)
- Approximate MPC on microcontrollers: imitation network "small enough to evaluate in milliseconds on an MCU", validated on real cartpole swing-ups; notes that explicit-MPC toolchains export standalone C for linear systems but the nonlinear case is harder — [arXiv 2404.05835](https://arxiv.org/abs/2404.05835v1)
- Quantised path-following network under 5 kB on an MCU, faster than the OpEn solver — [arXiv 2301.11909](https://arxiv.org/pdf/2301.11909v1.pdf)
- Deep-learning approximation of MPC with hardware-in-the-loop demonstration on embedded hardware, plus FPGA implementation (2018) — [arXiv 1810.04872](https://arxiv.org/pdf/1810.04872.pdf); [arXiv 1806.10644](https://www.arxiv.org/pdf/1806.10644)
- Review of neural networks for fast optimisation in MPC (Gonzalez Arango et al.) — [arXiv 2309.02668](https://arxiv.org/html/2309.02668v3)
- CPU-only reference points from sections above: 0.42 ms forward pass for an MS-DeepONet on a 14-core Intel machine "without any GPU acceleration, code generation, or solver specialization" — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289); about 3.5 ms inference and 3.0 ms adaptation on a Raspberry Pi 4 — [arXiv 2205.06908](https://ar5iv.arxiv.org/html/2205.06908); 0.28 ms per control move for Koopman MPC of KdV on a 2 GHz laptop CPU — [arXiv 1611.03537](https://ar5iv.arxiv.org/html/1611.03537)

### Inferences
- For process control at 100 ms to 1 s periods, every surrogate above is fast enough for a single evaluation; the budget is consumed by the optimiser (tens of iterations, each needing values and gradients) or by RL training volume (10^5 to 10^6 steps), not by one inference. Microsecond inference therefore pays off in training throughput and in MHE/MPC inner iterations, not in the plant loop itself.
- A linear head on cached features is the easiest neural model to export: the deployed artefact is a small matrix-vector product plus an RLS update, expressible in Structured Text or plain C without a runtime. This is an inference from the Beckhoff/ICSML material, not a published demonstration.
- Because no PINN-on-PLC or PINN hardware-in-the-loop result was found, even a modest HIL demonstration with measured jitter would be a first-of-kind data point (subject to the search limits noted below).

### Gaps
- No source found for PINN or neural-operator inference on a PLC, nor for HIL validation of a PINN-based controller or estimator (two searches). Absence may reflect search coverage; IFAC and J. Process Control were not searched directly.
- No authoritative process-control source (ISA standard, vendor DCS documentation, textbook) was obtained for loop-period budgets; the 1–10 ms / 100 ms–1 s figures come from a security preprint and should be replaced by a primary reference.
- No published microsecond-level CPU inference numbers for PINN-type models were found; the microsecond figures are FPGA (non-PINN) or GPU (HyperPINN).
- Export toolchains (ONNX to C, CasADi code generation, acados, TwinCAT ONNX import limits, MATLAB/Simulink PLC Coder) were not researched.
- Certification and functional-safety constraints (IEC 61508/61511) for learned models in the loop were not researched.

## 6. What remains unsolved, and what would be new for the target architecture

### Takeaway
Each ingredient exists separately (bilinear trunk/branch form, least-squares-fitted coefficients, online last-layer adaptation with guarantees in adaptive control, surrogates in MPC/RL); I found no work that combines a physics-trained cached space-time trunk, an RLS-updated coefficient head, closed-form action gradients and a compiled engine, and no benchmark of loop throughput. The strongest negative evidence in the literature is that surrogate-in-the-loop frequently fails to beat the mechanistic model or the solver-trained policy, so any novelty claim has to be carried by closed-loop comparisons.

### Cited Findings
- Closed-form / analytic gradients with respect to actions: analytic policy gradients are credited for outperforming SAC in physics-informed MBRL, but obtained by backpropagation through imagined trajectories, not in closed form — [arXiv 2212.02179](https://arxiv.org/abs/2212.02179v4); SHAC needs truncated windows and a critic because long-horizon backpropagated gradients explode or vanish — [arXiv 2204.07137](https://arxiv.org/abs/2204.07137v1)
- Derivative quality is the bottleneck in surrogate NMPC: PyTorch AD on the surrogate is less efficient than the modelling language's AD on the mechanistic model, and the mechanistic NMPC is faster on all three benchmarks — [arXiv 2501.06335](https://arxiv.org/pdf/2501.06335)
- Smoothness requirements for optimisers: ReLU surrogates "were observed to stall or destabilize IPOPT" and had to be replaced by a smooth rectifier to give a C^2 constraint function — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Online adaptation with guarantees: available for control-affine residual models (Neural-Fly) and, as preprints, for Koopman models (ISS, ultimate bounds under persistent excitation); recursive Koopman papers state that convergence/complexity analyses are lacking and that recursive updates have overlooked numerical vulnerabilities — [arXiv 2205.06908](https://ar5iv.arxiv.org/html/2205.06908); [arXiv 2509.08241](https://arxiv.org/html/2509.08241v1); [arXiv 2606.15317](https://arxiv.org/html/2606.15317)
- No disturbance rejection or model-error handling in PINC as published — [arXiv 2104.02556](https://arxiv.org/pdf/2104.02556)
- Open-loop accuracy is "necessary but not sufficient for control-readiness" — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Surrogate-only RL remains below solver-trained RL on convective/chaotic PDEs — [arXiv 2603.28074](https://arxiv.org/html/2603.28074v1); [arXiv 2302.07160](https://ar5iv.labs.arxiv.org/html/2302.07160)
- Co-design with compiled engines: the laser-MPC timings are explicitly "without any GPU acceleration, code generation, or solver specialization", i.e. the compiled path is named as future work rather than done — [arXiv 2607.13289](https://arxiv.org/pdf/2607.13289)
- Benchmarks: existing PDE-control suites standardise tasks and solvers (PDE Control Gym, controlgym, HydroGym, FluidGym) but no throughput metric was found in their summaries — [arXiv 2405.11401](https://arxiv.org/html/2405.11401v2); [arXiv 2512.17534](https://arxiv.org/abs/2512.17534v1); [arXiv 2601.15015](https://arxiv.org/html/2601.15015v1)
- Fixed-basis limits: Hyper-LR-PINN's stated scope is PDE coefficients only (not initial/boundary conditions), and MAML beats it on the combined CDR case, showing no conditioning scheme dominates — [arXiv 2310.09528](https://ar5iv.labs.arxiv.org/html/2310.09528)

### Inferences
What already exists (do not claim):
- The representation u = sum c_j(z) phi_j(x,t) (DeepONet, POD-DeepONet, two-step DeepONet).
- Least-squares determination of coefficients on a fixed learned basis (two-step DeepONet, PIELM, random-feature methods).
- Online RLS / Kalman / composite adaptation of last-layer coefficients on a meta-learned basis, with stability theorems, in robotics.
- Offline-basis / online-update ROMs inside MPC for distributed process systems (adaptive POD, Armaou group).
- Surrogates inside MPC and Dyna-style RL, including physics-informed ones.

What appears open (candidate contributions, each conditional on the search limits stated in Gaps):
1. RLS on the coefficients of a physics-trained space-time basis, where update rows mix sensor data and PDE-residual rows computed from cached trunk derivatives, used inside MPC/MHE/RL. No instance found.
2. Closed-form du/dz and d(cost)/d(action) through a linear head (du/dz = sum_j phi_j * dc_j/dz, with the trunk constant), exposed to the optimiser or policy gradient without reverse-mode AD through the trunk. The Waterloo benchmark isolates AD cost as the reason surrogates lose; nobody found addresses it architecturally.
3. A spectral-radius or contractivity certificate for chained windows computed in coefficient space.
4. Co-design with a compiled kernel: cached trunk plus head as a fixed-size dense kernel, with measured CPU latency at batch 1 and jitter. Published timings are GPU or Python-stack CPU.
5. A loop-throughput benchmark (environment steps per second and MPC iterations per second, fixed hardware, with closed-loop return/IAE alongside), which no PDE-control suite found reports.
6. A surrogate-only policy that matches a solver-trained policy on a process PDE, with the comparison actually run; current RL evidence says this fails on harder PDEs, so a positive or a carefully characterised negative result are both publishable.

Risks the literature points to:
- Fixed-subspace expressiveness on transport-dominated dynamics (evaporator/dryer fronts): expect the beta = 30 convection failure class unless the basis is enriched or z modulates more than the last layer.
- Surrogate speed not translating to loop speed when the optimiser dominates.
- RLS wind-up and conditioning without persistent excitation.
- Equal field error not implying equal closed-loop behaviour.

### Gaps
- All novelty statements above are bounded by roughly 45 searches/fetches in one session; IFAC-PapersOnLine, Journal of Process Control, Computers and Chemical Engineering and IEEE L-CSS were not searched by venue, and Google Scholar forward-citation checks on DeepONet, PINC and Neural-Fly were not done. A targeted check for "DeepONet + recursive least squares", "physics-informed Bayesian last layer", and "online operator learning adaptation control" is advisable before a claim of novelty goes into a paper.
- Not researched: certified reduced-basis methods and their a posteriori bounds; operator inference (Peherstorfer and Willcox) as a linear-in-coefficients competitor; SINDy with control; Gaussian-process and kernel world models; LNN/HNN structure-preserving models; neural-ODE latent models (PlaNet/Dreamer lineage) for PDE control.
- Several sources dated 2026 (arXiv 26xx identifiers) are very recent preprints seen only through search summaries; their claims are unreviewed and should be re-read before citation.
- Arithmetic performed by me, not published: 5.7x trunk-caching ratio; 3x to 130x NMPC slowdown ratios from the Waterloo Table 3; speedup ratios of roughly 1.7e4, 4.4e4, 3.3e4 for the DPC policy vs NMPC.
