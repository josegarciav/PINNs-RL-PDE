# Accepted PINN benchmarks and best published accuracy / timing records

Reading notes (retrieved 2026-11-19). Conventions: "PR" = peer-reviewed venue confirmed in the source I read; "preprint" = arXiv only as far as I could confirm. Every number is quoted as published, with its own problem definition; numbers in different rows are NOT comparable unless stated. Where I read a paper through an automated HTML/PDF extraction, column mapping risks are flagged.

## 1. What benchmark suites exist, what do they contain and measure, and which report wall-clock time and hardware?

### Takeaway
Only PINNacle is a PINN-method benchmark in the strict sense (22 tasks, 3 seeds, a runtime table in seconds, but no hardware model that I could find); the jaxpi / PirateNets line is the de facto accuracy leaderboard but reports final relative L2 error with at most ablation run times and no GPU model in the Expert's Guide. PDEBench, PDEArena, CFDBench and The Well are data-driven surrogate benchmarks where a PINN is at most one weak baseline, and their timing is "inference vs simulation", not "training time to a given error".

### Cited Findings

**PINNacle (Hao et al.) — PR, NeurIPS 2024 per the official repository**
- The repository states the paper is "accepted to NeurIPS 2024"; the arXiv PDF I extracted still carries "Preprint, under review" in its header (v1 15 Jun 2023, revised 5 Oct 2023) — [PINNacle GitHub](https://github.com/i207M/PINNacle); [arXiv 2306.08827](https://arxiv.org/abs/2306.08827)
- 22 tasks: Burgers1d-C, Burgers2d-C; Poisson2d-C, 2d-CG, 3d-CG, 2d-MS; Heat2d-VC, 2d-MS, 2d-CG, 2d-LT; NS2d-C (lid-driven), NS2d-CG (backward step), NS2d-LT; Wave1d-C, 2d-CG, 2d-MS; chaotic GS and KS; high-dimensional PNd, HNd (5-D); inverse PInv, HInv. Reference data from COMSOL FEM — [arXiv 2306.08827](https://arxiv.org/pdf/2306.08827)
- Methods: PINN, PINN-w, LBFGS, LRA, NTK, RAR, MultiAdam, gPINN, hp-VPINN, LAAF, GAAF, FBPINN — [arXiv 2306.08827](https://arxiv.org/pdf/2306.08827)
- Protocol: MLP 5 layers x 100 width, Adam lr 1e-3, 20,000 epochs, 8,192 interior + 2,048 boundary/initial points in 2-D, each experiment run three times, mean and std reported — [arXiv 2306.08827 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2306.08827)
- Metrics: L2RE, L1RE, MSE, max error, low/mid/high-frequency Fourier error, and average runtime (Tables 5-12) — [arXiv 2306.08827](https://arxiv.org/pdf/2306.08827)
- Headline: "Using an l2 relative error (L2RE) of 10% as a threshold for a successful solution, we find that vanilla PINN only solves 10 out of 22 tasks" — [arXiv 2306.08827](https://arxiv.org/pdf/2306.08827)
- Runtime IS reported (Table 12, "Average running time (seconds) for main experiments, we run all methods three times with 20000 epochs"). Vanilla PINN values: Burgers1d-C 2.84E+2 s; Burgers2d-C 3.11E+3; Poisson2d-C 3.39E+2; Heat2d-VC 1.16E+3; NS2d-C 7.52E+2; Wave1d-C 3.50E+2; GS 2.55E+3; KS 1.40E+3; PInv 4.53E+2; HInv 1.09E+3. FBPINN on Burgers1d-C 1.11E+3 s; hp-VPINN on Burgers1d-C 2.85E+2 s. The text says hp-VPINN is fastest in most problems — [arXiv 2306.08827](https://arxiv.org/pdf/2306.08827)
- Hardware: I found no GPU model in the extracted text; only a "multi-GPU parallel training" framework is described — [arXiv 2306.08827](https://arxiv.org/pdf/2306.08827)

**"An Expert's Guide to Training PINNs" (Wang, Sankaran, Wang, Perdikaris 2023) and jaxpi — preprint as read**
- Table 1 of the paper (relative L2): Allen-Cahn 5.37e-5; advection 6.88e-4; Stokes 8.04e-5; Kuramoto-Sivashinsky 1.61e-1; lid-driven cavity Re=3200 1.58e-1; Navier-Stokes in a torus 2.45e-1; flow around a cylinder not reported (no reliable reference for shedding onset) — [arXiv 2308.08468](https://arxiv.org/html/2308.08468)
- The jaxpi README lists the same values except NS in tori 3.53e-1 (paper: 2.45e-1) — a discrepancy between paper and repository — [jaxpi README](https://github.com/PredictiveIntelligenceLab/jaxpi)
- Run time is reported only for ablation runs, in minutes: Allen-Cahn full pipeline 16.26 min vs plain PINN 12.93 min; advection 9.18 vs 7.12; Stokes 9.51 vs 7.95; Kuramoto-Sivashinsky 13.33 vs 7.12. Default ablation setup: MLP 4 x 256, tanh, Adam lr 1e-3 with exponential decay 0.9 every 2,000 steps, batch 4,096. No GPU model is stated in the text read; the paper notes GPU run time varies with load. A weak-scaling study up to 256 GPUs is mentioned — [arXiv 2308.08468](https://arxiv.org/html/2308.08468)
- Internal inconsistency to respect: KS 1.61e-1 in Table 1 is the long chaotic run; the best KS ablation value (1.42e-4) is a different, short-horizon configuration — [arXiv 2308.08468](https://arxiv.org/html/2308.08468)
- jaxpi is GPU-only ("multi-GPU training is supported"), tested on JAX 0.4.36 / CUDA 12.4; the README gives no training-time figures; logs are public on Weights & Biases — [jaxpi README](https://github.com/PredictiveIntelligenceLab/jaxpi)

**PirateNets (Wang, Li, Chen, Perdikaris) — PR, JMLR 25 (2024) 1-51, published 12/24**
- Table 1 (PirateNet vs JAX-PI): Allen-Cahn 2.24e-5 vs 5.37e-5; Korteweg-De Vries 4.27e-4 vs 1.96e-3; Grey-Scott 3.61e-3 vs "6.13" (printed without exponent); Ginzburg-Landau 1.49e-2 vs 3.20e-2; lid-driven cavity Re=3200 4.21e-2 vs 1.58e-1 — [JMLR v25 24-0313](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)
- Table 9, PirateNet on PINNacle tasks (PINNacle best / PIELM / JAX-PI / PirateNet): Burgers 1D-C 1.33e-2 / 1.63e-1 / 3.58e-4 / 8.20e-5; Wave 1D-C 9.79e-2 / 8.58e-5 / 9.80e-5 / 5.15e-5; NS 2D-C 3.60e-2 / N/A / 2.39e-3 / 9.52e-4; NS 2D-CG 8.24e-2 / N/A / 4.10e-2 / 2.10e-2; Poisson inverse 1.96e-2 / N/A / 1.99e-2 / 5.34e-3. The authors state JAX-PI gives "about two orders of magnitude" better accuracy on forward problems than PINNacle's reported results — [JMLR v25 24-0313](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)
- No wall-clock time or GPU model: the paper only says the cost "is comparable to that of training a Modified MLP"; comparisons are averaged over five random seeds — [JMLR v25 24-0313](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)

**Causal training (Wang, Sankaran, Perdikaris 2022) — preprint as read; one of the few accuracy papers that gives throughput and hardware**
- Hardware NVIDIA RTX A6000; reported iterations/second: Allen-Cahn 120.30 (MLP) and 58.42 (modified MLP); Lorenz 957.41; Kuramoto-Sivashinsky 164.77 (regular) and 28.22 (chaotic); Navier-Stokes 68.29 — [arXiv 2203.07404 (ar5iv)](https://ar5iv.labs.arxiv.org/html/2203.07404)

**PDEBench (Takamoto et al.) — NeurIPS 2022 Datasets and Benchmarks**
- 11 PDEs / 35 datasets: 1-D advection, Burgers, diffusion-reaction, diffusion-sorption; 2-D diffusion-reaction, Darcy, shallow water, incompressible NS; 1/2/3-D compressible NS. Baselines U-Net, FNO, PINN (DeepXDE). Metrics RMSE, nRMSE, max error, cRMSE, bRMSE, fRMSE (low/mid/high) — [arXiv 2210.07182](https://arxiv.org/html/2210.07182v7); [NeurIPS 2022 D&B paper](https://papers.neurips.cc/paper_files/paper/2022/file/0a9747136d411fb83f0cf81820d44afb-Paper-Datasets_and_Benchmarks.pdf)
- Its PINN baseline is deliberately plain: 15,000 epochs, Adam lr 1e-3, depth 6 x 40 neurons; PINNs omitted for 2-D/3-D compressible NS and Darcy — [arXiv 2210.07182](https://arxiv.org/html/2210.07182v7)
- Timing table is for FNO, seconds (simulation / ML inference / ML training): diffusion-sorption (1024) 59.83 / 0.32 / 48,760; 2-D CFD 512^2 582.61 / 0.14 / 107,567; 3-D CFD 64^3 60.06 / 0.14 / 12,387 — [arXiv 2210.07182](https://arxiv.org/html/2210.07182v7)

**The Well (Ohana et al.) — NeurIPS 2024 Datasets and Benchmarks**
- "15TB of data across 16 datasets"; the abstract mentions example baselines; no PINNs and no runtime comparison on the abstract page — [arXiv 2412.00568](https://arxiv.org/abs/2412.00568)
- Secondary (search summary of a third paper): The Well uses VRMSE (RMSE normalised by the target's variance) while PDEBench uses nRMSE, so the two suites' numbers are not comparable — [MORPH, arXiv 2509.21670](https://arxiv.org/html/2509.21670v1)

**CFDBench — preprint as read**
- Four flow problems (lid-driven cavity, laminar boundary layer, dam break, flow around a cylinder); models take 5 to 10 ms per forward pass vs about 1 second per frame for the generating numerical method (search-result summary; hardware not confirmed) — [arXiv 2310.05963](https://arxiv.org/html/2310.05963)

**Other suites surfaced**
- PDENNEval (IJCAI 2024) evaluates 12 neural PDE methods with efficiency metrics including training time, inference time and convergence time (search-result summary only) — [IJCAI 2024 proceedings](https://ijcai.org/proceedings/2024/0573.pdf)

### Inferences
- There are two disjoint "accepted" protocols: (a) PINNacle: fixed small budget (20k Adam epochs, 5x100 MLP), 3 seeds, many tasks, runtime in seconds; (b) jaxpi/PirateNets: per-problem tuned pipeline, 1e5-5e5 steps, 256-wide nets, 5 seeds for ablations, final error only. The same task (Burgers1d-C) differs by 2+ orders of magnitude between them (1.33e-2 vs 8.20e-5), so the protocol, not the method, dominates the number.
- No suite reports error as a function of wall-clock time; PINNacle reports one (error, time) point per method at a fixed epoch count. A benchmark that publishes the full error-vs-time curve would be new relative to all suites above.
- No suite states a CPU timing. Every timing found is on a GPU, usually without model.

### Gaps
- PDEArena: I retrieved nothing first-hand (only a one-line search summary that it covers incompressible NS, shallow water and Maxwell). Treat as unverified.
- PINNacle hardware (GPU model) not found in the extracted text; the PINNacle runtime table extraction lacked clean LBFGS/RAR columns, so I only quote columns I could map unambiguously.
- PDEBench PINN error values and PINN training times are in appendices I did not read.
- I did not verify the peer-reviewed venue of the Expert's Guide or the causality paper (the latter is commonly cited as CMAME 2024; not confirmed here).
- CFDBench and The Well details beyond the abstract are second-hand.

## 2. Best published relative L2 errors per equation (method, network, time, hardware)

### Takeaway
On the equations with a shared definition (Burgers nu=0.01/pi, Raissi/Wu Allen-Cahn, Raissi Schrödinger) the frontier has moved from 1e-3..1e-4 (2019) to about 1e-5 with first-order pipelines (jaxpi, PirateNets) and to 1e-7..1e-8 with double-precision quasi-Newton (SSBroyden) at 2,000-3,000 s on a GPU with ~2-3k parameters; 2026 preprints claim far beyond that but are unverified. For 1-D heat, 2-D heat/Poisson and Cahn-Hilliard there is no shared problem definition and therefore no meaningful "record".

### Cited Findings

**Burgers, nu = 0.01/pi, x in [-1,1], t in [0,1], u0 = -sin(pi x) (Raissi form)**

| Rel. L2 | Method | Network | Budget / time | Hardware | Status | Source |
|---|---|---|---|---|---|---|
| 6.7e-4 | PINN, L-BFGS, Nu=100, Nf=10,000 | 9 layers x 20 (3,021 params) | "approximately 60 seconds" | single NVIDIA Titan X | PR (JCP 2019) | [Raissi et al. 2019](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf) |
| 8.2e-4 | discrete-time PINN, one implicit RK step to t=0.9 | 4 x 50 | not given | not given | PR (JCP 2019) | [Raissi et al. 2019](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf) |
| 1.45e-2 (std 1.59e-3); best variant 1.33e-2 (LBFGS column of Table 3) | PINNacle Burgers1d-C, 20k Adam epochs | 5 x 100 | 2.84E+2 s (vanilla) | GPU, model not found | PR (NeurIPS 2024) | [PINNacle](https://arxiv.org/pdf/2306.08827) |
| 3.58e-4 (JAX-PI), 8.20e-5 (PirateNet) | PINNacle Burgers 1D-C task | 256 wide | not given | not given | PR (JMLR 2024) | [PirateNets](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| 1.62e-8 | Adam 1,000 + SSBroyden (Wolfe) 30,000, double precision | 8 x 20, 3,021 params | 2,812 s | RTX A6000, TensorFlow | preprint as read (Kiyani, Shukla, Urbán, Darbon, Karniadakis) | [arXiv 2501.16371](https://arxiv.org/html/2501.16371) |
| 1.77e-8 / 1.79e-8 | PIKAN (Chebyshev deg 3), SSBFGS / SSBroyden, 50,000 its | 4 x 20, 5,040 params | 12,796 s / 14,169 s | RTX A6000 | preprint as read | [arXiv 2501.16371](https://arxiv.org/html/2501.16371) |
| 2.0e-6 | JAX/Optax/Optimistix, RAdam + BFGS, 30,000 its | 8 x 20, 3,501 params | 1,587 s | RTX 3090 | preprint as read | [arXiv 2501.16371](https://arxiv.org/html/2501.16371) |
| 1.67e-7 -> 1.51e-8 | SSBroyden without / with variational residual-based adaptivity (vRBA) | not extracted | not extracted | not extracted | preprint (search summary only) | [arXiv 2509.14198](https://arxiv.org/pdf/2509.14198) |
| 2.0e-4 | Levenberg-Marquardt (MATLAB fsolve), <=4,000 its | NN(2,25,25,1), 751 params | 2,281 s (BFGS: 2.2e-2 in 4,442 s) | Intel Core i5-10400 CPU, no GPU | preprint | [arXiv 2602.08515](https://arxiv.org/html/2602.08515v2) |
| 4.75e-7 "in under ten seconds" (single precision); Burgers improved by "five orders of magnitude" (double) | DSGNAR, doubly-sketched Gauss-Newton | not on abstract page | under 10 s | not on abstract page | preprint, 2 Jul 2026, abstract only | [arXiv 2607.02194](https://arxiv.org/abs/2607.02194) |

**Allen-Cahn (Raissi/Wu form: u_t - 0.0001 u_xx + 5u^3 - 5u = 0, periodic, u0 = x^2 cos(pi x))**

| Rel. L2 | Method | Network | Budget / time | Hardware | Status | Source |
|---|---|---|---|---|---|---|
| 6.99e-3 | discrete-time PINN (implicit RK, one step to t=0.9) — NOT the continuous space-time problem | 4 x 200 | not given | not given | PR (JCP 2019) | [Raissi et al. 2019](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf) |
| 4.98e-1 | "original formulation of Raissi et al." applied continuous-time (as tabulated by later papers) | — | — | — | PR table | [PirateNets Table 2](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| 2.33e-2 / 2.10e-2 / 1.68e-2 | adaptive time sampling (Wight and Zhao) / self-attention (McClenny and Braga-Neto) / time marching (Mattey and Ghosh) | — | — | — | as tabulated | [PirateNets Table 2](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| 1.43e-3 (MLP), 1.39e-4 (modified MLP; an appendix caption says 2.46e-4) | causal training | depth 6, width 128; 3e5 iterations | 120.30 it/s (MLP), 58.42 it/s (mod. MLP), i.e. roughly 42 min and 86 min by my division | RTX A6000 | preprint as read | [arXiv 2203.07404](https://ar5iv.labs.arxiv.org/html/2203.07404) |
| 6.29e-5 | Dirac-delta causal training (Es'kin et al. 2023) | — | — | — | as tabulated | [PirateNets Table 2](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| 5.37e-5 | JAX-PI full pipeline | 4 x 256 (ablation default) | ablation run 16.26 min | GPU, model not stated | preprint as read | [arXiv 2308.08468](https://arxiv.org/html/2308.08468) |
| 4.55e-5 | RBA-PINNs (Anagnostopoulos et al. 2023) | — | — | — | as tabulated | [PirateNets Table 2](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| 2.24e-5 | PirateNet | 3 residual blocks (9 hidden layers), 256 channels | not given | not given | PR (JMLR 2024) | [PirateNets](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf) |
| 9.43e-7 (double); 4.73e-4 (single) | Adam 5,000 + SSBroyden 30,000 (20,000 in single) | 3 x 30, 2,019 params | 2,000 s (double); 762 s (single) | not stated for this case | preprint as read | [arXiv 2501.16371](https://arxiv.org/html/2501.16371) |
| 9.01e-6 | Chebyshev PIKAN + SSBroyden | 4 x 10 deg 5, 1,368 params | 2,239.38 s | not stated | preprint as read | [arXiv 2501.16371](https://arxiv.org/html/2501.16371) |

**Korteweg-de Vries**
- Forward problem (jaxpi/PirateNets definition): Dirac-delta causal training 2.45e-3; JAX-PI 1.96e-3; PirateNet 4.27e-4. PirateNet config: 9 layers x 256 channels, tanh, 2e5 steps, batch 4,096, grad-norm weighting, causal tolerance 1.0, 16 chunks. No time or hardware — [PirateNets Tables 3 and 5](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)
- Raissi 2019 treats KdV only as an inverse (identification) problem with discrete-time PINNs: parameter errors 0.023% and 0.006% (noise-free), 0.057% and 0.017% (1% noise); network 4 hidden layers x 50 — [Raissi et al. 2019](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf)

**Kuramoto-Sivashinsky (definitions differ; do not merge)**
- Causal training: regular case 3.49e-4 (depth 5, width 256, 2e5 iterations x 10 windows, 164.77 it/s); chaotic case 2.46e-2 in text vs 2.26e-2 in a figure caption (depth 10, width 128, 2e5 iterations x 5 windows, 28.22 it/s), RTX A6000 — [arXiv 2203.07404](https://ar5iv.labs.arxiv.org/html/2203.07404)
- JAX-PI: 1.61e-1 for the long chaotic run; 1.42e-4 as best ablation value on a different configuration; ablation run time 13.33 min — [arXiv 2308.08468](https://arxiv.org/html/2308.08468)
- SSBroyden, double precision, 5 x 30 (4,411 params): 2.65e-5 on [0, 0.5] with 5 windows in 30,883 s; 6.51e-4 on [0, 1] with 20 windows in 130,222 s — [arXiv 2501.16371](https://arxiv.org/html/2501.16371)
- PINNacle KS task: every method near 100% error (vanilla 1.01E+0; best listed 9.57E-1, LRA); vanilla runtime 1.40E+3 s — [PINNacle](https://arxiv.org/pdf/2306.08827)

**Nonlinear Schrödinger (Raissi form, i h_t + 0.5 h_xx + |h|^2 h = 0, h0 = 2 sech x, periodic, t in [0, pi/2])**
- Raissi 2019: 1.97e-3, 5-layer network with 100 neurons per layer (40,902 parameters per a later paper) — [Raissi et al. 2019](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf)
- Shallow LM PINN: 2.1e-5 with two NN(3,25,25,1) networks (1,552 params) in 10,303 s on an Intel Core i5-10400 CPU; BFGS 2.7e-3 — preprint — [arXiv 2602.08515](https://arxiv.org/html/2602.08515v2)
- Competitive PINNs: PINN plateaus "about 4e-3" while CPINN reaches 6e-4 (search summary; their NLS setup not verified to be identical) — [arXiv 2204.11144](https://arxiv.org/html/2204.11144v2)
- A 2026 split-complex-valued PINN preprint lists PINN / SA-PINN / LA-PINN at 6.7e-4 / 4.80e-4 / 2.83e-4 and its own 8.45e-5 (search summary; the 6.7e-4 "PINN" figure looks like Raissi's Burgers number, so the attribution is suspect) — [arXiv 2607.15087](https://arxiv.org/html/2607.15087v1)
- Grossmann et al. solve 1-D and 2-D semilinear Schrödinger with PINN vs FEM: FEM solution time lower by 2 orders of magnitude and lower error on |h| in 1-D — [arXiv 2302.04107](https://arxiv.org/pdf/2302.04107)

**Cahn-Hilliard (no shared definition)**
- CI-PINN preprint (2026): mean relative L2 (4.26 +/- 0.18)e-1 for PINN vs (6.06 +/- 0.28)e-3 for CI-PINN over five runs; causal PINN (8.78 +/- 5.22)e-4 at its best tested epsilon — [arXiv 2609.36615](https://arxiv.org/pdf/2609.36615)
- 1-D causal study (Applied Sciences 2025): APM-PINN 0.003203 (16 x 128), TCAS-PINN 0.004568, standard PINN 1.025 (search summary) — [doi 10.3390/app15168863](https://doi.org/10.3390/app15168863)
- Mass-preserving adaptive PINN, 2-D Flory-Huggins: "approximately 10 hours" training, about 25% more than standard PINN, error better by more than one order (search summary) — [arXiv 2404.18054](https://arxiv.org/pdf/2404.18054)

**Lid-driven cavity (Reynolds number and metric differ)**
- Re = 3200, relative L2 of velocity: JAX-PI 1.58e-1 (curriculum Re 100/400/1000/3200 with 5e4, 5e4, 1e5, 5e5 iterations); PirateNet 4.21e-2 (6 residual blocks = 18 hidden layers; 1e4, 2e4, 5e4, 5e4, 5e5 iterations for Re 100/400/1000/1600/3200). No time or hardware in either — [arXiv 2308.08468](https://arxiv.org/html/2308.08468); [arXiv 2402.00326](https://arxiv.org/html/2402.00326)
- PINNacle NS2d-C (lid-driven, low Re): vanilla 4.70E-2 (std 1.12E-3), best 3.60E-2 (LAAF per the PDF column order; an automated HTML read attributed it to vPINN), vanilla runtime 7.52E+2 s; JAX-PI 2.39e-3 and PirateNet 9.52e-4 on the same task — [PINNacle](https://arxiv.org/pdf/2306.08827); [PirateNets Table 9](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)
- Related: lid-driven wedge Stokes flow with Adam 18,400 + SSBFGS trust-region 419,965 iterations, 8 x 64 (33,667 params), 120,000 points, errors u/v/p 3.25e-4 / 1.57e-4 / 1.10e-1, NVIDIA H100 80 GB — [arXiv 2501.16371](https://arxiv.org/html/2501.16371)
- jinns framework timing on NS2d-C (seconds, NVIDIA T600): jinns 265, DeepXDE-JAX 278, DeepXDE-PyTorch 441, PINA 1600, Modulus 275 — [arXiv 2412.14132](https://arxiv.org/html/2412.14132v1)

**2-D heat / Poisson**
- PINNacle (5x100, 20k epochs): Poisson2d-C vanilla 6.94E-1, best 1.23E-2 (NTK), vanilla 3.39E+2 s; Heat2d-VC vanilla 1.01E+0, best 2.12E-1 (LRA); Heat2d-MS vanilla 6.21E-2, best 1.73E-2 (LBFGS); Heat2d-LT ~1.0 for all methods; 5-D Poisson PNd best 4.58E-4 (LRA). Method names assigned by me from the 12-column order PINN, PINN-w, LBFGS, LRA, NTK, RAR, MultiAdam, gPINN, vPINN, LAAF, GAAF, FBPINN — [PINNacle](https://arxiv.org/pdf/2306.08827)
- Grossmann et al., Poisson 1-D/2-D/3-D with analytic solutions: PINN training 2-3 orders of magnitude slower than FEM in 1-D, FEM faster by 1-3 orders in 2-D and 3-D; PINN evaluation 2-3 orders faster than FEM interpolation in 2-D; architectures up to [60,60,60,60,60,1], Adam 15,000-20,000 epochs then L-BFGS — [arXiv 2302.04107](https://arxiv.org/pdf/2302.04107)
- DSGNAR preprint claims relative l2 "as low as 3e-16" in double precision and up to eight orders of improvement on a high-dimensional Poisson problem (abstract page only) — [arXiv 2607.02194](https://arxiv.org/abs/2607.02194)

**Convection / reaction (Krishnapriyan failure-mode problems, commonly used as a benchmark)**
- Regular PINN relative error: beta=1 7.84e-3; beta=10 1.08e-2; beta=20 7.50e-1; beta=30 8.97e-1; beta=40 9.61e-1. Curriculum: beta=20 9.84e-3; beta=30 2.02e-2; beta=40 5.33e-2. Network 4 x 50 tanh, L-BFGS, Nf=1000, at least ten seeds; no hardware or time — [arXiv 2109.01050](https://ar5iv.labs.arxiv.org/html/2109.01050)

### Inferences
- The numbers to beat, on the common definitions, at a small-network scale: Burgers 1.62e-8 in 2,812 s (3,021 params, A6000 GPU, float64) and Allen-Cahn 9.43e-7 in 2,000 s (2,019 params). At the first-order/GPU-pipeline scale: Allen-Cahn 2.24e-5, KdV 4.27e-4, LDC Re=3200 4.21e-2, with no published time. On CPU, the only timed points found are Burgers 2.0e-4 in 2,281 s and NLS 2.1e-5 in 10,303 s (i5-10400, MATLAB).
- Raissi's "60 seconds on a Titan X for 6.7e-4" remains the only widely quoted time-to-error datum for Burgers from a peer-reviewed source; it is a natural anchor for an error-vs-time plot.
- Because the best accuracy now comes from float64 quasi-Newton on tiny networks, an error-vs-time benchmark must state precision; the same method loses three orders of magnitude in single precision (Allen-Cahn 9.43e-7 vs 4.73e-4).
- Time-windowing changes the problem: KS and long-time results are per-window budgets multiplied by window count; a fair time axis must count all windows.

### Gaps
- 1-D heat/diffusion: I found no published "record" on a shared definition; it is absent from PINNacle (only 2-D heat variants) and from jaxpi. PDEBench has 1-D diffusion-sorption and diffusion-reaction but I did not retrieve its PINN errors.
- KdV forward and Cahn-Hilliard have no community-standard definition; Raissi's KdV is inverse only.
- Second-order results for KdV, NLS and LDC in arXiv 2501.16371 (if any) were in the part of the paper not read; its venue was not confirmed (a search result suggested a later journal version; unverified).
- DSGNAR (2607.02194) and vRBA (2509.14198) figures are from abstract/search summaries only; hardware, network and exact problem definition not verified. Treat as claims.
- Several "as tabulated" Allen-Cahn rows come from the PirateNets comparison table, not from the original papers.

## 3. What the critiques say and the protocol elements they demand

### Takeaway
The critiques converge on: PINNs lose to tuned classical solvers on forward problems by orders of magnitude in time, the literature hides this (only 12% of sampled PINN papers report efficiency vs a standard solver), and a credible comparison must be at equal accuracy or equal runtime against an efficient classical method, with cost-accuracy curves, justified hardware, multiple runs and reported negative results.

### Cited Findings

**McGreivy and Hakim 2024 — PR, DOI 10.1038/s42256-024-00897-5 (Nature Machine Intelligence prefix); arXiv 2407.07218**
- "79% (60/76) compare to a weak baseline"; 2/76 (2.6%) uncertain, 14/76 (18.4%) fair — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- "Rule 1 is to make comparisons at either equal accuracy or equal runtime. Rule 2 is to compare to an efficient numerical method." They are "necessary but not sufficient" — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- Reproduction of ten articles: "In 9/10 cases the stronger baseline is at least two orders of magnitude faster than the slower baseline"; "In 7/10 cases the stronger baseline outperforms the ML-based solver" — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- Reporting bias: of 232 abstracts, 94.8% (220/232) mention only positive results, 5.2% both, 0% only negative; of 82 articles meeting criteria, 76 (93%) claim to outperform a standard method, 4 (5%) to underperform — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- On PINNs specifically: PINN articles were excluded from the main review "because it is already well-known that standard numerical methods outperform PINNs for forward problems"; PINNs are "known to be orders of magnitude slower than standard numerical methods at forward problems". In a random sample of 60 fluid PINN articles (sample B), only 8% (5/60) report efficiency relative to standard solvers in the abstract and "Only 12% (7/60) ... in the entire article", vs 80% (61/76) in sample A. Eight PINN articles claiming to beat standard methods were excluded since each "compares to a weak baseline or, more often, doesn't account for the PINN optimization time" — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- "Using PINNs to get good accuracy on most PDEs takes (in most cases) hours to days, while doing so with a standard numerical method takes between a fraction of a second to minutes" — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- Recommendations: compare to both standard numerical methods and other ML solvers; reduce the classical solver's resolution/iterations until equal accuracy or equal runtime; "Ideally, articles would also make plots of cost versus accuracy"; a separate paragraph justifying baseline choice and why it is state of the art, ideally comparing several numerical methods; be cautious comparing a specialised ML solver to a general-purpose library (Rec. 1); justify why the hardware choice (CPU/GPU/TPU) is fair, "Usually, GPU-GPU or TPU-TPU comparisons will be most fair" (Rec. 2); report the number of surrogate evaluations N needed to amortise data and training cost, C_data + C_train + N t_B / s = N t_B (Rec. 3); failing to beat a baseline should not cause rejection, failing to follow best practice should — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)

**Grossmann, Komorowska, Latz, Schönlieb, "Can PINNs beat the FEM?" — arXiv 2302.04107 (journal version not confirmed in what I read)**
- PDEs: Poisson 1-D/2-D/3-D, Allen-Cahn 1-D, semilinear Schrödinger 1-D/2-D. FEM in FEniCS on CPU; PINNs (Adam then L-BFGS, Latin hypercube resampling each Adam epoch) on GPU. "All experiments were run on the same machine that has 12 CPU cores and we used a Quadro P6000 GPU"; codes run 10 times, times averaged; metric l2 relative error; solution time and evaluation time reported separately — [arXiv 2302.04107](https://arxiv.org/pdf/2302.04107)
- Results: 1-D Poisson PINN "training time is 2-3 orders of magnitude higher than in FEM"; 2-D Poisson FEM "faster by 1-3 orders of magnitude" but PINN evaluation "2-3 orders of magnitude faster then the interpolation of FEM on a new mesh"; 3-D Poisson PINNs "about 1-3 orders of magnitude slower in training time"; Allen-Cahn "FEM is 5-6 orders of magnitude faster than PINNs"; 1-D Schrödinger FEM lower solution time "by 2 orders of magnitude" — [arXiv 2302.04107](https://arxiv.org/pdf/2302.04107)
- Conclusion: "Considering the solution time and accuracy, PINNs are not able to beat the finite element method in our study"; "FEM solutions were generally faster at the same accuracy or at a higher accuracy"; PINN evaluation was faster only in the 3-D Poisson test; surprise that PINNs struggled with Allen-Cahn at small epsilon, attributed to the ill-conditioned strong form — [arXiv 2302.04107](https://arxiv.org/pdf/2302.04107)

**Krishnapriyan et al. 2021 — PR, NeurIPS 2021**
- Vanilla PINNs fail on convection (beta >= 20: 75-96% error), reaction and reaction-diffusion with larger coefficients even though closed-form solutions exist; cause is the loss landscape from the soft PDE constraint, not expressivity; curriculum regularisation and sequence-to-sequence training give "up to 1-2 orders of magnitude lower error". Protocol: each configuration run at least ten times with different seeds; L-BFGS learning rates swept; no time or hardware reported — [arXiv 2109.01050](https://ar5iv.labs.arxiv.org/html/2109.01050); [NeurIPS 2021](https://papers.nips.cc/paper_files/paper/2021/file/df438e5206f31600e6ae4af72f2725f1-Paper.pdf)

**Chuang and Barba 2022 / 2023**
- SciPy 2022 "pitfalls and frustration": cost-performance of PINNs (NVIDIA Modulus) not competitive with the CFD solver PetIBM; data-free PINNs gave a steady solution rather than vortex shedding — [SciPy 2022 proceedings](https://proceedings.scipy.org/articles/majora-212e5952-005)
- 2023 "Predictive limitations of PINNs in vortex shedding" (arXiv 2306.00230; marked "Preprint submitted to an Elsevier journal"): Taylor-Green verification, "the PINN solver took about 6 hours to converge with a V100 GPU, while PetIBM needed less than 10 seconds to get an error level of 10^-2 using a very dated K40 GPU". Cylinder Re=200: steady, unsteady and data-driven PINN runs "about 28 hours, 31 hours, and 33.5 hours using one A100 GPU" vs PetIBM "around 1.7 hours with a K40 GPU". Data-free PINNs failed to predict shedding; data-driven PINN reverted to steady flow outside the training window. The authors caveat that their PINN configuration "might not be optimal" and that computational performance was not the paper's focus — [arXiv 2306.00230](https://arxiv.org/pdf/2306.00230)

**PINNacle as a critique**
- Vanilla PINN solves only 10 of 22 tasks at a 10% threshold; all methods fail on Heat2d-LT, NS2d-LT, Wave2d, KS — [PINNacle](https://arxiv.org/pdf/2306.08827)

### Inferences
- Minimum credible protocol implied by the union of these sources: (1) a strong classical baseline per PDE, chosen and justified (spectral/Chebfun, FEM, tuned FD), run at several resolutions to produce its own error-vs-time curve; (2) comparison read off at equal error or equal time; (3) full PINN optimisation time counted (including all time windows, warm-up, JIT compile); (4) solution time separated from evaluation time (Grossmann); (5) >=5-10 seeds with spread (Krishnapriyan 10, Grossmann 10 timing repeats, PirateNets 5, PINNacle 3); (6) hardware named for both sides with a fairness argument; (7) precision stated; (8) failures reported.
- A PINN paper that reports efficiency relative to a classical solver at all is already in the top 12% of McGreivy and Hakim's sample B; one that does so at equal accuracy on CPU-CPU would be rare.
- The critiques concern forward problems; McGreivy and Hakim explicitly leave room for inverse/ill-posed problems ("PINNs have had some success"), which supports positioning a benchmark around inverse problems plus honest forward numbers.

### Gaps
- Journal venue/year of Grossmann et al. (commonly cited as IMA J. Appl. Math. 2024) and of Chuang and Barba 2023 were not confirmed from the texts I read.
- I did not retrieve absolute FEM/PINN seconds per case from Grossmann (they are in figures, not tables).
- The SciPy 2022 paper's "about 32 hours" figure appeared truncated in search; not verified.
- Other critiques (e.g. Markidis 2021 "The Old and the New", Frontiers in Big Data) surfaced only as titles.

## 4. How software packages report speed; published cross-framework timing

### Takeaway
Published cross-framework timings are scarce and none is neutral: the only table found is from the jinns authors (seconds on a laptop NVIDIA T600), plus PINNacle's per-method runtimes inside one DeepXDE-based codebase and throughput figures in the causal-training paper. I found no official step-time or throughput benchmark from DeepXDE, PhysicsNeMo/Modulus, NeuralPDE.jl, PINA, IDRLnet or TensorDiffEq, and no CPU cross-framework comparison in a paper.

### Cited Findings
- jinns (Gangloff and Jouvin; arXiv 2412.14132, stat.ML, preprint) Table 1, training time in seconds on a laptop NVIDIA T600 GPU, identical hyperparameters, vanilla MLP, no mini-batching: Burgers1D — jinns 445, DeepXDE-JAX 723, DeepXDE-PyTorch 671, PINA 1977, Modulus 646. NS2d-C — 265 / 278 / 441 / 1600 / 275. PInv — 149 / 218 / code crashed / 1509 / 135. Diffusion-Reaction inverse — 284 / not implementable / 3424 / 4061 / 2541. Navier-Stokes inverse — 175 / not implementable / 1511 / 1403 / 498. Problems taken from PINNacle and DeepXDE docs. Authors' caveats: short study, errors differ across libraries, Modulus samples points randomly. (Column mapping inferred from flattened text.) — [arXiv 2412.14132](https://arxiv.org/html/2412.14132v1)
- PINNacle Table 12: seconds per 20,000 epochs for each method on each of 22 tasks within one framework (e.g. Burgers1d-C: PINN 2.84E+2, LRA 7.64E+2, NTK 6.70E+2, MultiAdam 5.06E+2, gPINN 6.28E+2, hp-VPINN 2.85E+2, LAAF 3.61E+2, GAAF 3.56E+2, FBPINN 1.11E+3) — [PINNacle](https://arxiv.org/pdf/2306.08827)
- Causal-training paper reports iterations/second on an RTX A6000 (Allen-Cahn 120.30 MLP / 58.42 modified MLP; KS 164.77 / 28.22; NS 68.29) — [arXiv 2203.07404](https://ar5iv.labs.arxiv.org/html/2203.07404)
- Expert's Guide reports minutes per ablation run and a weak-scaling experiment (batch 40,960 per GPU, 10,000 iterations, up to 256 GPUs) but no GPU model in the text read — [arXiv 2308.08468](https://arxiv.org/html/2308.08468)
- jaxpi README: no timing numbers; GPU-only — [jaxpi README](https://github.com/PredictiveIntelligenceLab/jaxpi)
- Kiyani et al. give per-run seconds and GPU for every optimizer configuration (e.g. Burgers 2,812 s A6000; JAX BFGS 1,587 s RTX 3090), which permits a crude seconds-per-iteration figure (~0.09 s/iteration for SSBroyden at 3,021 params, by my division of 2,812 s by 31,000 iterations) — [arXiv 2501.16371](https://arxiv.org/html/2501.16371)
- A non-peer-reviewed GitHub pull request (PINNeAPPle) claims PINNeAPPle and DeepXDE are "within 4-11 % per iteration", PhysicsNeMo-Sym costs 1.4-1.9x more, and Sym's official 6x512 network costs 3.4 s per step on CPU (search summary; low reliability) — [PINNeAPPle PR 899](https://github.com/PINNeAPPle-Labs/PINNeAPPle/pull/899)
- A study "Comparing PINNs Across Frameworks: JAX, TensorFlow, and PyTorch" reports a largest speed-up of 51.94 for JAX on a KdV example, mostly against TensorFlow v1 (search summary only) — [ResearchGate 380824296](https://www.researchgate.net/publication/380824296_Comparing_PINNs_Across_Frameworks_JAX_TensorFlow_and_PyTorch)
- Chuang and Barba used NVIDIA Modulus: 6 h on a V100 for 2-D Taylor-Green; 28-33.5 h on an A100 for cylinder flow — [arXiv 2306.00230](https://arxiv.org/pdf/2306.00230)
- Grossmann et al. cite DeepXDE, NVIDIA Modulus (SimNet) and NeuroDiffEq as packages but time their own implementation on a Quadro P6000 — [arXiv 2302.04107](https://arxiv.org/pdf/2302.04107)

### Inferences
- Step time in the literature spans roughly 8 ms (120 it/s, 6x128 MLP on an A6000) to tens of ms per Adam step on a high-end GPU for 128-256-wide networks with batch ~4k-8k; PINNacle's vanilla Burgers run is 284 s / 20,000 epochs = ~14 ms per epoch (5x100, ~10k points, unknown GPU). These are the only public anchors for a "ms per step" claim, and all are GPU.
- A CPU step-time claim (e.g. single-digit ms per step for a small network) has no published competitor table to be compared against; the benchmark would need to run DeepXDE/jinns/PINA itself on the same CPU, using the jinns scripts as a template for hyperparameter parity.
- Framework comparisons at fixed epochs conflate step time with convergence; the jinns authors themselves note errors differ across libraries. Reporting both ms/step and time-to-target-error avoids this.

### Gaps
- No official speed documentation retrieved for DeepXDE, PhysicsNeMo/Modulus, NeuralPDE.jl, PINA, IDRLnet or TensorDiffEq; "none found" here means my searches did not surface any, not that none exists.
- jinns venue not confirmed (the preprint header had placeholder journal text); iterations per benchmark only in its repository.
- No published CPU-vs-GPU crossover study for small PINNs found.

## 5. What is missing

### Takeaway
No accepted benchmark provides (i) error-versus-wall-clock curves, (ii) CPU timings, (iii) a small-network track, or (iv) inverse problems scored on identifiability and uncertainty calibration; the existing inverse tasks are two noise-free coefficient-recovery cases scored by L2RE.

### Cited Findings
- PINNacle's time information is one runtime per method at a fixed 20,000 epochs; accuracy leaders (jaxpi, PirateNets) publish final error without time or hardware — [PINNacle](https://arxiv.org/pdf/2306.08827); [PirateNets](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf)
- McGreivy and Hakim ask for cost-versus-accuracy plots and equal-accuracy/equal-runtime comparisons, and find only 12% of sampled PINN articles report efficiency vs standard solvers at all — [arXiv 2407.07218](https://arxiv.org/pdf/2407.07218)
- Every timed result found is on a GPU except the shallow-LM preprint (Intel i5-10400, MATLAB): Burgers 2.0e-4 in 2,281 s, NLS 2.1e-5 in 10,303 s — [arXiv 2602.08515](https://arxiv.org/html/2602.08515v2)
- Small networks are where the best accuracy now is (2,019-4,411 parameters with SSBroyden in float64; 751-1,552 parameters with LM), while the suite defaults are 5x100 (PINNacle), 4x256 (jaxpi) and 9-18 layers x 256 (PirateNets) — [arXiv 2501.16371](https://arxiv.org/html/2501.16371); [arXiv 2602.08515](https://arxiv.org/html/2602.08515v2); [PINNacle (ar5iv)](https://ar5iv.labs.arxiv.org/html/2306.08827)
- Inverse problems in suites: PINNacle has PInv and HInv only, scored by L2RE (vanilla 9.42E-2 and 1.57E+0; best 1.96E-2 and 1.19E-2 by hp-VPINN per the main table); PirateNet 5.34e-3 on PInv; jinns times three inverse problems without accuracy scoring; PIELM "is not applicable to inverse problems" — [PINNacle](https://arxiv.org/pdf/2306.08827); [PirateNets](https://jmlr.org/papers/volume25/24-0313/24-0313.pdf); [arXiv 2412.14132](https://arxiv.org/html/2412.14132v1)
- Inverse results in method papers are parameter percentage errors at 0-10% noise (Raissi: Burgers lambda errors, KdV 0.023% / 0.006% noise-free) or absolute percentage errors (LM preprint: Allen-Cahn three parameters 0.0027 / 0.0034 / 0.0011 with LM vs 44.2306 for one parameter with BFGS), with no interval coverage or identifiability analysis — [Raissi et al. 2019](https://faculty.sites.iastate.edu/hliu/files/inline-files/PINN_RPK_2019_1.pdf); [arXiv 2602.08515](https://arxiv.org/html/2602.08515v2)
- Precision is rarely a declared benchmark axis although it changes results by orders of magnitude (Allen-Cahn 9.43e-7 double vs 4.73e-4 single) — [arXiv 2501.16371](https://arxiv.org/html/2501.16371)
- Metric fragmentation across data-driven suites (nRMSE in PDEBench vs VRMSE in The Well) prevents cross-suite comparison (second-hand) — [arXiv 2509.21670](https://arxiv.org/html/2509.21670v1)

### Inferences
- A benchmark that would be both credible and new: per equation, plot relative L2 error vs wall-clock seconds (log-log) for (a) a justified classical solver at several resolutions, (b) the proposed PINN, (c) at least one external PINN framework, all on one named CPU (and optionally one named GPU), float64 and float32 declared, >=5 seeds with bands, with the published anchor points from Section 2 overlaid and labelled with their own hardware.
- Equations with an agreed definition and published anchors: Burgers nu=0.01/pi, Raissi/Wu Allen-Cahn, Raissi NLS, PINNacle Burgers1d-C / NS2d-C / PInv / HInv (data downloadable), Krishnapriyan convection beta=30/40. Equations where the team must define the problem and cannot claim a "record": 1-D heat, 2-D heat/Poisson, Cahn-Hilliard, KdV forward (use the jaxpi definition to compare with 4.27e-4).
- An inverse-problem track with noise levels, parameter error, interval coverage and a declared identifiability check has no incumbent; the bar from the suites is only PInv 5.34e-3 and HInv 1.19e-2 L2RE.
- Honest framing matters: on forward problems the classical solver will win the error-vs-time plot by orders of magnitude (Grossmann; Chuang and Barba), so the contribution should be stated as "fastest PINN at a given error on CPU, with the classical gap shown", not as beating solvers.

### Gaps
- I did not search specifically for inverse-problem PINN benchmarks with UQ/calibration (e.g. Bayesian PINN benchmark papers), so "no incumbent" is supported only by absence in the suites reviewed here.
- 2025-2026 coverage is partial: only arXiv 2501.16371, 2509.14198, 2510.21262 (PINN Balls, ~1e-4 on Burgers with 20 submodels, search summary), 2602.08515, 2607.02194, 2607.15087 and 2609.36615 surfaced; none verified as peer-reviewed.
- No source found defining a standard "time-to-error" metric for PINNs (analogous to DAWNBench); this absence is an observation from the searches, not a cited fact.
