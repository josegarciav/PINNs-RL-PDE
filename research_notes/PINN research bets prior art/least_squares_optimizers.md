# Second-order and least-squares optimizers for PINN losses

Scope note: searched and read on 2026-11-19. Numbers are quoted as extracted from the papers' HTML (arXiv / ar5iv) by a fetch tool; several pages were truncated, so tables that live late in a paper or only in figures are listed under Gaps instead of being reconstructed. Notation: P (or p, n, d) = trainable parameters; N (or m, q) = residual / collocation rows; J = Jacobian of the residual vector with respect to parameters (N x P). "Peer-reviewed" / "preprint" is marked per paper.

## What are the key papers on Gauss-Newton, Levenberg-Marquardt, energy natural gradient, KFAC, quasi-Newton variants (SSBroyden, self-scaled BFGS), SOAP, NysNewton and other second-order methods for PINNs?

### Takeaway
The line of work runs from ENGD (ICML 2023, dense P x P Gramian) through KFAC and matrix-free Gauss-Newton (2024) to sample-space ("Woodbury" / dual / kernel) solves, randomized preconditioning and sketching (2025-2026); in parallel, a quasi-Newton line (SSBFGS / SSBroyden, Urbán, Kiyani, Karniadakis) and a preconditioned first-order line (SOAP) compete with it. By 2026 natural gradient, Gauss-Newton and Levenberg-Marquardt are treated as the same damped least-squares step, and the open contest is how to solve it cheaply.

### Cited Findings

Natural gradient / Gauss-Newton family
- Müller and Zeinhofer, "Achieving High Accuracy with PINNs via Energy Natural Gradient Descent", ICML 2023, PMLR 202:25471-25485 (peer-reviewed). An energy natural gradient step in parameter space corresponds to a Newton step in function space, up to an orthogonal projection onto the model's tangent space. — [PMLR](https://proceedings.mlr.press/v202/muller23b.html); [ar5iv full text](https://ar5iv.labs.arxiv.org/html/2302.13163)
- Jnini, Vella and Zeinhofer, "Gauss-Newton Natural Gradient Descent for Physics-Informed Computational Fluid Dynamics", arXiv:2402.10680, Feb 2024 (preprint; no venue confirmed). Gauss-Newton in function space, discretized in the tangent space; matrix-free via CG with Gramian-vector products "G v = J^T w, where w = J v". — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Dangel, Müller and Zeinhofer, "Kronecker-Factored Approximate Curvature for Physics-Informed Neural Networks", NeurIPS 2024 (peer-reviewed). Uses Taylor-mode AD to rewrite the differential operator as a forward network with shared weights, then applies KFAC for weight-sharing networks. — [NeurIPS proceedings](https://papers.nips.cc/paper_files/paper/2024/hash/3d27d607586984908900eaa8ce19c96c-Abstract-Conference.html); [arXiv HTML](https://arxiv.org/html/2405.15603v1)
- Schwencke and Furtlehner, "ANaGRAM: A Natural Gradient Relative to Adapted Model for efficient PINNs learning", ICLR 2025 (peer-reviewed). Natural gradient with cost min(P^2 S, S^2 P) for P parameters and S sample points; links the natural gradient to the Green's function of the operator on the tangent space. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/hash/38b445e98823a02479582302ee3aa8e4-Abstract-Conference.html); [arXiv](https://arxiv.org/abs/2412.10782v2)
- Guzmán-Cordero, Dangel, Goldshlager and Zeinhofer, "Improving Energy Natural Gradient Descent through Woodbury, Momentum, and Randomization", arXiv:2505.12149 (v2 Oct 2025; preprint, no venue on the arXiv page). Woodbury (sample-space) ENGD, SPRING momentum from variational Monte Carlo, Nyström randomization. — [arXiv abs](https://arxiv.org/abs/2505.12149); [arXiv HTML](https://arxiv.org/html/2505.12149v2)
- Jnini and Vella, "Dual Natural Gradient Descent for Scalable Training of Physics-Informed Neural Networks", arXiv:2505.21404 (v2 Oct 2025; on OpenReview, venue not named in the text read). LM-damped Gauss-Newton solved in residual space, dense Cholesky or Nyström-preconditioned CG, plus geodesic acceleration. — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Bioli, Marcati and Sangalli, "Accelerating Natural Gradient Descent for PINNs with Randomized Numerical Linear Algebra", arXiv:2505.11638; journal reference Computer Methods in Applied Mechanics and Engineering 464 (Part A), 119456, DOI 10.1016/j.cma.2026.119456 (peer-reviewed). Matrix-free NGD with CG preconditioned by randomized Nyström or RPCholesky. — [arXiv abs](https://arxiv.org/abs/2505.11638); [arXiv HTML](https://arxiv.org/html/2505.11638v6)
- Webb, Jerad and Cartis, "An Optimisation Framework for the Well-Conditioned Training of Physics-Informed Neural Networks" (DSGNAR, Doubly-Sketched Gauss-Newton with Adaptive Ratio), arXiv:2607.02194, 2 July 2026 (preprint). — [arXiv abs](https://arxiv.org/abs/2607.02194); [arXiv HTML](https://arxiv.org/html/2607.02194)
- Best McKay, Lawrence, Wetton and Gopaluni, "Error whitening: Why Gauss-Newton outperforms Newton", arXiv:2605.11316, 11 May 2026 (preprint, labelled a NeurIPS preprint). Generalized Gauss-Newton projects the Newton direction onto the model tangent space, replacing J J^T with the identity in function-space dynamics; reports Gauss-Newton outperforming Newton, Adam and Muon in supervised learning, physics-informed learning and approximate dynamic programming. — [arXiv abs](https://arxiv.org/abs/2605.11316)

Levenberg-Marquardt
- Bonfanti, Bruno and Cipriani, "The Challenges of the Nonlinear Regime for Physics-Informed Neural Networks", NeurIPS 2024 poster 95966 (peer-reviewed; arXiv:2402.03864). For nonlinear PDEs the NTK is random at initialization and not constant in training, and the residual Hessian's spectral norm stays bounded below as width grows; uses LM, theta <- theta - [J^T J + lambda I]^-1 grad L. — [arXiv HTML](https://arxiv.org/html/2402.03864); [NeurIPS poster page](https://nips.cc/virtual/2024/poster/95966)
- Bonfanti, Medina, List, Staeves, Santana and Ellero, "PINN Balls: Scaling Second-Order Methods for PINNs with Domain Decomposition and Adaptive Sampling", arXiv:2510.21262, Oct 2025 (listed "Accepted Conference Paper", conference not named). Local mixture of experts with learnable domain decomposition so that LM stays affordable. — [arXiv abs](https://arxiv.org/abs/2510.21262)
- Shahab, Mukhlash and Susanto, "Do physics-informed neural networks (PINNs) need to be deep? Shallow PINNs using the Levenberg-Marquardt algorithm", arXiv:2602.08515 (v3 Aug 2026; preprint). Analytic layer-by-layer derivative formulas for the residual Jacobian, two-hidden-layer networks, MATLAB on CPU. — [arXiv HTML](https://arxiv.org/html/2602.08515)

Hessian-based Newton
- Rathore, Lei, Frangella, Lu and Udell, "Challenges in Training PINNs: A Loss Landscape Perspective", ICML 2024 oral, PMLR 235 (peer-reviewed). Introduces NysNewton-CG (NNCG): damped Newton step (H + mu I)^-1 grad L solved by Nyström-preconditioned CG on Hessian-vector products. — [PMLR](https://proceedings.mlr.press/v235/rathore24a.html); [arXiv HTML](https://arxiv.org/html/2402.01868v1)

Quasi-Newton
- Urbán, Stefanou and Pons, "Unveiling the optimization process of Physics Informed Neural Networks: How accurate and competitive can PINNs be?", arXiv:2405.04230; the arXiv HTML states Journal of Computational Physics (peer-reviewed). Self-scaled BFGS and self-scaled Broyden plus a transformed loss g(J) with g = sqrt or log. — [arXiv HTML](https://arxiv.org/html/2405.04230v2)
- Kiyani, Shukla, Urbán, Darbon and Karniadakis, "Optimizing the Optimizer for Physics-Informed Neural Networks and Kolmogorov-Arnold Networks" (earlier versions titled "Which Optimizer Works Best for ..."), arXiv:2501.16371 (v6, 8 Feb 2026; venue not stated on the page read). — [arXiv HTML](https://arxiv.org/html/2501.16371v6)
- Jnini, Kiyani, Shukla, Urbán, Ahmadi Daryakenari, Müller, Zeinhofer and Karniadakis, "Curvature-Aware Optimization for High-Accuracy Physics-Informed Neural Networks", arXiv:2604.05230, 6 April 2026 (preprint). The most complete head-to-head: Adam, AdamW, SOAP, BFGS, SSBFGS, SSBroyden, NG (Gauss-Newton with LM damping) and SPRING, with a roofline analysis. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)

Preconditioned first-order
- Wang, Bhartari, Li and Perdikaris, "Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective", NeurIPS 2025 (peer-reviewed; arXiv:2502.00604). Argues SOAP approximates a Newton preconditioner and resolves gradient conflicts between loss terms; first PINN results on turbulent flows at Reynolds numbers up to 10,000. — [NeurIPS poster](https://neurips.cc/virtual/2025/poster/116510); [arXiv HTML](https://arxiv.org/html/2502.00604v2)

### Inferences
- Three camps are visible: (a) Gauss-Newton / natural gradient (Zeinhofer, Müller, Dangel, Jnini, Bioli, Cartis groups), (b) self-scaled quasi-Newton (Urbán, Kiyani, Karniadakis), (c) Kronecker-preconditioned Adam (Perdikaris). The April 2026 joint paper (2604.05230) has authors from camps (a) and (b), which makes its recommendation ("natural-gradient methods as a first choice for elliptic and parabolic PDEs") the closest thing to a consensus statement.
- "A matrix-free Gauss-Newton / LM optimizer for PINNs" is not new as a method: Jnini et al. 2024, the Hessian-free baseline in Dangel et al. 2024, Jnini and Vella 2025 (PCG variant) and Bioli et al. (CMAME) all do it.

### Gaps
- The peer-review status of 2501.16371 (Kiyani et al.), 2402.10680 (Jnini et al.) and 2505.21404 (D-NGD) was not confirmed from the pages read; they are treated as preprints here.
- TINNs (arXiv:2601.20361), "Alternating Levenberg-Marquardt Training of PINNs with Fourier-Enhanced Features" (arXiv:2608.05892), "Higher-Order Geometric Updates for Levenberg-Marquardt Method via Riemann Normal Coordinates" (arXiv:2607.07623) and AMStraMGRAM (arXiv:2510.15998) surfaced in search results only; they were not read.
- Older Gauss-Newton / LM work on PINNs before 2023 and the Shampoo / SOAP originals were not searched.

## What are the strongest reported results (relative L2 error by equation, network size, wall-clock time) against Adam and Adam followed by L-BFGS?

### Takeaway
On small networks (hundreds to tens of thousands of parameters) in float64 on a GPU, Gauss-Newton-type and self-scaled quasi-Newton methods report relative L2 errors of 1e-6 to 1e-9 where Adam stays at 1e-2 to 1e-3 and L-BFGS at 1e-3 to 1e-4, often in less wall-clock time; the July 2026 DSGNAR preprint claims errors down to 3e-16. Results on shocks, chaotic problems and large Reynolds numbers are much weaker.

### Cited Findings

ENGD (ICML 2023; RTX 3080 Laptop GPU, float64, JAX; 10 initializations; shallow tanh width 64, 257 weights; 900 interior + 120 boundary points)
- Poisson 2D, median relative L2: GD 8.2e-3, Adam 1.1e-3, BFGS 4.4e-4, E-NGD 2.4e-7 (min 1.0e-7, max 4.1e-7). E-NGD ran 500 iterations; GD and Adam ran 2e5. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163)
- Heat 1D (u_t = 1/4 u_xx), median relative L2: GD 1.6e-2, Adam 1.0e-3, BFGS 1.4e-4, E-NGD 6.3e-6 (min 2.3e-6, max 5.6e-1). E-NGD ran 2e3 iterations. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163)
- Time per iteration / full run, Poisson 2D: GD 1.8e-2 s / 1 h; Adam 3.7e-2 s / 1 h 6 min; E-NGD 8.6e-2 s / 43 s; BFGS 1.8 s / 15 min. Heat 1D: Adam 3.8e-2 s / 2 h 6 min; E-NGD 8.3e-2 s / 2 min 48 s; BFGS 4.3 s / 35 min 48 s. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163)
- Deep network (three hidden layers of width 50, about 5k weights), Poisson 2D: E-NGD median L2 2.5e-6 after 100 iterations at 1.1 min per iteration, 1.8 h in total. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163)
- 5D Poisson: no table; E-NGD is "about two orders of magnitude" more accurate than GD and Adam, and BFGS was no more accurate than Adam. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163)

Gauss-Newton natural gradient on Navier-Stokes (preprint; A100 80 GB, float64; medians)
- Kovasznay (2D steady, Re = 40; 4 x 50 network; 2601 interior + 400 boundary points): GNNG 4.84e-06, BFGS 1.51e-04, Adam 2.05e-03. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Beltrami (3D unsteady, Re = 1; 10,000 interior points): GNNG 4.90e-06, BFGS 3.19e-04, Adam 4.99e-02. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Taylor-Green (2D unsteady, Re = 500; 4 x 50; 8,000 interior points; hard constraints): GNNG 5.68e-07, BFGS 6.24e-05, Adam 6.46e-04. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Matrix-free Taylor-Green, soft constraints, 10 x 100 network (about 91k parameters), CG tolerance 1e-5: GNNG 3.57e-05, L-BFGS 4.96e-02, Adam 2.29e-01. Iteration budgets: GNNG 5,000 (1,000 matrix-free), BFGS 5,000, Adam 200,000 (100,000 matrix-free). No wall-clock times reported. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)

NysNewton-CG (ICML 2024; three hidden layers, widths 50-400; 10,000 residual points; 41,000 total iterations; best across widths)
- Relative L2 (L2RE), Adam / L-BFGS / Adam+L-BFGS: convection (beta = 40) 5.96e-2 / 8.26e-3 / 4.19e-3; reaction (rho = 5) 2.12e-2 / 3.83e-2 / 1.92e-2; wave (beta = 5) 3.49e-1 / 3.35e-1 / 5.52e-2. — [arXiv HTML](https://arxiv.org/html/2402.01868v1)
- NNCG applied after Adam+L-BFGS: convection 4.19e-3 -> 1.94e-3; reaction 1.92e-2 -> 9.92e-3; wave 5.52e-2 -> 1.27e-2. Gradient descent applied at the same point leaves L2RE unchanged. — [arXiv HTML](https://arxiv.org/html/2402.01868v1)

Levenberg-Marquardt
- Bonfanti et al. (NeurIPS 2024; 5 x 20 tanh; A10 GPU; Adam 1e5 iterations, LM 1e3): on Burgers, Adam and L-BFGS reach roughly 1e-3 relative L2 and LM about 1e-5 "in few iterations and very short GPU time"; convection solved with beta up to 100. No exact wall-clock figures in the text read. — [arXiv HTML](https://arxiv.org/html/2402.03864)
- Shahab et al. (preprint; MATLAB R2024a on a six-core Intel Core i5-10400 CPU): Burgers, NN(2,25,25,1), 751 parameters, 10,000 collocation points: LM relative L2 2.0e-4 against BFGS 2.2e-2; LM time 1962 (table units unlabeled, captions say seconds). Schrödinger, 1552 parameters: LM 2.1e-5 against BFGS 2.7e-3; LM time 8856. Inverse Allen-Cahn, 754 parameters: final loss LM 1.2e-8 against BFGS 1.7e-5, time 1677 against 834. Inverse 3D Bratu, 778 parameters: final loss 9.4e-7 against 1.2e-3, time 5074 against 14465. Iteration caps differ: 4000 for LM, 20,000 for the others. — [arXiv HTML](https://arxiv.org/html/2602.08515)

Dual NGD (preprint; A100 80 GB, JAX, Taylor-mode AD, tanh MLPs; median over ten seeds; 3000 s budget, 9000 s for the cavity). Format (parameters n, residuals m): SGD / Adam / L-BFGS / best D-NGD variant
- Heat 10+1d (118k, 6k): 3.48e-3 / 1.45e-3 / 9.82e-5 / 8.52e-6 (dense + geodesic acceleration). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Kovásznay 2d (8k, 0.8k): 6.94e-3 / 4.50e-3 / 9.48e-5 / 5.23e-7 (dense). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Poisson 10d (42k, 10k): 6.26e-2 / 3.51e-2 / 5.47e-4 / 2.74e-4 (PCG). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Fokker-Planck 9+1d (118k, 3k): 5.48e-2 / 4.80e-2 / 9.30e-1 / 2.47e-3. — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Allen-Cahn 1+1d (31k, 5.6k): 9.93e-1 / 4.92e-1 / 9.92e-1 / 9.13e-6. — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Lid-driven cavity Re = 3000 (66k, 50k): 1.10 / 6.93e-1 / 3.85e-1 / 3.59e-3 (PCG, with a Reynolds curriculum 100, 400, 1000). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Poisson 1e5-d (13M, 100): 3.67e-3 / 1.87e-4 / 1.78e-3 / 1.14e-5, with the Laplacian estimated stochastically (STDE). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)

Randomized-preconditioned matrix-free NGD (CMAME; RTX 4090, float64; median of 10 runs; p = 8641 for 3D Poisson and 3+1D heat). Format: median relative H1 error / wall-clock seconds (H1, not L2)
- Poisson 3D: NGD full 1.56e-6 / 317.7; NyströmNGD 1.66e-6 / 66.8; RPCholNGD 1.69e-6 / 74.1; SSBroyden 2.69e-6 / 482.6; BFGS 8.91e-5 / 540.4; L-BFGS 7.91e-4 / 1104.9; Adam 1.95e-2 / 412.7. — [arXiv HTML](https://arxiv.org/html/2505.11638v6)
- Heat 3+1D: NGD full 1.75e-6 / 279.2; NyströmNGD 1.89e-6 / 106.1; SSBroyden 2.39e-6 / 477.4; BFGS 1.03e-4 / 684.0; L-BFGS 3.91e-4 / 867.5; Adam 1.87e-3 / 1077.9. — [arXiv HTML](https://arxiv.org/html/2505.11638v6)
- Kovasznay (Re = 40): NGD full 1.08e-6 / 5708.3; NyströmNGD 8.24e-7 / 878.5; RPCholNGD 1.12e-6 / 654.4; SSBroyden 5.45e-7 / 329.3; BFGS 2.92e-5 / 418.6; L-BFGS 1.04e-4 / 774.2; Adam 2.47e-3 / 609.8. SSBroyden is both faster and more accurate here. — [arXiv HTML](https://arxiv.org/html/2505.11638v6)

Woodbury ENGD / SPRING (preprint; RTX 6000 24 GiB, float64; fixed time budgets)
- 5d Poisson (10,065 parameters, batch 3500, 7000 s budget): reaches ENGD's accuracy up to 75x faster; ENGD-W takes over 30x more steps in the budget. 10d Poisson (118,145 parameters): standard ENGD runs out of memory. 9+1d log-Fokker-Planck: SPRING's L2 error about an order of magnitude below KFAC's. — [arXiv HTML](https://arxiv.org/html/2505.12149v2)

Self-scaled quasi-Newton
- Kiyani et al., Burgers, 4 x 20 PINN, 1,341 parameters: Adam[1000] + SSBroyden/Wolfe[50000] 7.57e-8 in 1354 s; + BFGS/Wolfe 1.50e-5 in 1292 s; + L-BFGS/Wolfe 2.05e-3 in 713 s. With 8 x 20 (3,021 parameters): SSBroyden 1.62e-8 in 2812 s; BFGS 8.52e-6 in 3049 s. — [arXiv HTML](https://arxiv.org/html/2501.16371v6)
- Kiyani et al., Allen-Cahn, 3 x 30, 2,019 parameters: Adam[5000] + BFGS/Wolfe[20000] 7.59e-4 in 733 s; + SSBroyden/Wolfe[20000] 1.15e-6 in 973 s; with 30000 iterations 9.84e-4 in 1493 s against 9.43e-7 in 2000 s. — [arXiv HTML](https://arxiv.org/html/2501.16371v6)
- Kiyani et al., Kuramoto-Sivashinsky, 5 x 30, 4,411 parameters, 5 time windows: BFGS/Wolfe[30000] 2.39e-3 in 26697 s; SSBroyden/Wolfe[30000] 2.65e-5 in 30883 s. With 20 windows: 3.65e-2 in 119286 s against 6.51e-4 in 130222 s. Hardware: RTX A6000 and RTX 3090 GPUs. — [arXiv HTML](https://arxiv.org/html/2501.16371v6)
- Urbán et al. (JCP): Allen-Cahn with SSBroyden 2.2e-6 relative L2; on the current-free Grad-Shafranov problem BFGS reaches a loss about six orders of magnitude below Adam, with SSBFGS and SSBroyden roughly two and three further orders lower. — [arXiv HTML](https://arxiv.org/html/2405.04230v2)

Head-to-head, April 2026 preprint (float64; H100 GPU for Optax, NG and SPRING runs; SciPy-based PyTorch runs on CPU)
- 2D Helmholtz (a1, a2) = (1, 4), k = 1, 2,971 parameters, 15,000 collocation points, relative L2 / seconds: NG 4.2e-9 / 251; SPRING 2.0e-9 / 540; SSBroyden (optx trust region) 3.6e-9 / 319; SSBFGS (optx trust region) 7.2e-9 / 564; SSBroyden (PyTorch/SciPy) 7.7e-9 / 8215; BFGS (PyTorch/SciPy) 3.7e-7 / 7066; SOAP 4.2e-4 / 1038. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- 2D Helmholtz (10, 10), 6,691 parameters: SSBFGS TR 3.1e-3 / 1530; SSBroyden TR 6.7e-3 / 2194; NG 1.9e-5 / 3000; SPRING 3.8e-5 / 3000. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- 2D Stokes, 8 x 64 tanh, 33,667 parameters, relative L2 (u, v, p): Adam (500,001 iterations) (2.70e-2, 2.58e-2, 1.10e-1) in 3600 s; SOAP (2.02e-2, 1.87e-2, 1.90e-1) in 3600 s; SOAP + SSBFGS TR (3.25e-4, 1.59e-4, 1.07e-1) in 3600 s; NG with 5,000 points per iteration (3.25e-4, 1.57e-4, 1.07e-1) in 150 s. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Unsteady 2D viscous Burgers (nu = 0.004), 3,881 parameters: SOAP (21001 iterations) 1.7e-2 in 414 s; Adam + SSBroyden Wolfe 5.3e-7 in 612 s; Adam + SSBFGS Wolfe 5.7e-7 in 624 s; NG (20000 iterations) 5.8e-7 in 75 s. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Inviscid Burgers with Roe linearization, 10,000 points: SSBroyden relative L2 7.8e-2 in 31 s; NG 7.1e-2 in 193 s. 1D Euler (4,833 parameters), relative L1 (rho, u, p): SSBroyden (9.2, 8.0, 5.0)e-4 in 160 s; NG (6.5, 9.1, 3.3)e-3 in 452 s. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Stiff PK-PD ODE system, relative L2 of W / seconds: Adam 1.31e-1 / 323; SOAP 5.79e-3 / 461; BFGS 3.26e-2 / 118; SSBFGS 1.92e-2 / 83; SSBroyden 2.79e-3 / 98; NG 2.55e-3 / 302. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)

SOAP (NeurIPS 2025; PirateNet architecture), relative L2, Adam / SOAP
- Wave 5.15e-5 / 8.05e-6; Burgers 8.20e-5 / 4.03e-5; Allen-Cahn 2.24e-5 / 3.48e-6; Korteweg-De Vries 7.04e-4 / 3.40e-4; Kuramoto-Sivashinsky 7.48e-2 / 3.51e-2; Grey-Scott 3.61e-3 / 1.88e-4; Ginzburg-Landau 1.49e-2 / 4.58e-3; lid-driven cavity (Re = 5e3) 3.24e-1 / 3.99e-2; Kolmogorov flow (Re = 1e4) 2.04e-1 / 3.20e-2; Rayleigh-Taylor (Ra = 1e6) 7.32e-2 / 5.22e-3. "SOAP requires approximately 2x longer training time compared to baselines." — [arXiv HTML](https://arxiv.org/html/2502.00604v2)

DSGNAR (preprint, July 2026; single H100 GPU; comparison times copied from the cited papers, so hardware differs). Format: parameters, sketch size s, relative L2, seconds, against the cited benchmark
- Burgers, single precision: 1,447, s = 700, 4.75e-7 in 9.8 s against 4.04e-5 in 179 s. Burgers, double: 11,285, s = 4,000, 7.97e-14 in 346.1 s against 1.62e-8. — [arXiv HTML](https://arxiv.org/html/2607.02194)
- Kuramoto-Sivashinsky, double: 15,265, s = 5,000, 5.12e-7 in 5,226.6 s against 6.51e-4 in 130,222 s. — [arXiv HTML](https://arxiv.org/html/2607.02194)
- 5D Poisson, double: 13,313, s = 4,000, 3.03e-16 in 406.7 s against about 1e-7 in about 7,000 s. 10D Poisson: 17,241, s = 5,000, 5.96e-12 in 3,239.7 s against about 1e-5 in about 8,000 s. — [arXiv HTML](https://arxiv.org/html/2607.02194)
- Navier-Stokes: 9,219, s = 4,000, 1.13e-4 in 402.9 s against 1.43e-2 in about 90 s. — [arXiv HTML](https://arxiv.org/html/2607.02194)

KFAC (NeurIPS 2024)
- Architectures: 2 -> 64 -> 1 (D = 257), 2 -> 64 -> 64 -> 48 -> 48 -> 1 (D = 9,873), 2 -> 256 -> 256 -> 128 -> 128 -> 1 (D = 116,097); Poisson up to 100d with "parameters up to a million"; RTX 6000 GPUs, float64; 1,000 s budget per run on 2d Poisson, same budget for every optimizer on a problem. ENGD has no entry at D = 116,097 because it ran out of memory. Final errors are in figures only. — [arXiv HTML](https://arxiv.org/html/2405.15603v1)

### Inferences
- Equal-wall-clock comparisons do exist on GPUs (KFAC paper, Woodbury ENGD, D-NGD use fixed time budgets; 2604.05230 and Bioli et al. report seconds). The claim "nobody compares at equal time" would be false; "nobody compares at equal time on a CPU with a compiled engine" appears to hold in what was read.
- Typical speed advantage of Gauss-Newton over Adam followed by a self-scaled quasi-Newton method, where both reach the same error on an H100: about 8x on viscous Burgers (75 s against 612 s) and 24x on Stokes (150 s against 3600 s). It reverses on shocks (inviscid Burgers, Euler) and on Kovasznay in Bioli et al.
- The cross-paper benchmarks are not comparable to each other: SOAP numbers use PirateNet and long Adam-scale schedules; the Gauss-Newton numbers use plain small MLPs. SOAP on a plain MLP in 2604.05230 reaches only 4.2e-4 on Helmholtz and 1.7e-2 on viscous Burgers.
- DSGNAR's table compares its own H100 times with times published by others on other hardware; the speed ratios in it are not a controlled measurement.

### Gaps
- KFAC paper: final L2 errors per optimizer are only in figures; not extracted.
- NNCG: no measured wall-clock comparison with L-BFGS found in the text read; the paper only states NNCG is slower per iteration.
- ANaGRAM: experimental tables not read.
- The ENGD table for the Hilbert natural gradient on Poisson 2D came through as median 1.2, min 4.0, max 2.1, which cannot be right as ordered; check the PMLR PDF before quoting H-NGD.
- DSGNAR rows for wave, KdV and multi-scale problems were ambiguous in extraction and are omitted. Collocation counts for DSGNAR were not found.
- No KdV or Navier-Stokes result for ENGD itself; no heat-equation result for SSBroyden other than Bioli's 3+1D case.
- Ginzburg-Landau, Lorenz and wedge results in Kiyani et al. were in the truncated part.

## Which methods are matrix-free and which form the Gram or Jacobian matrix? What does each cost per step and how does it scale?

### Takeaway
Most high-accuracy results form a matrix: the P x P Gramian (ENGD, LM), the N x N kernel (Woodbury ENGD, dense D-NGD, the NG of 2604.05230, ANaGRAM), a dense P x P inverse Hessian (BFGS family), or a sketched Jacobian (DSGNAR). Truly matrix-free CG variants exist (Jnini et al. 2024, Hessian-free baseline, NNCG, PCG D-NGD, Bioli et al.) and all report that unpreconditioned CG is limited by the Gramian's conditioning.

### Cited Findings
- ENGD: forms the P x P Gramian and solves a least-squares problem by SVD; solve cost stated as O(P^3) against O(P) for a gradient; grid line search on a logarithmic grid in [0, 1]. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163)
- KFAC paper's summary of prior second-order PINN methods: they "only scale to networks with a few thousand parameters"; ENGD "runs out of memory for networks with more than 10 000 parameters". KFAC stores and inverts two Kronecker factors per layer. — [arXiv HTML](https://arxiv.org/html/2405.15603v1)
- Taylor-mode in the KFAC paper: for a PDE of order k in d dimensions the number of derivative vectors propagated is S = C(d+k, k); the forward Laplacian fuses them. — [arXiv HTML](https://arxiv.org/html/2405.15603v1)
- Hessian-free baseline in the KFAC paper: truncated CG with exact Gramian-vector products, adaptive damping, CG backtracking and backtracking line search; best run on 2d Poisson (D = 257) used GGN curvature, initial damping 100, max 50 CG iterations. — [arXiv HTML](https://arxiv.org/html/2405.15603v1)
- Jnini, Vella, Zeinhofer: matrix-free product G v = J^T (J v) by forward then reverse AD, claimed to need "only constant overhead over a gradient computation"; CG for large networks and a direct solver otherwise; damping min(1e-5, L) Id; logarithmic-grid line search. The dense Gramian for the 91k-parameter network would need about 62.79 GB. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Woodbury ENGD: solves J^T (J J^T + lambda I)^-1 r in sample space; per-iteration cost drops from O(P^3) to O(N^2 P). The authors conclude "Woodbury is the right way to implement second-order methods such as ENGD." SPRING: phi_k = mu phi_(k-1) + J^T (J J^T + lambda I)^-1 (r_k - mu J phi_(k-1)); removes the line search. — [arXiv HTML](https://arxiv.org/html/2505.12149v2)
- D-NGD complexity table: primal Gauss-Newton / LM O(m n^2 + n^3) time, O(n^2) memory; dense dual O(m^2 n + m^3) time, O(m^2) memory; dual PCG about O(t (m n + m l)) per solve for t iterations and l Nyström landmarks, memory linear in m and n plus O(l^2). Dense is efficient "for m up to a few thousand". PCG uses kernel-vector products without forming J or the kernel. — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- ANaGRAM: cost min(P^2 S, S^2 P). — [arXiv](https://arxiv.org/abs/2412.10782v2)
- Bioli et al., three matvec strategies: precomputed Gramian (setup O(q [FP] + p^2 q), matvec O(p^2 l), memory O(p^2 + p B)); precomputed Jacobian (setup O(q [FP]), matvec O(p q l), memory O(p q)); batch-wise recomputation (no setup, matvec O(q [FP] + p q l), memory O(p B)). Matvecs use forward-mode JVPs and reverse-mode VJPs. Nyström preconditioner: build O(C_mv(l) + p l^2), apply O(p l), sketch size tied to the effective dimension d_eff(mu) = sum lambda_i / (lambda_i + mu). Settings: rank at most 500, at most 20 PCG iterations; the unpreconditioned variant was given up to 520 CG iterations to match the matvec budget. — [arXiv HTML](https://arxiv.org/html/2505.11638v6)
- NNCG: Hessian-vector products cost O((n_res + n_bc) p) against O(m p) for an L-BFGS update; sketch size 60, preconditioner refreshed every 20 iterations, up to 1000 CG iterations at tolerance 1e-16, damping mu in {1e-5, ..., 1e-1} with 1e-2 or 1e-1 best, Armijo backtracking. — [arXiv HTML](https://arxiv.org/html/2402.01868v1)
- DSGNAR: forms the Jacobian only in sketched form (CountSketch on residual rows, subsampled randomized cosine transform on parameter columns), then one SVD of the s x s sketch gives LM steps for any lambda; it avoids the normal equations because they square the condition number. Naive LM is stated as O(N d^2 + d^3). Recommended sketch size is one third to one half of the parameter count. It uses reverse mode on sub-batches, argued to be "far cheaper than forward-mode" when the sub-batch is much smaller than the parameter count. — [arXiv HTML](https://arxiv.org/html/2607.02194)
- NG in 2604.05230: uses the kernel trick (an N x N system in the batch size), not a matrix-free solve. Roofline on an H100 (FP64 peak 67 TFLOP/s): the Stokes Gauss-Newton kernel has arithmetic intensity 391.4 FLOP/byte and reaches 29.640 TFLOP/s; dense SSBFGS at 33,667 parameters holds a Hessian of roughly 9 GB, and its memory footprint "becomes a dominant factor" as networks grow; the small 1D inviscid Burgers kernel is memory-bound. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- LM in Bonfanti et al.: stores J^T J; "the challenge of storing the matrix J^T J in GPU memory becomes infeasible" for medium-to-large networks; inexact LM with LSQR / LSMR on Jacobian-vector products is suggested, not tested. — [arXiv HTML](https://arxiv.org/html/2402.03864)
- Shallow LM (Shahab et al.): explicit Jacobian from hand-derived formulas for tanh networks. — [arXiv HTML](https://arxiv.org/html/2602.08515)
- SOAP: O(n^2) storage and O(n^3) computation per n x n weight matrix (eigendecompositions of Kronecker factors). — [arXiv HTML](https://arxiv.org/html/2502.00604v2)

### Inferences
- Summary by structure. Forms P x P: ENGD, LM (Bonfanti, Shahab), BFGS / SSBFGS / SSBroyden (dense inverse Hessian; inferred from the update formulas and the 9 GB figure). Forms N x N: Woodbury ENGD, SPRING, dense D-NGD, NG of 2604.05230, ANaGRAM (whichever side is smaller). Factored: KFAC, SOAP. Sketched: DSGNAR. Matrix-free CG: Jnini et al. 2024, Hessian-free baseline, NNCG (Hessian, not Gauss-Newton), PCG D-NGD, Bioli et al.
- For the small networks of interest (P about 1e3, N about 1e3 to 1e4), the dense routes are cheap: J is 8 to 80 MB in float64 and a P x P Cholesky is about P^3 / 3, roughly 3e8 flops. A matrix-free solver competes there against a dense solve that costs little, so its advantage must come from skipping Jacobian assembly, not from the linear algebra. None of the papers read measures this crossover.
- A JVP of the PDE residual with respect to parameters, in a Taylor-mode engine, means differentiating the Taylor-mode forward pass once more in forward mode; Bioli et al. and Jnini et al. obtain it from JAX's generic AD. No paper read builds a dedicated compiled kernel for the pair (J v, J^T w).

### Gaps
- CG iteration counts actually used per outer step are not reported by Jnini et al. (only tolerance 1e-5) or, in the part read, by Bioli et al. (only caps).
- No per-step timings in milliseconds for matrix-free against dense at a fixed (P, N) on the same hardware were found, on GPU or CPU.
- KFAC per-iteration complexity is not given as a formula in the text read.

## What failure modes are documented?

### Takeaway
Documented failures cluster into five groups: ill-conditioned Gramians that stall CG or need damping; memory and cubic cost at scale; fragility under minibatching and single precision; stand-alone instability early in training (saddles, no convergence without warm-up); and problem classes where curvature does not help (shocks, hyperbolic PDEs).

### Cited Findings
- Conditioning of the inner solve: matrix-free CG "avoid[s] explicit matrix inversion", but the Gramian's ill-conditioning slows CG; unpreconditioned matrix-free NGD is clearly less accurate than the preconditioned variants. — [arXiv abs](https://arxiv.org/abs/2505.11638); [arXiv HTML](https://arxiv.org/html/2505.11638v6)
- PINN Balls leaves matrix-free CG out of scope because such methods "are often sensitive to poor conditioning of the underlying matrix" (from a search snippet of the PDF, not the full text). — [arXiv PDF](https://arxiv.org/pdf/2510.21262)
- In the matrix-free Navier-Stokes experiment, accuracy is "primarily constrained" by the CG tolerance, and soft constraints cost about an order of magnitude against hard constraints. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Stand-alone instability: E-NGD "occasionally failed to train" on the heat equation (max error 5.6e-1 over 10 seeds), mitigated by GD or Adam pre-training; the Hilbert natural gradient did not converge on Poisson. On Kovasznay, ENGD used alone gave no convergence: "We were however not able to achieve any convergence when used as a stand-alone method." — [ar5iv ENGD](https://ar5iv.labs.arxiv.org/html/2302.13163); [ar5iv GNNG](https://ar5iv.labs.arxiv.org/html/2402.10680)
- Saddle points: Rathore et al. run Adam first because Newton-type methods are attracted to saddles; Urbán et al. state quasi-Newton methods can be trapped at saddle points and also start with Adam. — [arXiv HTML](https://arxiv.org/html/2402.01868v1); [arXiv HTML](https://arxiv.org/html/2405.04230v2)
- Cost at scale: ENGD out of memory above about 10,000 parameters; LM's J^T J infeasible for medium-to-large networks; dense SSBFGS Hessian about 9 GB at 33,667 parameters. — [arXiv HTML](https://arxiv.org/html/2405.15603v1); [arXiv HTML](https://arxiv.org/html/2402.03864); [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Stochastic batches: Wang et al. exclude natural gradient methods from their main comparison, saying they need full-batch training, float64 and small networks, and that in their own Poisson and heat tests NGD failed for some seeds; they exclude L-BFGS variants because gradient noise disrupts Hessian updates and the line search. — [arXiv HTML](https://arxiv.org/html/2502.00604v2)
- Batched quasi-Newton: standard L-BFGS-style batching is unstable for PINNs; the fix in 2604.05230 updates the Hessian approximation only when a curvature condition holds. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Precision: double precision is called "crucial for achieving high accuracy" (Jnini et al.) and is used throughout 2604.05230, ENGD, KFAC, Bioli et al. DSGNAR is the exception, claiming robustness to precision with a single-precision Burgers result of 4.75e-7. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680); [arXiv abs](https://arxiv.org/abs/2607.02194)
- Damping and hyperparameters: KFAC authors did not explore adaptive damping or heuristics for splitting damping across factors; tuned damping values in that paper range from 1e-10 to 1e-12 for ENGD and KFAC and an initial 100 for Hessian-free. ENGD-W and SPRING hyperparameters "are not yet adaptive", so the methods are not black-box. D-NGD lists automated damping from stochastic spectral estimates as future work. — [arXiv HTML](https://arxiv.org/html/2405.15603v1); [arXiv HTML](https://arxiv.org/html/2505.12149v2); [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Randomization limits: Nyström sketching helps only early in training on low-dimensional problems with large batches (about 2x at a 10 % sketch on 5d Poisson, none above 25 %); it fails late in training and in high dimension, because the regularized kernel's effective dimension plateaus above 50 % of the batch size. — [arXiv HTML](https://arxiv.org/html/2505.12149v2)
- Problem classes: on inviscid Burgers and Euler, NG is slower and no more accurate than SSBroyden; "optimization alone is not sufficient for discontinuous problems". NG was "particularly sensitive to increased collocation density near drug administration" in the stiff PK-PD case. Quasi-Newton methods needed warm-up or a domain-expansion curriculum on high-wavenumber Helmholtz. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Curricula still needed: D-NGD at Re = 3000 used a Reynolds curriculum (100, 400, 1000). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Geodesic acceleration is not uniformly better (Kovásznay 5.49e-7 with it against 5.23e-7 without). — [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Line search matters for quasi-Newton methods: Wolfe gave the lowest error with SSBroyden in PyTorch runs, trust region was more efficient on Helmholtz in the 2026 study; BFGS-type methods "require significantly more iterations" than the Adam warm-up. — [arXiv HTML](https://arxiv.org/html/2501.16371v6); [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- LM per-iteration cost: on inverse Allen-Cahn the shallow-network LM took 1677 against 834 for BFGS, while reaching a lower loss. — [arXiv HTML](https://arxiv.org/html/2602.08515)
- Loss-weight sensitivity: De Ryck et al. find the boundary weight that minimizes the condition number scales roughly like K^2 for Poisson with maximum frequency K. DSGNAR includes automatic condition reweighting at each iteration. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05801); [arXiv HTML](https://arxiv.org/html/2607.02194)

### Inferences
- The recurring recipe (Adam warm-up, then the second-order method; float64; full batch or slowly resampled batch) is itself evidence that these methods are local refiners, not global optimizers.
- For a matrix-free CG design the specific documented risk is the inner solve: without a preconditioner the published evidence says accuracy drops, and with a 1e-5 tolerance the outer accuracy saturates near 1e-5.
- Sensitivity of Gauss-Newton steps to loss weights between residual, boundary and data rows is discussed only indirectly; no paper read reports a weight sweep for a Gauss-Newton or LM optimizer.

### Gaps
- No study found that measures the effect of single precision on CG-based Gauss-Newton specifically.
- No systematic study of damping schedules (LM gain ratio, trust region, fixed) for PINNs apart from DSGNAR's target-ratio rule.
- Inverse problems: only Shahab et al. test LM on inverse problems; 2604.05230 and Urbán et al. state inverse problems are left open.

## What explains why first-order methods stall on PINNs?

### Takeaway
The accepted explanation is conditioning inherited from the differential operator: the PINN loss Hessian behaves like the Hermitian square of the PDE operator composed with the tangent kernel, so gradient descent needs a number of steps proportional to a condition number that grows with frequency content and with the number of residual points. A second, complementary explanation is conflicting gradients between loss terms.

### Cited Findings
- De Ryck, Bonnet, Mishra and de Bézenac, "An operator preconditioning perspective on training in physics-informed machine learning", ICLR 2024 (peer-reviewed). Gradient descent converges at rate (1 - c / kappa(A))^k, so the step count scales as O(kappa(A) ln(1 / eps)); kappa(A) >= kappa(A_op o T T*), where A_op = D* D is the Hermitian square of the PDE operator and T T* the tangent-kernel integral operator. — [ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/f122a6778999d1d2b25cdc6927930f1a-Abstract-Conference.html); [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05801)
- Same paper: for Poisson with Fourier features the unpreconditioned condition number satisfies kappa >= K^4 for maximum frequency K; rescaling features gives kappa = 1 + O(1 / gamma). For linear advection the condition number grows like beta^2. For linear models the Hessian equals A, so quasi-Newton methods implicitly approximate its inverse. Limits: linear PDEs, NTK regime, no universal preconditioner for neural networks. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05801)
- Rathore et al.: top Hessian eigenvalues exceed 1e4 (convection), 1e3 (reaction) and 1e5 (wave); the residual loss, which contains the differential operator, is the worst-conditioned component; L-BFGS preconditioning lowers the top eigenvalue and condition number by at least 1e3. Theorem 8.4 (informal): if eigenvalues of the operator composed with the kernel decay as O(j^(-2 beta)) with beta > 1/2, then the condition number is Omega(n_res^beta) with high probability. — [arXiv HTML](https://arxiv.org/html/2402.01868v1)
- Rathore et al.: on convection a loss near 1e-3 gives L2RE near 1e-1 and a loss near 1e-5 gives L2RE near 1e-2, so low error requires very low loss. Gradient descent followed by damped Newton contracts the loss by (2/3)^k in the second phase, independent of the condition number, but only locally. — [arXiv HTML](https://arxiv.org/html/2402.01868v1)
- Urbán et al.: on the current-free Grad-Shafranov problem the Hessian condition number is about 1e12; reaching a loss of about 1e-13 took about 40,000 BFGS iterations against about 1,000 with the self-scaled modifications. — [arXiv HTML](https://arxiv.org/html/2405.04230v2)
- Bonfanti et al.: for nonlinear PDEs the Hessian term does not vanish even at infinite width, so first-order (NTK-linear) reasoning fails; in the linear regime the benefit of second order is relief from spectral bias. — [arXiv HTML](https://arxiv.org/html/2402.03864)
- Wang et al.: directional gradient conflicts between loss terms force first-order methods to zigzag; they define a gradient alignment score and argue SOAP approximates the Newton preconditioner. — [arXiv HTML](https://arxiv.org/html/2502.00604v2)
- Müller and Zeinhofer and Best McKay et al.: Gauss-Newton / energy natural gradient steps are Newton steps in function space projected on the tangent space, which removes the parameterization's distortion (J J^T replaced by the identity). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163); [arXiv abs](https://arxiv.org/abs/2605.11316)

### Inferences
- The two theories predict where Gauss-Newton helps most: smooth elliptic and parabolic problems where the operator's conditioning is the bottleneck. This matches the empirical recommendation in 2604.05230 and the weak results on shocks.
- Theorem 8.4 of Rathore et al. implies conditioning worsens as collocation points are added, which predicts more CG iterations per step at larger N for an unpreconditioned matrix-free solver. No paper read tests this directly.

### Gaps
- The theory is for linear PDEs or local regimes; no quantitative conditioning theory was found for nonlinear PDEs with finite-width networks.
- No measurement found of how the Gauss-Newton matrix's spectrum (as opposed to the Hessian's) changes during training on PINNs, apart from the effective-dimension observation in Guzmán-Cordero et al.

## What remains unsolved or untested (CPU, small networks, forward-mode or Taylor-mode derivatives, equal wall-clock time)?

### Takeaway
Nearly all evidence is from GPUs in float64 with JAX or PyTorch generic AD; in what was read, the only CPU evidence is a MATLAB dense-Jacobian LM study and slow SciPy runs, and no paper measures a matrix-free Gauss-Newton or LM solver inside a compiled CPU engine with dedicated Taylor-mode JVP / VJP kernels. That combination, measured at equal wall-clock time against tuned Adam and L-BFGS on the same engine, appears to be open.

### Cited Findings
- Hardware across the literature: RTX 3080 Laptop (ENGD), RTX 6000 (KFAC, Woodbury ENGD), A100 80 GB (GNNG, D-NGD), RTX 4090 (Bioli et al.), A10 (Bonfanti et al.), RTX A6000 / 3090 (Kiyani et al.), H100 (2604.05230, DSGNAR). — [ar5iv](https://ar5iv.labs.arxiv.org/html/2302.13163); [arXiv HTML](https://arxiv.org/html/2405.15603v1); [arXiv HTML](https://arxiv.org/html/2505.21404v2); [arXiv HTML](https://arxiv.org/html/2505.11638v6); [arXiv HTML](https://arxiv.org/html/2604.05230v1); [arXiv HTML](https://arxiv.org/html/2607.02194)
- CPU evidence: Shahab et al. run LM in MATLAB on a six-core Core i5-10400 with networks of 751 to 1552 parameters and 10,000 collocation points, with an explicit Jacobian. In 2604.05230 the SciPy-based PyTorch runs are on CPU and are 15 to 25 times slower than the Optax GPU runs for the same optimizer (SSBroyden 8215 s against 319 s on Helmholtz). — [arXiv HTML](https://arxiv.org/html/2602.08515); [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- Taylor-mode use so far: the KFAC paper uses it to structure the curvature approximation; D-NGD uses JAX Taylor-mode AD for the residual. Both on GPU. — [arXiv HTML](https://arxiv.org/html/2405.15603v1); [arXiv HTML](https://arxiv.org/html/2505.21404v2)
- Forward mode for the Gauss-Newton product: used through JAX in Jnini et al. and Bioli et al.; DSGNAR argues reverse mode on sub-batches is "far cheaper than forward-mode" for its sketched Jacobian and does not mention Taylor mode; 2604.05230 does not discuss matrix-free, Taylor-mode or forward-mode implementations. — [ar5iv](https://ar5iv.labs.arxiv.org/html/2402.10680); [arXiv HTML](https://arxiv.org/html/2505.11638v6); [arXiv HTML](https://arxiv.org/html/2607.02194); [arXiv HTML](https://arxiv.org/html/2604.05230v1)
- A targeted search for a PINN Gauss-Newton with forward-mode JVPs on a CPU with small networks and a wall-clock comparison against L-BFGS returned no matching paper. — [arXiv HTML, nearest hit](https://arxiv.org/html/2602.08515)
- Open items named by the authors themselves: inexact LM with LSQR / LSMR on Jacobian-vector products (Bonfanti et al.); adaptive damping and removing matrix inversions for KFAC (Dangel et al.); adaptive hyperparameters and sketch rank (Guzmán-Cordero et al.); curvature-driven residual sampling and automated damping (Jnini and Vella); measured peak memory (Bioli et al.); inverse problems (Urbán et al.; 2604.05230); sketch theory for the doubly-sketched square system (Webb et al.); nonlinear PDEs in the conditioning theory (De Ryck et al.). — [arXiv HTML](https://arxiv.org/html/2402.03864); [arXiv HTML](https://arxiv.org/html/2405.15603v1); [arXiv HTML](https://arxiv.org/html/2505.12149v2); [arXiv HTML](https://arxiv.org/html/2505.21404v2); [arXiv HTML](https://arxiv.org/html/2505.11638v6); [arXiv HTML](https://arxiv.org/html/2607.02194); [ar5iv](https://ar5iv.labs.arxiv.org/html/2310.05801)
- Small, memory-bound kernels behave differently from the large ones: in the H100 roofline the 1D inviscid Burgers kernel is memory-bound because the network is small. — [arXiv HTML](https://arxiv.org/html/2604.05230v1)

### Inferences
- Already known, so not a contribution: Gauss-Newton / LM beats Adam and L-BFGS by two to five orders of magnitude in error on smooth PDEs with small networks; the sample-space (N x N) form is the right dense formulation when N < P; matrix-free CG needs a preconditioner; float64 matters; Taylor mode pairs naturally with these methods.
- Plausibly new for a compiled Taylor-mode CPU engine: (1) dedicated compiled kernels for J v and J^T w of the PDE residual, with measured cost relative to one gradient (Jnini et al. claim "constant overhead" without timings); (2) the dense-versus-matrix-free crossover in (P, N) on a CPU, including networks under 1,000 parameters where no GPU paper operates efficiently; (3) equal-wall-clock curves against Adam and L-BFGS on the same engine and the same machine, which removes the framework and hardware confound present in every cross-paper comparison; (4) CG iteration counts per outer step as a function of N and damping, which the literature does not report; (5) LM on inverse problems and on parameter-plus-network unknowns, tested only once (Shahab et al.); (6) behavior in single precision for CG-based steps.
- Risks to the novelty claim: DSGNAR (July 2026) already reports seconds-scale training on Burgers (9.8 s, single precision, 1,447 parameters, H100), and Shahab et al. already show CPU LM on sub-1k-parameter networks. A CPU result needs to be positioned against both, ideally by reproducing their Burgers setups.
- Because small-network kernels are memory-bound on a GPU, a CPU with the working set in cache may be competitive in this regime; that is a hypothesis the literature read neither supports nor refutes with measurements.

### Gaps
- No paper found reporting Gauss-Newton or LM for PINNs with energy or time per solution on a CPU against a GPU.
- No reproduction study found that reruns ENGD, SSBroyden, NG and DSGNAR on one machine with one framework; 2604.05230 is the closest, without DSGNAR or KFAC.
- Remaining parts of several papers (later appendices of 2604.05230, Bioli et al., DSGNAR, Rathore et al.) were not read; statements that something is "not mentioned" refer to the parts read.
- Work between July and November 2026 was covered only as far as the search surfaced it (DSGNAR, arXiv:2607.07623, arXiv:2608.05892); a further sweep of late-2026 preprints would be prudent before a novelty claim is written.
