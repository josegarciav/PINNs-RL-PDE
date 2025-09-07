# PINNs-RL-PDE

Physics-Informed Neural Networks (PINNs) combined with reinforcement learning (RL) and adaptive weighting strategies for solving **Partial Differential Equations (PDEs)**.
This is an **active personal research project**.

---

## Overview

This repository provides a framework to:

* Solve or simulate PDEs using PINNs with physics-based loss terms.
* Explore adaptive weighting strategies and sampling schedules for collocation points.
* Compare architectures and optimization methods for stability and convergence.
* Visualize training and PDE solutions interactively through a GUI (`interactive_trainer.py`).

---

## Current focus

* Implementing PINNs for canonical PDEs (Heat, Allen–Cahn, KdV).
* Studying adaptive collocation point selection.
* Testing curriculum and multi-objective loss balancing.
* Providing an interactive GUI for experiments.

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/josegarciav/PINNs-RL-PDE.git
cd PINNs-RL-PDE
```

### 2. Create and activate a virtual environment called `pinn`

Using Python’s built-in `venv`:

```bash
python3 -m venv .venv/pinn
source .venv/pinn/bin/activate
```

You should now see `(pinn)` in your terminal prompt.

### 3. Install dependencies

Upgrade pip and install requirements:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the interactive trainer (GUI)

From the project root:

```bash
python src/interactive_trainer.py
```

---

## Roadmap

* [ ] Aligning dashboard to account for real-time training metrics
* [ ] Adding a docs page
* [ ] Adaptive collocation via residual-driven sampling
* [ ] Automated multi-objective weight tuning
* [ ] Domain decomposition + V-cycle training strategies
* [ ] RL-based schedulers for adaptive training

---

## License

MIT

## Contributing

Contributions are welcome! If you’d like to propose improvements, bug fixes, or new features, please [reach out](https://josegarciav.github.io/).
