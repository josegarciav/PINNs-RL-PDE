# The software paper

A paper describes the package's goals, its design and its benchmarks in one citable place: the statement of need, the state of the field, the trainer, the inverse-modelling toolbox, the process units, the native engine, and tables from the experiments (the training step against PyTorch, JAX and DeepXDE; forward accuracy against classical solvers; sampling at equal compute; parameter identification and calibration; the process twins against their classical estimators).

- **Source:** [`paper/paper.md`](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/paper/paper.md), with its references in [`paper/paper.bib`](https://github.com/josegarciav/PINNs-RL-PDE/blob/main/paper/paper.bib).
- **Status:** a draft in the repository; not yet submitted.
- **Reproducing its tables:** each one names the folder under [`experiments/`](https://github.com/josegarciav/PINNs-RL-PDE/tree/main/experiments) whose script produces it; the same numbers are on the [benchmarks page](benchmarks.md).

## Citing `pinnrl`

Until the paper is published, cite the software:

```bibtex
@software{garcia_pinnrl,
  author = {Garcia Ventura, Jose},
  title  = {pinnrl: physics-informed neural networks with adaptive collocation sampling},
  url    = {https://github.com/josegarciav/PINNs-RL-PDE}
}
```

The package started as the MSc thesis *On Solving Partial Differential Equations (PDEs) Using Physics-Informed Neural Networks (PINNs)* (University of the Basque Country UPV/EHU, 2025), which is [here](assets/thesis.pdf).
