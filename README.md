# Adam: A Method for Stochastic Optimization — PyTorch Reproduction

> **Paper:** Kingma, D.P. & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* Published at ICLR 2015.
> [arXiv:1412.6980](https://arxiv.org/abs/1412.6980)

## Overview

This repository contains a complete PyTorch reproduction of the headline experiments from the Adam optimizer paper (Kingma & Ba, 2015). We reproduce Figures 1–4 from the original paper — comparing Adam against SGD, AdaGrad, and RMSProp on logistic regression, a multi-layer perceptron, and a CNN — and replicate the bias-correction ablation study using a Variational Autoencoder (VAE). All experiments run on a standard laptop CPU within approximately 2.5 hours total.

## Reproduced Results

We reproduce the following specific results from the paper:

| Figure | Description | Dataset | Status |
|--------|-------------|---------|--------|
| Figure 1 | Logistic regression training cost comparison | MNIST | ✅ Reproduced |
| Figure 2 | MLP with dropout training cost comparison | MNIST | ✅ Reproduced |
| Figure 3 | CNN training cost comparison | CIFAR-10 | ✅ Reproduced |
| Figure 4 | Bias-correction ablation (VAE) | MNIST | ✅ Reproduced |

## Repository Structure

```
adam-research-project/
├── main.py                        # Main experiment runner (CLI interface)
├── models/
│   ├── logistic_regression.py     # Logistic regression model (Figure 1)
│   ├── mlp.py                     # Multi-layer perceptron with dropout (Figure 2)
│   ├── cnn.py                     # Convolutional neural network (Figure 3)
│   └── vae.py                     # Variational autoencoder (Figure 4)
├── optimizers/
│   └── adam_uncorrected.py        # Adam with toggleable bias correction
├── utils/
│   ├── data.py                    # MNIST and CIFAR-10 data loaders
│   └── plot.py                    # Figure generation and plotting
├── results/                       # Generated figures (after running experiments)
│   ├── fig1_logreg.png            # Reproduction of Figure 1
│   ├── fig2_mlp.png               # Reproduction of Figure 2
│   ├── fig3_cnn.png               # Reproduction of Figure 3
│   └── fig4_vae_ablation.png      # Reproduction of Figure 4
├── docs/
│   ├── 0_adam_main.pdf            # Original paper (PDF)
│   └── adam.md                    # Paper notes / markdown summary
├── data/                          # Auto-downloaded datasets (MNIST, CIFAR-10)
├── report.md                      # Scientific reproduction report
├── slides.md                      # Presentation slides (Marp format)
├── criteria.md                    # Project assignment criteria
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Requirements

**Python 3.8+** with the following packages:

- `torch` — deep learning framework
- `torchvision` — datasets and transforms
- `matplotlib` — figure generation

Install all dependencies:

```bash
pip install -r requirements.txt
```

## How to Reproduce

### Full pipeline (all experiments)

```bash
python main.py --all
```

This runs all four experiments sequentially and generates all figures in the `results/` directory.

### Individual experiments

Run specific experiments independently:

```bash
python main.py --logreg    # Figure 1: Logistic Regression (MNIST)
python main.py --mlp       # Figure 2: MLP with Dropout (MNIST)
python main.py --cnn       # Figure 3: CNN (CIFAR-10)
python main.py --vae       # Figure 4: VAE Bias-Correction Ablation (MNIST)
```

### Combining experiments

Flags can be combined to run a subset:

```bash
python main.py --logreg --mlp    # Run only Figures 1 and 2
```

### First run

On the first run, MNIST (~11 MB) and CIFAR-10 (~170 MB) will be automatically downloaded via `torchvision.datasets` into the `data/` directory.

## Expected Runtime

Approximate wall-clock times on a modern CPU (no GPU):

| Experiment | Approximate Time |
|------------|-----------------|
| Logistic Regression (45 epochs) | ~5 minutes |
| MLP with Dropout (45 epochs) | ~15 minutes |
| CNN on CIFAR-10 (45 epochs × 3 optimizers) | ~90 minutes |
| VAE Ablation (multiple configs × 10/100 epochs) | ~30 minutes |
| **Total (`--all`)** | **~2.5 hours** |

If a CUDA-capable GPU is available, PyTorch will automatically use it, reducing runtimes significantly (especially for the CNN experiment).

## Reproducibility

- **Deterministic training:** For reproducible results, set a fixed random seed before running:
  ```python
  torch.manual_seed(42)
  ```
- **Hardware:** All experiments are designed to run on CPU. GPU acceleration is supported but not required.
- **Datasets:** MNIST and CIFAR-10 are public, freely available, and downloaded automatically.
- **Framework:** Results may differ slightly from the original paper (which used Theano) due to differences in default weight initialization, dropout implementation, and numerical backends.

## Output

After running `python main.py --all`, the following figures are generated in `results/`:

| File | Content |
|------|---------|
| `fig1_logreg.png` | Training cost curves for logistic regression — Adam vs. SGD+Nesterov vs. AdaGrad (cf. Figure 1 in Kingma & Ba, 2015) |
| `fig2_mlp.png` | Training cost curves for MLP with dropout — Adam vs. RMSProp vs. AdaGrad vs. SGD (cf. Figure 2a) |
| `fig3_cnn.png` | Training cost curves for CNN on CIFAR-10 — Adam vs. AdaGrad vs. SGD (cf. Figure 3) |
| `fig4_vae_ablation.png` | Bias-correction ablation on VAE — corrected vs. uncorrected Adam across $\beta_2$ and $\alpha$ values (cf. Figure 4) |

## Report & Presentation

- **Scientific report:** [`report.md`](report.md) — Covers paper summary, method derivation, experimental setup, results analysis, ablation findings, and reproducibility notes (~10–15 pages).
- **Presentation slides:** [`slides.md`](slides.md) — 14 slides in [Marp](https://marp.app/) format for a 10-minute talk + 5-minute Q&A. Render with:
  ```bash
  npx @marp-team/marp-cli slides.md --html --pdf
  ```

## Reference

```bibtex
@inproceedings{kingma2015adam,
  title     = {Adam: A Method for Stochastic Optimization},
  author    = {Kingma, Diederik P. and Ba, Jimmy},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2015},
  url       = {https://arxiv.org/abs/1412.6980}
}
```