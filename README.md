# Adam Paper Reproduction (PyTorch)

This repository contains a full PyTorch reproduction of the headline experiments from the paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2015).

## Structure
- `models/`: PyTorch Architectures (Logistic Regression, MLP, CNN, VAE).
- `optimizers/`: Custom `AdamUncorrected` subclass for the ablation study.
- `utils/`: Data loaders (`torchvision`) and plotting logic.
- `results/`: Generated figures.
- `report.md`: The scientific report.
- `slides.md`: Presentation slides.

## Requirements
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

## How to Reproduce
Run all experiments with a single command:
```bash
python main.py --all
```
This will automatically stream the datasets, execute the full training loop across all models on your local CPU/GPU, and generate the reproduction plots in the `results/` directory.