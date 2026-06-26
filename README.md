# Adam Paper Reproduction (PyTorch)

This repository contains a full PyTorch reproduction of the headline experiments from the paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2015).

## Structure
- `experiments/`: Reproduced experiments from the paper.
- `optimizers/`: Custom `AdamUncorrected` subclass for the ablation study.
- `report.md`: The scientific report.

## Requirements
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

Install with:
```shell
pip install -r requirements.txt
```

## How to Reproduce
Experiment reproduces 4 figures from the paper. Each one has corresponding jupyter notebook designated for it located in 
`experiments/` directory and their outcomes (figure reproductions) at `experiments/results/`
