# Adam Paper Reproduction (PyTorch)

This repository contains a full PyTorch reproduction of the headline experiments from the paper [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) (Kingma & Ba, 2015).

## Structure
- `experiments/`: Reproduced experiments from the paper.
- `optimizers/`: Custom `AdamUncorrected` subclass for the ablation study.
- `results/`: Generated figures.
- `report.md`: The scientific report.

## Requirements
- `torch`
- `torchvision`
- `matplotlib`
- `numpy`

## How to Reproduce
Run all experiments with a single command:

```shell
python main.py --all
```
Results will be saved under `results/fig<N>_reproduction`.

or run one of 4 individual experiments (`logistic`, `multilayer`, `convolutional`, `ablation`) using:
```shell
python main.py --experiment <experiment_name>
```