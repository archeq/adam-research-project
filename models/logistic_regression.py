import torch.nn as nn

def get_logistic_regression():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    )
