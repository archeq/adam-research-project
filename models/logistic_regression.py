"""Logistic regression model for MNIST classification (Figure 1 reproduction)."""
import torch.nn as nn


def get_logistic_regression():
    """Create a logistic regression model for 28x28 grayscale images (10 classes).

    Returns:
        nn.Sequential: A simple linear classifier (Flatten → Linear).
    """
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10)
    )
