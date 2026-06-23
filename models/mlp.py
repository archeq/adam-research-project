"""Multi-layer perceptron model for MNIST classification (Figure 2 reproduction)."""
import torch.nn as nn


def get_mlp(dropout=False):
    """Create a two-hidden-layer MLP (1000-1000) for 28x28 grayscale images.

    Args:
        dropout: If True, add 50% dropout after each hidden layer.

    Returns:
        nn.Sequential: MLP model with optional dropout regularization.
    """
    layers = [
        nn.Flatten(),
        nn.Linear(28 * 28, 1000),
        nn.ReLU()
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
        
    layers.extend([
        nn.Linear(1000, 1000),
        nn.ReLU()
    ])
    if dropout:
        layers.append(nn.Dropout(0.5))
        
    layers.append(nn.Linear(1000, 10))
    
    return nn.Sequential(*layers)
