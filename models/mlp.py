import torch.nn as nn

def get_mlp(dropout=False):
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
