import torch.nn as nn

def get_cnn(dropout=True):
    # Architecture: c64-c64-c128-1000
    layers = [
        # 3x32x32 -> 64x32x32 -> 64x16x16
        nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        
        # 64x16x16 -> 64x16x16 -> 64x8x8
        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        
        # 64x8x8 -> 128x8x8 -> 128x4x4
        nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        
        nn.Flatten()
    ]
    
    if dropout:
        layers.append(nn.Dropout(0.5))
        
    layers.extend([
        nn.Linear(128 * 4 * 4, 1000),
        nn.ReLU()
    ])
    
    if dropout:
        layers.append(nn.Dropout(0.5))
        
    layers.append(nn.Linear(1000, 10))
    
    return nn.Sequential(*layers)
