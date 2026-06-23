"""Data loading utilities for MNIST and CIFAR-10 datasets."""
import torch
from torchvision import datasets, transforms
import os


def get_mnist_loaders(batch_size=128, data_dir="data"):
    """Create MNIST train and test DataLoaders.

    Args:
        batch_size: Batch size for both loaders (default: 128).
        data_dir: Directory to store/load MNIST data.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, persistent_workers=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, persistent_workers=True, pin_memory=True
    )
    
    return train_loader, test_loader


def get_cifar10_loaders(batch_size=128, data_dir="data"):
    """Create CIFAR-10 train and test DataLoaders.

    Args:
        batch_size: Batch size for both loaders (default: 128).
        data_dir: Directory to store/load CIFAR-10 data.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(data_dir, train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, persistent_workers=True, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, persistent_workers=True, pin_memory=True
    )
    
    return train_loader, test_loader
