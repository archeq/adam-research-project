"""
The third experiment uses a convolutional network on CIFAR-10.
Unlike fully connected networks, CNNs exhibit highly varying gradient magnitudes across layers,
making per-parameter learning rate adaptation particularly relevant.

| Parameter                   | Value                                                       | Source                      |
| --------------------------- | ----------------------------------------------------------- | --------------------------- |
| Dataset                     | CIFAR-10 (50k train / 10k test, 32×32 RGB)                  | paper                       |
| Architecture                | 3× [5×5 conv → 3×3 max-pool (stride 2)], FC 1000 units ReLU | paper                       |
| Filter counts               | 64, 64, 128 (c64-c64-c128-1000)                             | paper (figure caption)      |
| Input preprocessing         | whitening                                                   | paper                       |
| Dropout                     | applied to input layer and FC layer                         | paper                       |
| Minibatch size              | 128                                                         | paper                       |
| Optimizers                  | Adam, AdaGrad, SGDNesterov (with and without dropout)       | paper                       |
| Approximate training epochs | ~45                                                         | inferred from Fig. 3 x-axis |
| Exact whitening method      | per-channel mean subtraction + std normalization            | *(our choice)*              |
| Dropout rate                | 0.5 on FC, 0.2 on input                                     | *(our choice)*              |
| Weight initialization       | Kaiming uniform                                             | *(our choice)*              |

"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, Adagrad, SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CIFAR10CNN(nn.Module):
    def __init__(self, use_dropout=True):
        super().__init__()
        self.use_dropout = use_dropout
        # Input dropout (applied to input image)
        self.input_dropout = nn.Dropout2d(0.2) if use_dropout else nn.Identity()
        # Conv block 1: 5x5 conv, 64 filters -> 3x3 max-pool stride 2
        self.conv1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Conv block 2: 5x5 conv, 64 filters -> 3x3 max-pool stride 2
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Conv block 3: 5x5 conv, 128 filters -> 3x3 max-pool stride 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # FC layer
        self.fc1 = nn.Linear(128 * 4 * 4, 1000)
        self.fc_dropout = nn.Dropout(0.5) if use_dropout else nn.Identity()
        self.fc2 = nn.Linear(1000, 10)

        # Kaiming uniform initialization for all conv and linear layers
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_dropout(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = self.fc2(x)
        return x


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def get_cifar10_loader(train=True, batch_size=128):
    """Return a DataLoader for CIFAR-10 with per-channel whitening."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/", train=train, download=False, transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=2)


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def make_optimizer(name, model, lr):
    """Create an optimizer by name."""
    if name == "Adam":
        return Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    elif name == "AdaGrad":
        return Adagrad(model.parameters(), lr=lr)
    elif name == "SGDNesterov":
        return SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def make_scheduler(optimizer):
    """αₜ = α / √t where t is iteration count (1-indexed)."""
    return LambdaLR(optimizer, lr_lambda=lambda t: 1.0 / np.sqrt(max(t, 1)))


def train_one_config(model, optimizer, scheduler, train_loader, num_epochs, device):
    """Train *model* for *num_epochs* and return per-batch losses."""
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    batch_losses = []

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_losses.append(loss.item())

    return batch_losses


def smooth(values, window=50):
    """Simple moving-average smoothing."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    # Use 'valid' convolution then pad the beginning with the first smoothed value
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------

def grid_search(opt_name, lr_candidates, train_loader, num_epochs, device):
    """Run grid search over learning rates, return best lr and its batch losses."""
    best_lr = None
    best_cost = float("inf")
    best_losses = None

    batches_per_epoch = len(train_loader)
    last_5_epoch_batches = 5 * batches_per_epoch

    for lr in lr_candidates:
        print(f"  Grid search {opt_name} lr={lr}")
        model = CIFAR10CNN(use_dropout=True).to(device)
        optimizer = make_optimizer(opt_name, model, lr)
        scheduler = make_scheduler(optimizer)
        losses = train_one_config(model, optimizer, scheduler, train_loader, num_epochs, device)

        # Evaluate: average smoothed training cost over last 5 epochs
        smoothed = smooth(np.array(losses), window=50)
        tail = smoothed[-last_5_epoch_batches:]
        avg_cost = float(np.mean(tail))
        print(f"    avg smoothed cost (last 5 epochs) = {avg_cost:.4f}")

        if avg_cost < best_cost:
            best_cost = avg_cost
            best_lr = lr
            best_losses = losses

    print(f"  Best {opt_name} lr = {best_lr} (cost = {best_cost:.4f})")
    return best_lr, best_losses


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results, batches_per_epoch, save_path):
    """Create the two-panel Figure 3 reproduction."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"Adam": "#1f77b4", "AdaGrad": "#ff7f0e", "SGDNesterov": "#2ca02c"}

    for name in ["Adam", "AdaGrad", "SGDNesterov"]:
        raw = np.array(results[name]["losses"])
        smoothed = smooth(raw, window=50)
        epochs = np.arange(len(smoothed)) / batches_per_epoch

        label = name
        c = colors[name]

        # Left: first 3 epochs (linear scale)
        mask_left = epochs <= 3.0
        ax_left.plot(epochs[mask_left], smoothed[mask_left],
                     label=label, color=c, linewidth=1.5)

        # Right: full 45 epochs (log scale)
        ax_right.plot(epochs, smoothed, label=label, color=c, linewidth=1.5)

    # --- Left subplot formatting ---
    ax_left.set_title("First 3 Epochs")
    ax_left.set_xlabel("Epochs")
    ax_left.set_ylabel("Training Cost")
    ax_left.set_xlim(0, 3)
    ax_left.set_ylim(0.5, 3.0)
    ax_left.legend(loc="upper right")
    ax_left.grid(True, alpha=0.3)

    # --- Right subplot formatting ---
    ax_right.set_title("Full Training (45 epochs)")
    ax_right.set_xlabel("Epochs")
    ax_right.set_ylabel("Training Cost")
    ax_right.set_xlim(0, 45)
    ax_right.set_yscale("log")
    ax_right.legend(loc="upper right")
    ax_right.grid(True, alpha=0.3, which="both")

    fig.suptitle("CNN Training Cost on CIFAR-10 (c64-c64-c128-1000)", fontsize=14)
    fig.text(0.5, -0.02,
             "Reproduction of Figure 3 in Kingma & Ba (2015).",
             ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {save_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run():
    """Run the convolutional-networks experiment (Figure 3 reproduction)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    num_epochs = 45
    train_loader = get_cifar10_loader(train=True, batch_size=128)
    batches_per_epoch = len(train_loader)
    print(f"Batches per epoch: {batches_per_epoch}")

    results = {}

    # ---- Adam (fixed lr=0.001) ----
    print("Training Adam (lr=0.001) ...")
    model_adam = CIFAR10CNN(use_dropout=True).to(device)
    opt_adam = make_optimizer("Adam", model_adam, lr=0.001)
    sched_adam = make_scheduler(opt_adam)
    adam_losses = train_one_config(model_adam, opt_adam, sched_adam,
                                  train_loader, num_epochs, device)
    results["Adam"] = {"lr": 0.001, "losses": adam_losses}

    # ---- AdaGrad (grid search) ----
    print("Grid search for AdaGrad ...")
    lr_candidates = [0.01, 0.1, 0.3]
    best_lr_ada, best_losses_ada = grid_search(
        "AdaGrad", lr_candidates, train_loader, num_epochs, device)
    results["AdaGrad"] = {"lr": best_lr_ada, "losses": best_losses_ada}

    # ---- SGDNesterov (grid search) ----
    print("Grid search for SGDNesterov ...")
    best_lr_sgd, best_losses_sgd = grid_search(
        "SGDNesterov", lr_candidates, train_loader, num_epochs, device)
    results["SGDNesterov"] = {"lr": best_lr_sgd, "losses": best_losses_sgd}

    # ---- Plot ----
    save_path = "results/fig3_reproduction.png"
    plot_results(results, batches_per_epoch, save_path)

    # ---- Summary ----
    print("\n=== Results Summary ===")
    for name, info in results.items():
        final_smooth = smooth(np.array(info["losses"]), window=50)[-1]
        print(f"  {name}: lr={info['lr']}, final smoothed cost={final_smooth:.4f}")

    return results


if __name__ == "__main__":
    run()