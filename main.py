"""
Adam Paper Reproduction — Main Experiment Runner

Reproduces Figures 1-4 from:
    Kingma & Ba (2015), "Adam: A Method for Stochastic Optimization", ICLR 2015.
    arXiv:1412.6980

Usage:
    python main.py --all          # Run all experiments
    python main.py --all --seed 42 # Run all with specific seed
"""
import torch
import torch.nn as nn
import time
import argparse
import random
import numpy as np
import math

from utils.data import get_mnist_loaders, get_cifar10_loaders
from utils.plot import plot_loss_curves, plot_vae_ablation, plot_cnn_two_panel
from models.logistic_regression import get_logistic_regression
from models.mlp import get_mlp
from models.cnn import get_cnn
from models.vae import VAE, vae_loss
from optimizers.adam_uncorrected import AdamUncorrected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def set_seed(seed=42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_classifier(model, optimizer, train_loader, epochs=45, scheduler=None):
    """Train a classifier model and return per-epoch average training losses.
    
    Args:
        model: PyTorch model to train.
        optimizer: Optimizer instance.
        train_loader: DataLoader for training data.
        epochs: Number of training epochs.
        scheduler: Optional learning rate scheduler (stepped per epoch).
    
    Returns:
        List of average training losses per epoch.
    """
    criterion = nn.CrossEntropyLoss()
    losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()
        
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if scheduler is not None:
            scheduler.step()
        
        print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {time.time()-t0:.1f}s", flush=True)
        
    return losses


def experiment_logistic_regression(seed=42):
    """Reproduce Figure 1 (left): Logistic Regression on MNIST.
    
    Compares Adam, SGD+Nesterov, and AdaGrad with L2 regularization
    and 1/sqrt(t) learning rate decay as specified in the paper (Section 6.1).
    """
    print("\n" + "="*60)
    print("Experiment 1: Logistic Regression (MNIST) — Paper Figure 1")
    print("="*60)
    set_seed(seed)
    train_loader, _ = get_mnist_loaders()
    results = {}
    weight_decay = 1e-4  # L2 regularization as per paper Section 6.1
    
    print("Training Adam...")
    set_seed(seed)
    model = get_logistic_regression().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: 1.0 / math.sqrt(ep + 1))
    results['Adam'] = train_classifier(model, opt, train_loader, epochs=45, scheduler=sched)
    
    print("Training SGD+Nesterov...")
    set_seed(seed)
    model = get_logistic_regression().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda ep: 1.0 / math.sqrt(ep + 1))
    results['SGD+Nesterov'] = train_classifier(model, opt, train_loader, epochs=45, scheduler=sched)
    
    print("Training AdaGrad...")
    set_seed(seed)
    model = get_logistic_regression().to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=0.01, weight_decay=weight_decay)
    results['AdaGrad'] = train_classifier(model, opt, train_loader, epochs=45)
    
    plot_loss_curves(results, 'Logistic Regression (MNIST)', 'fig1_logreg.png',
                     caption='Reproduction of Figure 1 (left) in Kingma & Ba (2015)')


def experiment_mlp(seed=42):
    """Reproduce Figure 2(a): MLP with Dropout on MNIST.
    
    Compares Adam, RMSProp, AdaGrad, SGD+Nesterov, and AdaDelta,
    all with 50% dropout regularization.
    """
    print("\n" + "="*60)
    print("Experiment 2: MLP with Dropout (MNIST) — Paper Figure 2(a)")
    print("="*60)
    set_seed(seed)
    train_loader, _ = get_mnist_loaders()
    results = {}
    
    print("Training Adam+Dropout...")
    set_seed(seed)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results['Adam'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training RMSProp+Dropout...")
    set_seed(seed)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
    results['RMSProp'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training AdaGrad+Dropout...")
    set_seed(seed)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=0.01)
    results['AdaGrad'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training SGD+Nesterov+Dropout...")
    set_seed(seed)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    results['SGD+Nesterov'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training AdaDelta+Dropout...")
    set_seed(seed)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.Adadelta(model.parameters())
    results['AdaDelta'] = train_classifier(model, opt, train_loader, epochs=45)
    
    plot_loss_curves(results, 'MLP with Dropout (MNIST)', 'fig2_mlp.png',
                     caption='Reproduction of Figure 2(a) in Kingma & Ba (2015)')


def experiment_cnn(seed=42):
    """Reproduce Figure 3: CNN on CIFAR-10.
    
    Compares Adam, RMSProp, AdaGrad, and SGD+Nesterov with dropout.
    Epochs reduced from 45 to 20 to meet the ≤4-hour wall-clock constraint
    on CPU hardware. The convergence pattern is clearly visible by epoch 20.
    """
    print("\n" + "="*60)
    print("Experiment 3: CNN (CIFAR-10) — Paper Figure 3")
    print("="*60)
    set_seed(seed)
    train_loader, _ = get_cifar10_loaders()
    results = {}
    
    epochs = 20  # Paper uses 45; reduced to meet ≤4h wall-clock constraint on CPU
    
    print("Training Adam+Dropout...")
    set_seed(seed)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results['Adam'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    print("Training RMSProp+Dropout...")
    set_seed(seed)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
    results['RMSProp'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    print("Training AdaGrad+Dropout...")
    set_seed(seed)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=0.01)
    results['AdaGrad'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    print("Training SGD+Nesterov+Dropout...")
    set_seed(seed)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    results['SGD+Nesterov'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    plot_cnn_two_panel(results, 'fig3_cnn.png',
                       caption='Reproduction of Figure 3 in Kingma & Ba (2015)')


def experiment_vae(seed=42):
    """Reproduce Figure 4: Bias-correction ablation on VAE (MNIST).
    
    Trains a VAE with and without bias correction across different
    learning rates (alpha) and beta2 values. Uses a 500-sample subset
    of MNIST to keep training time feasible (documented deviation from paper).
    """
    print("\n" + "="*60)
    print("Experiment 4: VAE Ablation (MNIST) — Paper Figure 4")
    print("="*60)
    set_seed(seed)
    full_train_loader, _ = get_mnist_loaders()
    # Use 500-sample subset for feasibility (deviation from paper — documented in report)
    subset = torch.utils.data.Subset(full_train_loader.dataset, range(500))
    train_loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True)
    
    def train_vae(opt_class, lr, beta1, beta2, bias_correction, epochs=10):
        set_seed(seed)
        model = VAE().to(device)
        opt = opt_class(model.parameters(), lr=lr, betas=(beta1, beta2), bias_correction=bias_correction)
        
        loss_val = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for data, _ in train_loader:
                data = (data > 0.5).float().to(device)
                data = data.view(data.size(0), -1)
                
                opt.zero_grad()
                recon_logits, mu, logvar = model(data)
                loss = vae_loss(recon_logits, data, mu, logvar)
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
            loss_val = epoch_loss / len(train_loader)
        return loss_val

    results_10 = {}
    results_100 = {}
    
    alphas = [1e-4, 1e-3, 1e-2]
    log_alphas = [-4, -3, -2]
    betas = [(0.9, 0.99), (0.9, 0.999), (0.9, 0.9999)]
    
    for beta1, beta2 in betas:
        label_corrected = f"beta2={beta2} (corrected)"
        label_uncorrected = f"beta2={beta2} (uncorrected)"
        
        losses_10_c, losses_10_u = [], []
        losses_100_c, losses_100_u = [], []
        
        for alpha in alphas:
            print(f"  VAE: 10ep, α={alpha}, β2={beta2}, corrected")
            losses_10_c.append(train_vae(AdamUncorrected, alpha, beta1, beta2, True, epochs=10))
            
            print(f"  VAE: 10ep, α={alpha}, β2={beta2}, uncorrected")
            losses_10_u.append(train_vae(AdamUncorrected, alpha, beta1, beta2, False, epochs=10))
            
            print(f"  VAE: 100ep, α={alpha}, β2={beta2}, corrected")
            losses_100_c.append(train_vae(AdamUncorrected, alpha, beta1, beta2, True, epochs=100))
            
            print(f"  VAE: 100ep, α={alpha}, β2={beta2}, uncorrected")
            losses_100_u.append(train_vae(AdamUncorrected, alpha, beta1, beta2, False, epochs=100))
            
        results_10[label_corrected] = losses_10_c
        results_10[label_uncorrected] = losses_10_u
        results_100[label_corrected] = losses_100_c
        results_100[label_uncorrected] = losses_100_u
        
    plot_vae_ablation(results_10, results_100, log_alphas, 'fig4_vae_ablation.png',
                      caption='Reproduction of Figure 4 in Kingma & Ba (2015)')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adam Paper Reproduction — Kingma & Ba (2015)')
    parser.add_argument('--logreg', action='store_true', help='Run logistic regression experiment (Figure 1)')
    parser.add_argument('--mlp', action='store_true', help='Run MLP experiment (Figure 2)')
    parser.add_argument('--cnn', action='store_true', help='Run CNN experiment (Figure 3)')
    parser.add_argument('--vae', action='store_true', help='Run VAE ablation experiment (Figure 4)')
    parser.add_argument('--all', action='store_true', help='Run all experiments')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    args = parser.parse_args()
    
    total_start = time.time()
    
    if args.logreg or args.all:
        t0 = time.time()
        experiment_logistic_regression(seed=args.seed)
        print(f"\n  → Logistic Regression completed in {time.time()-t0:.1f}s")
    if args.mlp or args.all:
        t0 = time.time()
        experiment_mlp(seed=args.seed)
        print(f"\n  → MLP completed in {time.time()-t0:.1f}s")
    if args.cnn or args.all:
        t0 = time.time()
        experiment_cnn(seed=args.seed)
        print(f"\n  → CNN completed in {time.time()-t0:.1f}s")
    if args.vae or args.all:
        t0 = time.time()
        experiment_vae(seed=args.seed)
        print(f"\n  → VAE Ablation completed in {time.time()-t0:.1f}s")
    
    print(f"\n{'='*60}")
    print(f"Total wall-clock time: {time.time()-total_start:.1f}s ({(time.time()-total_start)/60:.1f} min)")
    print(f"{'='*60}")
