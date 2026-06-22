import torch
import torch.nn as nn
import time
import argparse

from utils.data import get_mnist_loaders, get_cifar10_loaders
from utils.plot import plot_loss_curves, plot_vae_ablation
from models.logistic_regression import get_logistic_regression
from models.mlp import get_mlp
from models.cnn import get_cnn
from models.vae import VAE, vae_loss
from optimizers.adam_uncorrected import AdamUncorrected

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def train_classifier(model, optimizer, train_loader, epochs=45):
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
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {time.time()-t0:.2f}s", flush=True)
        
    return losses

def experiment_logistic_regression():
    print("--- Running Logistic Regression (MNIST) ---", flush=True)
    train_loader, _ = get_mnist_loaders()
    results = {}
    
    print("Training Adam...", flush=True)
    model = get_logistic_regression().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results['Adam'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training SGD+Nesterov...", flush=True)
    model = get_logistic_regression().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    results['SGD+Nesterov'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training AdaGrad...", flush=True)
    model = get_logistic_regression().to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=0.01)
    results['AdaGrad'] = train_classifier(model, opt, train_loader, epochs=45)
    
    plot_loss_curves(results, 'Logistic Regression (MNIST)', 'fig1_logreg.png')

def experiment_mlp():
    print("--- Running MLP (MNIST) ---", flush=True)
    train_loader, _ = get_mnist_loaders()
    results = {}
    
    print("Training Adam+Dropout...", flush=True)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results['Adam+Dropout'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training RMSProp+Dropout...", flush=True)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=0.001)
    results['RMSProp+Dropout'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training AdaGrad+Dropout...", flush=True)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=0.01)
    results['AdaGrad+Dropout'] = train_classifier(model, opt, train_loader, epochs=45)
    
    print("Training SGD+Nesterov+Dropout...", flush=True)
    model = get_mlp(dropout=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    results['SGD+Nesterov+Dropout'] = train_classifier(model, opt, train_loader, epochs=45)
    
    plot_loss_curves(results, 'MLP with Dropout (MNIST)', 'fig2_mlp.png')

def experiment_cnn():
    print("--- Running CNN (CIFAR-10) ---", flush=True)
    train_loader, _ = get_cifar10_loaders()
    results = {}
    
    epochs = 45 # With PyTorch, CNN takes seconds/minutes instead of hours!
    
    print("Training Adam+Dropout...", flush=True)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    results['Adam+Dropout'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    print("Training AdaGrad+Dropout...", flush=True)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.Adagrad(model.parameters(), lr=0.01)
    results['AdaGrad+Dropout'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    print("Training SGD+Nesterov+Dropout...", flush=True)
    model = get_cnn(dropout=True).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    results['SGD+Nesterov+Dropout'] = train_classifier(model, opt, train_loader, epochs=epochs)
    
    plot_loss_curves(results, 'CNN (CIFAR-10)', 'fig3_cnn.png')

def experiment_vae():
    print("--- Running VAE Ablation (MNIST) ---", flush=True)
    train_loader, _ = get_mnist_loaders()
    
    def train_vae(opt_class, lr, beta1, beta2, bias_correction, epochs=10):
        model = VAE().to(device)
        opt = opt_class(model.parameters(), lr=lr, betas=(beta1, beta2), bias_correction=bias_correction)
        
        loss_val = 0
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0
            for data, _ in train_loader:
                # binarize for BCE
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
            print(f"Training VAE - 10 epochs, alpha={alpha}, beta2={beta2}, corrected", flush=True)
            losses_10_c.append(train_vae(AdamUncorrected, alpha, beta1, beta2, True, epochs=10))
            
            print(f"Training VAE - 10 epochs, alpha={alpha}, beta2={beta2}, uncorrected", flush=True)
            losses_10_u.append(train_vae(AdamUncorrected, alpha, beta1, beta2, False, epochs=10))
            
            print(f"Training VAE - 100 epochs, alpha={alpha}, beta2={beta2}, corrected", flush=True)
            losses_100_c.append(train_vae(AdamUncorrected, alpha, beta1, beta2, True, epochs=100))
            
            print(f"Training VAE - 100 epochs, alpha={alpha}, beta2={beta2}, uncorrected", flush=True)
            losses_100_u.append(train_vae(AdamUncorrected, alpha, beta1, beta2, False, epochs=100))
            
        results_10[label_corrected] = losses_10_c
        results_10[label_uncorrected] = losses_10_u
        results_100[label_corrected] = losses_100_c
        results_100[label_uncorrected] = losses_100_u
        
    plot_vae_ablation(results_10, results_100, log_alphas, 'fig4_vae_ablation.png')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--logreg', action='store_true')
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--vae', action='store_true')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    
    if args.logreg or args.all:
        experiment_logistic_regression()
    if args.mlp or args.all:
        experiment_mlp()
    if args.cnn or args.all:
        experiment_cnn()
    if args.vae or args.all:
        experiment_vae()
