"""Variational Auto-Encoder for bias-correction ablation study (Figure 4 reproduction)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Auto-Encoder with Gaussian latent space.

    Architecture: single hidden layer (softplus) encoder/decoder with
    500 hidden units and 50 latent dimensions, following the paper's
    specification for the bias-correction ablation experiment (Figure 4).
    """

    def __init__(self, input_dim=784, hidden_dim=500, latent_dim=50):
        super(VAE, self).__init__()
        
        # Encoder
        self.enc_h = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.dec_h = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.softplus(self.enc_h(x))
        return self.enc_mu(h), self.enc_logvar(h)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
            
    def decode(self, z):
        h = F.softplus(self.dec_h(z))
        return self.dec_out(h)
        
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        return recon_logits, mu, logvar


def vae_loss(recon_logits, x, mu, logvar):
    """Compute the VAE loss: reconstruction (BCE) + KL divergence.

    Args:
        recon_logits: Decoder output logits (before sigmoid).
        x: Original binarized input images (flattened).
        mu: Encoder mean of the latent distribution.
        logvar: Encoder log-variance of the latent distribution.

    Returns:
        Per-sample average loss (BCE + KL).
    """
    # BCE loss
    bce = F.binary_cross_entropy_with_logits(recon_logits, x, reduction='sum')
    
    # KL Divergence
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return (bce + kl) / x.size(0)
