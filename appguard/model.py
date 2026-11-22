"""PyTorch Variational Autoencoder (VAE) for insurance application anomaly detection.

Architecture:
- Encoder: Maps tabular input vectors to a low-dimensional latent Gaussian distribution (mu, logvar).
- Reparameterization Trick: Enables gradient propagation through stochastic sampling z ~ N(mu, sigma^2).
- Decoder: Reconstructs the original tabular feature representation from latent vectors.
- Reconstruction Attribution: Computes individual feature reconstruction deviations for explainability.
"""

from typing import Dict, List, Tuple, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TabularVAE(nn.Module):
    """Variational Autoencoder designed for mixed tabular data representation and reconstruction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        latent_dim: int = 8,
        dropout_rate: float = 0.05,
        beta: float = 0.01,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate
        self.beta = beta

        # 1. Build Encoder
        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim
        for h_dim in self.hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.LayerNorm(h_dim))
            encoder_layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.encoder_backbone = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_logvar = nn.Linear(prev_dim, latent_dim)

        # 2. Build Decoder
        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim
        for h_dim in reversed(self.hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.LayerNorm(h_dim))
            decoder_layers.append(nn.LeakyReLU(0.1))
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps input tensor to latent parameters mu and logvar."""
        h = self.encoder_backbone(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + std * epsilon."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # Deterministic representation during evaluation

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstructs original input from latent representation."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass returning (reconstruction, mu, logvar)."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def compute_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes total VAE loss = Mean Squared Error + beta * KL Divergence."""
        # Feature-wise MSE reconstruction loss
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        total_loss = recon_loss + (self.beta * kl_div)
        return total_loss, recon_loss, kl_div

    @torch.no_grad()
    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sample-level anomaly score as Mean Squared Reconstruction Error per sample."""
        self.eval()
        mu, _ = self.encode(x)
        x_recon = self.decode(mu)  # Deterministic reconstruction using latent mean
        # Per-sample MSE across all input dimensions
        sample_errors = torch.mean((x - x_recon) ** 2, dim=1)
        return sample_errors

    def get_config(self) -> Dict[str, Any]:
        """Returns architecture configuration dictionary."""
        return {
            "input_dim": self.input_dim,
            "hidden_dims": self.hidden_dims,
            "latent_dim": self.latent_dim,
            "dropout_rate": self.dropout_rate,
            "beta": self.beta,
        }
