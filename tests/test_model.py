"""Unit tests for TabularVAE architecture and scoring."""

import torch
import pytest
from appguard.model import TabularVAE


def test_vae_forward_and_shapes():
    batch_size = 16
    input_dim = 30
    latent_dim = 6
    vae = TabularVAE(input_dim=input_dim, hidden_dims=[32, 16], latent_dim=latent_dim)

    x = torch.randn(batch_size, input_dim)
    x_recon, mu, logvar = vae(x)

    assert x_recon.shape == (batch_size, input_dim)
    assert mu.shape == (batch_size, latent_dim)
    assert logvar.shape == (batch_size, latent_dim)


def test_vae_loss_computation():
    input_dim = 25
    vae = TabularVAE(input_dim=input_dim, hidden_dims=[32, 16], latent_dim=6, beta=0.01)

    x = torch.randn(8, input_dim)
    x_recon, mu, logvar = vae(x)
    total_loss, recon_loss, kl_div = vae.compute_loss(x, x_recon, mu, logvar)

    assert total_loss.item() > 0
    assert recon_loss.item() >= 0
    assert kl_div.item() >= 0
    assert torch.isfinite(total_loss)


def test_vae_score_samples_deterministic():
    input_dim = 20
    vae = TabularVAE(input_dim=input_dim, hidden_dims=[32, 16], latent_dim=4)
    vae.eval()

    x = torch.randn(5, input_dim)
    scores1 = vae.score_samples(x)
    scores2 = vae.score_samples(x)

    assert scores1.shape == (5,)
    assert torch.all(scores1 >= 0)
    # Since eval uses deterministic latent mean, scores must be identical across calls
    assert torch.allclose(scores1, scores2)
