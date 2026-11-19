"""Analytic independent-point Gaussian reference plus whole-set neural corrections."""
import torch
from torch import nn
from torch.nn import functional as F
from privacy_synth.spatial_set_diffusion import Schedule, SetDenoiser


class GaussianReference(nn.Module):
    def __init__(self, parameters, prior, dtype=torch.float32):
        super().__init__()
        mean = torch.tensor([p['mean'] for p in parameters], dtype=torch.float64)
        cov = torch.tensor([p['covariance'] for p in parameters], dtype=torch.float64)
        prior = torch.tensor(prior, dtype=torch.float64)
        if mean.shape != (len(prior), 3) or cov.shape != (len(prior), 3, 3):
            raise ValueError('Registered 3D Gaussian components required')
        if not torch.isfinite(mean).all() or not torch.isfinite(cov).all() or not torch.isfinite(prior).all() or not (prior > 0).all():
            raise ValueError('Finite supported reference, no fallback')
        if not torch.allclose(cov, cov.transpose(-1, -2), rtol=0, atol=1e-12):
            raise ValueError('Symmetric covariance required')
        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        if not (eigenvalues > 0).all():
            raise ValueError('Positive covariance, no clipping')
        self.classes = len(prior)
        self.register_buffer('mean', mean.to(dtype))
        self.register_buffer('eigenvalues', eigenvalues.to(dtype))
        self.register_buffer('eigenvectors', eigenvectors.to(dtype))
        self.register_buffer('log_prior', (prior/prior.sum()).log().to(dtype))
        self.register_buffer('bar', Schedule().bar.to(dtype))

    @torch.no_grad()
    def forward(self, noisy, observed, t):
        bar = self.bar[t]
        a = bar.sqrt()[:, None, None]
        b = (1-bar).sqrt()[:, None, None]
        denominator = bar[:, None, None]*self.eigenvalues[None]+(1-bar)[:, None, None]
        delta = noisy[:, :, None, :]-a[:, :, :, None]*self.mean[None, None]
        projected = torch.einsum('bnkd,kde->bnke', delta, self.eigenvectors)
        quadratic = (projected.square()/denominator[:, None]).sum(-1)
        logits = self.log_prior[None, None]-.5*(quadratic+denominator.log().sum(-1)[:, None])
        weights = logits.softmax(-1)
        known = observed < self.classes
        hard = F.one_hot(observed.clamp_max(self.classes-1), self.classes).to(noisy.dtype)
        weights = torch.where(known[:, :, None], hard, weights)
        gain = a[:, 0, 0][:, None, None]*self.eigenvalues[None]/denominator
        conditional_mean = self.mean[None, None]+torch.einsum('bnke,kde->bnkd', gain[:, None]*projected, self.eigenvectors)
        mean0 = (weights[:, :, :, None]*conditional_mean).sum(2)
        velocity = (a*noisy-mean0)/b
        return velocity, logits


class ReferenceSetDenoiser(nn.Module):
    """Same trainable backbone; zero heads initialize exactly at the reference."""
    def __init__(self, parameters, prior):
        super().__init__()
        self.reference = GaussianReference(parameters, prior)
        self.classes = self.reference.classes
        self.network = SetDenoiser(self.classes)
        for layer in [self.network.velocity, self.network.labels]:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, noisy, observed, valid, condition, t):
        velocity, logits = self.reference(noisy, observed, t)
        residual_v, residual_logits = self.network(noisy, observed, valid, condition, t)
        return velocity+residual_v, logits+residual_logits


class PriorOnlyDenoiser(nn.Module):
    def __init__(self, parameters, prior):
        super().__init__()
        self.reference = GaussianReference(parameters, prior)
        self.classes = self.reference.classes

    def forward(self, noisy, observed, valid, condition, t):
        return self.reference(noisy, observed, t)
