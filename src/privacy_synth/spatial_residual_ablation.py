"""Inference-only branch removal; never a claim of training-equivalent replacement."""
import torch
from torch import nn


MODES = ('full', 'no-v-residual', 'no-class-residual')


class FrozenBranchDenoiser(nn.Module):
    def __init__(self, model, mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('Explicit frozen branch mode required')
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise ValueError('Only frozen eval models may enter branch diagnosis')
        self.model = model
        self.classes = model.classes
        self.mode = mode

    @torch.no_grad()
    def forward(self, noisy, observed, valid, condition, t):
        prior_v, prior_logits = self.model.reference(noisy, observed, t)
        residual_v, residual_logits = self.model.network(noisy, observed, valid, condition, t)
        velocity = prior_v if self.mode == 'no-v-residual' else prior_v+residual_v
        logits = prior_logits if self.mode == 'no-class-residual' else prior_logits+residual_logits
        return velocity, logits
