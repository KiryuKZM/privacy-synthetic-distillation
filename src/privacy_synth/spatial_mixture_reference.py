"""Chunked analytic posterior for the fixed class-conditioned moment mixture."""
import math
import torch
from torch import nn
from privacy_synth.spatial_set_diffusion import Schedule


class GaussianMixtureReference(nn.Module):
    def __init__(self, parameters, prior, query_chunk=64, dtype=torch.float32):
        super().__init__()
        if not isinstance(query_chunk, int) or isinstance(query_chunk, bool) or not 1 <= query_chunk <= 64:
            raise ValueError('Explicit query chunk in1..64 required')
        probability = torch.tensor(prior, dtype=torch.float64)
        if len(parameters) != len(prior) or not torch.isfinite(probability).all() or not (probability > 0).all():
            raise ValueError('Positive supported class prior required')
        probability /= probability.sum()
        centers, classes, covariance, logweights = [], [], [], []
        for k, p in enumerate(parameters):
            source = torch.tensor(p['centers'], dtype=torch.float64)
            mu = torch.tensor(p['mean'], dtype=torch.float64)
            beta = p['beta']
            cov = torch.tensor(p['noise_covariance'], dtype=torch.float64)
            if source.shape != (p['n'], 3) or mu.shape != (3,) or p['n'] < 1 or not math.isfinite(beta) or not 0 <= beta <= 1:
                raise ValueError('Full registered mixture components required')
            if not torch.isfinite(source).all() or not torch.isfinite(mu).all() or cov.shape != (3, 3) or not torch.isfinite(cov).all() or not torch.allclose(cov, cov.T, rtol=0, atol=1e-12):
                raise ValueError('Finite symmetric mixture definition required')
            centers.append(mu+beta*(source-mu))
            classes.extend([k]*p['n'])
            covariance.append(cov)
            logweights.extend([float(probability[k].log())-math.log(p['n'])]*p['n'])
        eig, U = torch.linalg.eigh(torch.stack(covariance))
        if not (eig > 0).all():
            raise ValueError('Positive component covariance, no clipping/fallback')
        component_class = torch.tensor(classes, dtype=torch.long)
        self.classes = len(prior)
        self.query_chunk = query_chunk
        self.register_buffer('centers', torch.cat(centers).to(dtype))
        self.register_buffer('component_class', component_class)
        self.register_buffer('eigenvalues', eig[component_class].to(dtype))
        self.register_buffer('eigenvectors', U[component_class].to(dtype))
        self.register_buffer('log_component_prior', torch.tensor(logweights, dtype=dtype))
        self.register_buffer('bar', Schedule().bar.to(dtype))

    @torch.no_grad()
    def _chunk(self, noisy, observed, bar):
        a, b = bar.sqrt(), (1-bar).sqrt()
        denominator = bar[:, None, None]*self.eigenvalues[None]+(1-bar)[:, None, None]
        delta = noisy[:, None]-a[:, None, None]*self.centers[None]
        projected = torch.einsum('qmd,mde->qme', delta, self.eigenvectors)
        logw = self.log_component_prior[None]-.5*((projected.square()/denominator).sum(-1)+denominator.log().sum(-1))
        index = self.component_class[None].expand(len(noisy), -1)
        maximum = torch.full((len(noisy), self.classes), -torch.inf, device=noisy.device, dtype=noisy.dtype)
        maximum.scatter_reduce_(1, index, logw, reduce='amax', include_self=True)
        mass = torch.zeros_like(maximum)
        mass.scatter_add_(1, index, (logw-maximum.gather(1, index)).exp())
        logits = maximum+mass.log()
        allowed = (observed[:, None] == self.classes) | (observed[:, None] == index)
        weight = torch.where(allowed, logw, -torch.inf).softmax(-1)
        gain = a[:, None, None]*self.eigenvalues[None]/denominator
        mean = self.centers[None]+torch.einsum('qme,mde->qmd', gain*projected, self.eigenvectors)
        mean0 = (weight[:, :, None]*mean).sum(1)
        return (a[:, None]*noisy-mean0)/b[:, None], logits

    @torch.no_grad()
    def forward(self, noisy, observed, t):
        if noisy.ndim != 3 or noisy.shape[-1] != 3 or observed.shape != noisy.shape[:2] or len(t) != len(noisy):
            raise ValueError('Padded BxNx3 queries with per-room time required')
        if bool(((observed < 0) | (observed > self.classes)).any()) or bool(((t < 1) | (t > 64)).any()):
            raise ValueError('Only registered clean/MASK categories andtrainedtime1..64')
        flat = noisy.reshape(-1, 3)
        labels = observed.reshape(-1)
        bar = self.bar[t][:, None].expand_as(observed).reshape(-1)
        velocity, logits = [], []
        for start in range(0, len(flat), self.query_chunk):
            end = start+self.query_chunk
            v, c = self._chunk(flat[start:end], labels[start:end], bar[start:end])
            velocity.append(v)
            logits.append(c)
        return torch.cat(velocity).reshape_as(noisy), torch.cat(logits).reshape(*observed.shape, self.classes)


class MixturePriorOnlyDenoiser(nn.Module):
    def __init__(self, parameters, prior, query_chunk=64):
        super().__init__()
        self.reference = GaussianMixtureReference(parameters, prior, query_chunk=query_chunk)
        self.classes = self.reference.classes

    def forward(self, noisy, observed, valid, condition, t):
        return self.reference(noisy, observed, t)
