"""Small permutation-equivariant mixed coordinate/category set diffusion.

Counts and room dimensions are conditions, not generated quantities.  No object
IDs, object-order embeddings, clipping, spatial unit assumptions, or vendor code.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Schedule:
    def __init__(self, steps=64, device="cpu"):
        self.steps = steps
        def abar(u):
            return math.cos((u + .008) / 1.008 * math.pi / 2) ** 2
        beta = [min(1 - abar(i / steps) / abar((i - 1) / steps), .999)
                for i in range(1, steps + 1)]
        self.beta = torch.tensor([0., *beta], device=device)
        self.alpha = 1 - self.beta
        self.bar = self.alpha.cumprod(0)

    def noisy_velocity(self, clean, noise, t):
        a = self.bar[t].sqrt()[:, None, None]
        b = (1 - self.bar[t]).sqrt()[:, None, None]
        return a * clean + b * noise, a * noise - b * clean

    def posterior(self, noisy, velocity, t, noise):
        a, prev = self.bar[t], self.bar[t - 1]
        clean = a.sqrt() * noisy - (1 - a).sqrt() * velocity
        mean = (prev.sqrt() * self.beta[t] / (1 - a) * clean
                + self.alpha[t].sqrt() * (1 - prev) / (1 - a) * noisy)
        variance = self.beta[t] * (1 - prev) / (1 - a)
        return mean + variance.sqrt() * noise if t > 1 else mean


class SetDenoiser(nn.Module):
    def __init__(self, classes=35, width=96, layers=3):
        super().__init__()
        self.classes = classes
        self.category = nn.Embedding(classes + 1, width)
        self.position = nn.Linear(3, width)
        self.condition = nn.Sequential(nn.Linear(4, width), nn.SiLU(), nn.Linear(width, width))
        self.time = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, width * 2, dropout=0.,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.set_layers = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.velocity = nn.Linear(width, 3)
        self.labels = nn.Linear(width, classes)

    def forward(self, xyz, labels, valid, condition, t):
        freq = torch.exp(torch.arange(16, device=xyz.device) * (-math.log(10000.) / 15))
        phase = (t.float() / 64 * 1000)[:, None] * freq[None, :]
        time = self.time(torch.cat([phase.sin(), phase.cos()], -1))
        hidden = (self.position(xyz) + self.category(labels)
                  + (self.condition(condition) + time)[:, None, :])
        hidden = self.set_layers(hidden, src_key_padding_mask=~valid)
        hidden = self.norm(hidden)
        return self.velocity(hidden), self.labels(hidden)


def batch(rows, normalization, device):
    """Pad only; retain every object, including the largest scenes."""
    n = max(r['count'] for r in rows)
    if n == 0:
        raise ValueError('Empty-only batch requires explicit future contract')
    xyz = torch.zeros(len(rows), n, 3, device=device)
    labels = torch.zeros(len(rows), n, dtype=torch.long, device=device)
    valid = torch.zeros(len(rows), n, dtype=torch.bool, device=device)
    cond = []
    for i, row in enumerate(rows):
        k = row['count']
        if k != len(row['objects']) or k == 0:
            raise ValueError('Unexpected empty or mismatched original room, do not drop')
        xyz[i, :k] = torch.tensor([o['position_latent'] for o in row['objects']], device=device)
        labels[i, :k] = torch.tensor([o['category_id'] for o in row['objects']], device=device)
        valid[i, :k] = True
        raw = [*map(math.log, row['room_size_native']), math.log1p(k)]
        cond.append([(v-m)/s for v,m,s in zip(raw, normalization['mean'], normalization['std'])])
    return xyz, labels, valid, torch.tensor(cond, device=device)


def absorbing_reverse(labels, probabilities, t, classes, generator):
    proposed = torch.multinomial(probabilities.reshape(-1, classes), 1,
                                 generator=generator).reshape_as(labels)
    reveal = torch.rand(labels.shape, device=labels.device, generator=generator) < 1 / t
    return torch.where((labels == classes) & reveal, proposed, labels)


def denoising_loss(model, schedule, xyz, labels, valid, condition):
    size = len(xyz)
    t = torch.randint(1, schedule.steps + 1, (size,), device=xyz.device)
    noise = torch.randn_like(xyz)
    noisy, velocity = schedule.noisy_velocity(xyz, noise, t)
    masked = (torch.rand_like(labels, dtype=torch.float32) < (t / schedule.steps)[:, None]) & valid
    corrupt = torch.where(masked, model.classes, labels)
    pred, logits = model(noisy, corrupt, valid, condition, t)
    mse = ((pred - velocity).square().mean(-1) * valid).sum(-1) / valid.sum(-1)
    ce = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
    cat = (ce * masked).sum(-1) / masked.sum(-1).clamp_min(1)
    return (mse + cat).mean(), dict(v_mse=float(mse.mean().detach()),
        category_ce=float(cat.mean().detach()), zero_mask_rooms=int((masked.sum(-1) == 0).sum()),
        rooms=size, objects=int(valid.sum()), max_objects=int(valid.sum(-1).max()))


@torch.no_grad()
def sample(model, schedule, condition, count, generator):
    device = condition.device
    xyz = torch.randn((1, count, 3), device=device, generator=generator)
    labels = torch.full((1, count), model.classes, device=device, dtype=torch.long)
    valid = torch.ones_like(labels, dtype=torch.bool)
    for t in range(schedule.steps, 0, -1):
        ts = torch.tensor([t], device=device)
        velocity, logits = model(xyz, labels, valid, condition, ts)
        noise = torch.randn(xyz.shape, device=device, generator=generator)
        xyz = schedule.posterior(xyz, velocity, t, noise)
        labels = absorbing_reverse(labels, logits.softmax(-1), t, model.classes, generator)
    return xyz[0].cpu(), labels[0].cpu()
