"""Pure GAD tensor objectives; no model loading, data access or network.

GAD arXiv:2511.10643v3 equations 3, 6, 8. Torch imports are lazy.
"""
from __future__ import annotations


def bradley_terry(teacher, student):
    import torch
    if teacher.numel() not in (1, student.numel()) or not student.numel():
        raise ValueError('One shared or one paired teacher score is required')
    return -torch.nn.functional.logsigmoid(teacher - student).mean()


def last_nonpadding(hidden, attention_mask):
    import torch
    if hidden.shape[:2] != attention_mask.shape or not bool(attention_mask.any(-1).all()):
        raise ValueError('Invalid or empty sequence mask')
    positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
    last = positions.masked_fill(~attention_mask.bool(), -1).max(-1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), last]


def group_advantages(rewards, epsilon=1e-6):
    import torch
    if rewards.ndim != 1 or rewards.numel() < 2:
        raise ValueError('A response group needs at least two rewards')
    if epsilon<=0 or not bool(torch.isfinite(rewards).all()):
        raise ValueError('Rewards must be finite and epsilon positive')
    frozen = rewards.detach()
    return (frozen - frozen.mean()) / (frozen.std(correction=1) + epsilon)


def clipped_grpo(log_probs, old_log_probs, reference_log_probs, advantage,
                 clip=0.2, beta=0.001):
    """One response's token mean; caller averages equally over responses.

    Frozen old/reference/advantage tensors cannot receive gradients. Uses the
    nonnegative k3 KL estimator exp(ref-new) - (ref-new) - 1. No padding tokens
    may be passed. Sampling/log probabilities must use the same temperature.
    """
    import torch
    if log_probs.ndim != 1 or not log_probs.numel():
        raise ValueError('Expected actual response token log probabilities')
    if log_probs.shape != old_log_probs.shape or log_probs.shape != reference_log_probs.shape:
        raise ValueError('Log probability shapes differ')
    if not 0 < clip < 1 or beta < 0 or advantage.numel()!=1:
        raise ValueError('Invalid GRPO hyperparameters or non-scalar advantage')
    old, ref, adv = old_log_probs.detach(), reference_log_probs.detach(), advantage.detach()
    ratio = (log_probs - old).exp()
    surrogate = torch.minimum(ratio * adv, ratio.clamp(1-clip, 1+clip) * adv)
    difference = ref - log_probs
    kl = difference.exp() - difference - 1
    loss = (-surrogate + beta * kl).mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError('Nonfinite GRPO loss')
    return loss


def response_log_probs(hidden, output_weight, input_ids, prompt_length,
                       temperature=0.8, chunk_tokens=32):
    """Exact full-vocabulary response log-probs with checkpointed projection.

    Keeps the complete prompt and actual generated EOS if present; never adds
    EOS to a truncated response. Hidden states contain one unpadded sequence.
    """
    import torch
    from torch.utils.checkpoint import checkpoint
    if hidden.shape[0] != 1 or not 0 < prompt_length < input_ids.shape[1]:
        raise ValueError('Expected one full prompt plus nonempty response')
    if temperature<=0 or chunk_tokens<1 or hidden.shape[:2]!=input_ids.shape:
        raise ValueError('Invalid token/projection shape, temperature or chunk size')
    values = []
    def project(h, weight, targets):
        logits = torch.nn.functional.linear(h, weight).float() / temperature
        return logits.gather(-1, targets[:, None]).squeeze(-1) - logits.logsumexp(-1)
    for begin in range(prompt_length-1, input_ids.shape[1]-1, chunk_tokens):
        end = min(begin+chunk_tokens, input_ids.shape[1]-1)
        h = hidden[0, begin:end]
        targets = input_ids[0, begin+1:end+1]
        if torch.is_grad_enabled() and h.requires_grad:
            value = checkpoint(project, h, output_weight, targets, use_reentrant=False)
        else:
            value = project(h, output_weight, targets)
        values.append(value)
    return torch.cat(values)
