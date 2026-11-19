"""Bidirectional IPD decisions and source-split loss, independent of a server.

Paper: https://arxiv.org/html/2609.36546v1#S3.SS2 (equations 9--11).
Torch is imported only by the tensor-loss entry point.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IPDDecision:
    token_id: int
    source: str
    next_driver: str
    event: str


def decide_token(driver, student_token, teacher_token, student_topk, teacher_topk):
    if driver not in {'S', 'T'}:
        raise ValueError('driver must be S or T')
    if not student_topk or not teacher_topk:
        raise ValueError('both acceptance supports must be nonempty')
    if driver == 'S':
        if student_token is None:
            raise ValueError('student-driving requires a student proposal')
        if student_token in teacher_topk:
            return IPDDecision(student_token, 'S', 'S', 'student_accepted')
        if teacher_token is None:
            raise ValueError('rejected student proposal requires teacher replacement')
        return IPDDecision(teacher_token, 'T', 'T', 'takeover')
    if teacher_token is None:
        raise ValueError('teacher-driving requires a teacher proposal')
    if teacher_token in student_topk:
        return IPDDecision(teacher_token, 'T', 'S', 'handback')
    return IPDDecision(teacher_token, 'T', 'T', 'teacher_continues')


def source_split_kl(student_logits, teacher_logits, mask, student_source):
    """Mean source-split KL on one explicit common support per position.

    Both tensors must score exactly the same support IDs. Actual teacher
    logits or native log probabilities are valid: common-support log_softmax
    removes the per-position normalization constant. A union support needs
    actual scores for student-only IDs; teacher-only support does not include
    those IDs. Choosing a support is a protocol decision outside this function.
    Padding is never support, and missing feedback cannot be filled in.
    """
    import torch

    if student_logits.shape != teacher_logits.shape or mask.shape != student_logits.shape:
        raise ValueError('student/teacher/support shapes differ')
    if student_logits.ndim != 2 or student_source.shape != (student_logits.shape[0],):
        raise ValueError('one source flag is required for each token')
    if mask.dtype != torch.bool or student_source.dtype != torch.bool:
        raise ValueError('support and source flags must be boolean')
    if not bool(mask.any(dim=-1).all()):
        raise ValueError('a support row is empty')
    if not bool(torch.isfinite(student_logits[mask]).all()) or not bool(torch.isfinite(teacher_logits[mask]).all()):
        raise ValueError('every supported token requires finite actual scores')
    live = student_logits.float() if student_logits.dtype != torch.float64 else student_logits
    frozen = teacher_logits.detach().to(dtype=live.dtype)
    log_p = live.masked_fill(~mask, float('-inf')).log_softmax(-1).masked_fill(~mask, 0)
    log_q = frozen.masked_fill(~mask, float('-inf')).log_softmax(-1).masked_fill(~mask, 0)
    p = log_p.exp() * mask
    q = log_q.exp() * mask
    reverse = (p * (log_p - log_q)).sum(-1)
    forward = (q * (log_q - log_p)).sum(-1)
    per_token = torch.where(student_source, reverse, forward)
    return per_token.mean(), per_token, reverse, forward
