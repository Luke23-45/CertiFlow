"""Delay-aligned geometry operators for CertiQ dispatch.

Provides SED and QMD index functions, corresponding hard/soft policies,
and the delay-aligned certificate envelope used by KL-projection certification.
"""

import torch
from torch import Tensor


def sed_index(Q: Tensor, mu: Tensor) -> Tensor:
    """Shortest Expected Delay index ``(Q_i + 1) / mu_i``."""
    return (Q + 1.0) / mu.clamp_min(torch.finfo(mu.dtype).tiny)


def quadratic_drift_index(Q: Tensor, mu: Tensor) -> Tensor:
    """Quadratic Lyapunov drift index ``(2 * Q_i + 1) / mu_i``."""
    return (2.0 * Q + 1.0) / mu.clamp_min(torch.finfo(mu.dtype).tiny)


def sed_hard_policy(Q: Tensor, mu: Tensor) -> Tensor:
    """Return a one-hot hard policy selecting argmin SED index."""
    idx = sed_index(Q, mu).argmin(dim=-1)
    pi = torch.zeros_like(Q)
    pi.scatter_(1, idx.unsqueeze(-1), 1.0)
    return pi


def sed_soft_policy(Q: Tensor, mu: Tensor, tau: float = 1.0) -> Tensor:
    """Return a softmax policy over the SED index with temperature ``tau``."""
    logits = -sed_index(Q, mu) / tau
    pi = torch.softmax(logits, dim=-1)
    eps = torch.finfo(pi.dtype).tiny
    pi = pi.clamp_min(eps)
    return pi / pi.sum(dim=-1, keepdim=True)


def delay_arrival_coordinate(pi: Tensor, d: Tensor) -> Tensor:
    """Return ``sum_i pi_i * d_i``."""
    return (pi * d).sum(dim=-1)


def delay_envelope(m_d: Tensor, C_d: Tensor) -> Tensor:
    """Return the delay envelope ``m_d + C_d``."""
    return m_d + C_d
