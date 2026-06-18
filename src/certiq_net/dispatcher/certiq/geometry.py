"""Certified base geometry for z3 dispatch."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def inv_softplus(x: float) -> float:
    """Return the scalar inverse softplus for strictly positive ``x``."""
    if x <= 0:
        raise ValueError("Softplus inverse input must be strictly positive.")
    return math.log(math.expm1(x))


class CertifiedGeometry(nn.Module):
    """Capacity-aware energy geometry with constrained scalar parameters."""

    def __init__(
        self,
        *,
        alpha_min: float,
        beta_min: float,
        gamma_max: float,
        alpha_init: float,
        beta_init: float,
        gamma_init: float,
        c_init: float,
        C: float,
    ) -> None:
        super().__init__()
        assert alpha_min > 0 and beta_min > 0, "Geometry lower bounds must be positive."
        assert gamma_max > 0, "gamma_max must be positive."
        assert C >= 0, "Certificate constant must be non-negative."
        self.alpha_min = float(alpha_min)
        self.beta_min = float(beta_min)
        self.gamma_max = float(gamma_max)
        self.C = float(C)
        self.raw_alpha = nn.Parameter(torch.tensor(inv_softplus(alpha_init - alpha_min)))
        self.raw_beta = nn.Parameter(torch.tensor(inv_softplus(beta_init - beta_min)))
        self.raw_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.raw_c = nn.Parameter(torch.tensor(inv_softplus(max(float(c_init), 0.0) + 1e-6)))

    @property
    def alpha(self) -> Tensor:
        return self.alpha_min + F.softplus(self.raw_alpha)

    @property
    def beta(self) -> Tensor:
        return self.beta_min + F.softplus(self.raw_beta)

    @property
    def gamma(self) -> Tensor:
        return self.gamma_max * torch.tanh(self.raw_gamma)

    @property
    def c(self) -> Tensor:
        return F.softplus(self.raw_c)

    def _scalar_params(self, reference: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return scalar parameters on the same device/dtype as ``reference``."""
        alpha = self.alpha.to(device=reference.device, dtype=reference.dtype)
        beta = self.beta.to(device=reference.device, dtype=reference.dtype)
        gamma = self.gamma.to(device=reference.device, dtype=reference.dtype)
        c = self.c.to(device=reference.device, dtype=reference.dtype)
        c_constant = torch.as_tensor(self.C, device=reference.device, dtype=reference.dtype)
        return alpha, beta, gamma, c, c_constant

    def weighted_workload(self, Q: Tensor, mu: Tensor) -> Tensor:
        """Return ``Q_i / mu_i^beta``."""
        assert Q.shape == mu.shape, "Q and mu must have identical shape."
        assert (Q >= 0).all(), "Queue lengths must be non-negative."
        assert (mu > 0).all(), "Service rates must be positive."
        _, beta, _, _, _ = self._scalar_params(Q)
        return Q / mu.pow(beta).clamp_min(torch.finfo(mu.dtype).tiny)

    def logits(self, Q: Tensor, mu: Tensor) -> Tensor:
        """Return certified base logits."""
        alpha, beta, gamma, c, _ = self._scalar_params(Q)
        y_shift = (Q + c) / mu.pow(beta).clamp_min(torch.finfo(mu.dtype).tiny)
        return gamma * torch.log(mu.clamp_min(torch.finfo(mu.dtype).tiny)) - alpha * y_shift

    def policy(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(p_cert, u_cert)``."""
        u = self.logits(Q, mu)
        p = torch.softmax(u, dim=-1)
        p = p.clamp_min(torch.finfo(p.dtype).tiny)
        return p / p.sum(dim=-1, keepdim=True), u

    def envelope(self, Q: Tensor, mu: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(m(Q), B(Q))``."""
        y = self.weighted_workload(Q, mu)
        m_q = y.min(dim=-1).values
        _, _, _, _, c_constant = self._scalar_params(Q)
        return m_q, m_q + c_constant
