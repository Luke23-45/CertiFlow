"""Learned per-resource cost function (Phase 3)."""

import torch
import torch.nn as nn
from torch import Tensor


class CostLearner(nn.Module):
    """Small MLP that predicts a per-resource cost from ``(Q_i, mu_i, xi_i)``.

    When used as ``cost_fn="learned"`` in :class:`CertiQIndexModel`, this
    network replaces the analytic cost (SED / QMD).  Gradients flow from the
    Lagrangian loss back through the predicted cost into the CostLearner's
    parameters, allowing it to adapt to the policy's state distribution.
    """

    def __init__(self, N: int, hidden_dim: int = 64, d_xi: int = 0) -> None:
        super().__init__()
        self.N = N
        self.d_xi = d_xi
        input_dim = 2 + d_xi
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Q: Tensor, mu: Tensor, xi: Tensor | None = None) -> Tensor:
        if mu.dim() == 1:
            mu_b = mu.unsqueeze(0).expand(Q.shape[0], -1)
        else:
            mu_b = mu
        feat = torch.stack([Q, mu_b], dim=-1)
        if xi is not None and self.d_xi > 0:
            xi_e = xi.unsqueeze(1).expand(-1, Q.shape[1], -1)
            feat = torch.cat([feat, xi_e], dim=-1)
        return self.net(feat).squeeze(-1)
