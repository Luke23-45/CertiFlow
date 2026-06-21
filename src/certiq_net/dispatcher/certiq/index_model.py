"""CertiQ index model for marginal-cost dispatch with PPO-Lagrangian constraint."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from certiq_net.dispatcher.certiq.certificate import (
    DifferentiableKLProjection,
)
from certiq_net.dispatcher.certiq.interaction import (
    DispatchInteractionEncoder,
    index_token_features,
)
from certiq_net.dispatcher.delay_geometry import quadratic_drift_index, sed_index
from certiq_net.dispatcher.types import DispatcherDiagnostics, DispatcherForward

ALLOWED_COST_FNS = {"sed", "qmd"}


def expand_mu(Q: Tensor, mu: Tensor) -> Tensor:
    if mu.dim() == 1:
        return mu.unsqueeze(0).expand(Q.shape[0], -1)
    return mu


class MarginalIndexHead(nn.Module):
    """Learned marginal cost index I_i(Q, mu) per resource.

    Produces raw logits (not wrapped in softmax).  The output is a
    pure learned per-resource score â€” no analytic baseline is baked in.
    """

    def __init__(
        self,
        N: int,
        hidden_dim: int = 64,
        d_xi: int = 0,
        *,
        encoder_layers: int = 2,
        num_heads: int = 4,
        num_inducing_points: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.N = N
        self.d_xi = int(d_xi)
        self.encoder = DispatchInteractionEncoder(
            feature_dim=6 + self.d_xi,
            d_model=hidden_dim,
            d_global=hidden_dim,
            num_layers=encoder_layers,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
            global_feature_dim=8,
        )
        self.index_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, Q: Tensor, mu: Tensor, xi: Tensor | None = None) -> tuple[Tensor, Tensor]:
        token_features, global_features = index_token_features(Q, mu, xi, d_xi=self.d_xi)
        z_local, z_global = self.encoder(token_features, global_features)
        z_global_expanded = z_global.unsqueeze(1).expand(-1, Q.shape[1], -1)
        z_combined = torch.cat([z_local, z_global_expanded], dim=-1)
        logits = self.index_head(z_combined).squeeze(-1)
        value = self.value_head(z_global).squeeze(-1)
        return logits, value


class CertiQIndexModel(nn.Module):
    """Dispatch by learned marginal cost index with PPO-Lagrangian constraint.

    The forward pass:

        1. Compute a per-resource **cost** (delay proxy, default SED).
        2. Compute the certificate envelope ``budget = min_i cost_i + C``.
        3. Produce raw **logits** from the learned encoder + index head.
        4. Output ``pi = softmax(logits / tau)`` â€” the policy is a plain
           softmax.  The budget constraint is enforced by the Lagrangian
           loss during training, not by a hard projection at forward time.

    Parameters
    ----------
    constraint_mode : str
        One of ``"lagrangian"`` (default, constraint via loss),
        ``"projection"`` (hard KL projection â€” requires re-adding
        projection diagnostics), or ``"unconstrained"`` (plain softmax
        with no constraint mechanism).
    """

    def __init__(
        self,
        N: int,
        hidden_dim: int = 64,
        tau: float = 1.0,
        exploration_temperature: float = 1.5,
        C: float = 2.0,
        cost_fn: str = "qmd",
        d_xi: int = 0,
        encoder_layers: int = 2,
        num_heads: int = 4,
        num_inducing_points: int = 4,
        dropout: float = 0.0,
        constraint_mode: str = "lagrangian",
    ) -> None:
        super().__init__()
        self.N = N
        self.tau = tau
        self.exploration_temperature = float(exploration_temperature)
        self.C = C
        self.cost_fn = cost_fn
        self.d_xi = int(d_xi)
        self.constraint_mode = constraint_mode
        if self.cost_fn not in ALLOWED_COST_FNS:
            raise ValueError(
                f"Unknown cost_fn: {self.cost_fn}. "
                f"Supported cost functions: {sorted(ALLOWED_COST_FNS)}"
            )
        self.index_head = MarginalIndexHead(
            N,
            hidden_dim=hidden_dim,
            d_xi=self.d_xi,
            encoder_layers=encoder_layers,
            num_heads=num_heads,
            num_inducing_points=num_inducing_points,
            dropout=dropout,
        )

    def reset_dispatch_state(self) -> None:
        """Stateless model; kept for API compatibility."""

    def forward_full(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        training_mode: bool = False,
    ) -> DispatcherForward:
        batch, n = Q.shape
        mu_b = expand_mu(Q, mu)

        if self.cost_fn == "sed":
            cost = sed_index(Q, mu_b)
        elif self.cost_fn == "qmd":
            cost = quadratic_drift_index(Q, mu_b)
        else:
            raise ValueError(f"Unknown cost_fn: {self.cost_fn}")
        cost_min = cost.min(dim=-1).values
        budget = cost_min + self.C

        learned_logits, value = self.index_head(Q, mu_b, xi)
        effective_tau = self.tau * (
            self.exploration_temperature if training_mode else 1.0
        )
        proposal_logits = -learned_logits / effective_tau
        # Clamp logits to prevent overflow in softmax,
        # especially when Q values are large (QGym data).
        proposal_logits = proposal_logits.clamp(min=-20, max=20)

        if self.constraint_mode == "projection":
            p_cert, nu, solver_status = DifferentiableKLProjection.apply(
                proposal_logits, cost, budget
            )
            pi = p_cert
            a_cert = (pi * cost).sum(dim=-1)
            slack = budget - a_cert
        else:
            # Lagrangian and unconstrained modes both use plain softmax;
            # the difference is only in the training loss.
            pi = torch.softmax(proposal_logits, dim=-1)
            p_cert = None
            nu = None
            solver_status = None
            a_cert = None
            slack = None
        # Final safety net: replace any NaN/Inf with small uniform noise
        if torch.isnan(pi).any() or torch.isinf(pi).any():
            pi = torch.nan_to_num(pi, nan=1e-8, posinf=1e-8, neginf=1e-8)
            pi = pi / pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Diagnostics depend on constraint mode
        if self.constraint_mode == "projection":
            a_proposal = (torch.softmax(proposal_logits, dim=-1) * cost).sum(dim=-1)
            a_final = a_cert
            constraint_violation_val = torch.zeros_like(budget)
            slack_val = slack
            p_cert_out = p_cert
        else:
            a_proposal = (pi * cost).sum(dim=-1)
            a_final = a_proposal
            constraint_violation_val = (a_final - budget).clamp(min=0.0)
            slack_val = budget - a_final
            p_cert_out = pi

        ent = -(pi * pi.clamp_min(1e-9).log()).sum(dim=-1)

        diag = DispatcherDiagnostics(
            A_proposal=a_proposal,
            A_final=a_final,
            m_Q=cost_min,
            B_Q=budget,
            certificate_slack=slack_val,
            constraint_violation=constraint_violation_val,
            usage_raw=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            usage_final=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            usage_cap=torch.ones(batch, device=Q.device, dtype=Q.dtype),
            policy_entropy=ent,
            selected_resource=pi.argmax(dim=-1),
            pressure_mean=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_max=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            pressure_update_norm=torch.zeros(batch, device=Q.device, dtype=Q.dtype),
            A_cert=a_cert,
            solver_status=solver_status,
            nu=nu,
        )
        return DispatcherForward(
            pi=pi,
            diagnostics=diag,
            value=value,
            p_cert=p_cert_out,
            p_proposal=pi,
            usage_raw=diag.usage_raw,
            usage_final=diag.usage_final,
            proposal_logits=proposal_logits,
            index_values=-proposal_logits * effective_tau,
        )

    def forward(
        self,
        Q: Tensor,
        mu: Tensor,
        xi: Tensor | None = None,
        *,
        training_mode: bool = False,
    ) -> tuple[Tensor, DispatcherDiagnostics]:
        out = self.forward_full(Q, mu, xi, training_mode=training_mode)
        return out.pi, out.diagnostics
