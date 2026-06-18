"""Typed public objects for the z3 CertiQ Dispatcher."""

from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class DispatcherDiagnostics:
    """Diagnostics for one forward pass (constraint-violation fields are
    for logging / monitoring only — the Lagrangian dual variable is
    managed by the training module, not produced by the model)."""

    A_proposal: Tensor
    A_final: Tensor
    m_Q: Tensor
    B_Q: Tensor
    certificate_slack: Tensor
    constraint_violation: Tensor
    usage_raw: Tensor
    usage_final: Tensor
    usage_cap: Tensor
    policy_entropy: Tensor
    selected_resource: Tensor
    pressure_mean: Tensor
    pressure_max: Tensor
    pressure_update_norm: Tensor
    A_cert: Tensor | None = None
    solver_status: Tensor | None = None
    nu: Tensor | None = None


@dataclass(frozen=True)
class DispatcherForward:
    """Full dispatcher output used by training and evaluation."""

    pi: Tensor
    diagnostics: DispatcherDiagnostics
    value: Tensor
    p_cert: Tensor
    p_proposal: Tensor
    usage_raw: Tensor
    usage_final: Tensor
    proposal_logits: Tensor
    index_values: Tensor | None = None
