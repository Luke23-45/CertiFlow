"""Certificate operators for z3 dispatch.

Contains the differentiable KL projection layer that maps raw policy logits
to a certified distribution satisfying an expected-cost constraint, with a
correct backward pass via implicit differentiation of the KKT conditions.
"""

import torch
from torch import Tensor


def arrival_coordinate(pi: Tensor, y: Tensor) -> Tensor:
    return (pi * y).sum(dim=-1)


def _solve_kl_projection(
    q: Tensor,
    cost: Tensor,
    budget: Tensor,
    *,
    iterations: int = 48,
    tolerance: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Bisection solver for the KL projection Lagrange multiplier.

    Returns ``(p, nu, solver_status)`` where
    ``p_i = q_i * exp(-nu * c_i) / Z(nu)``.
    """
    work_dtype = torch.float64 if cost.dtype in {torch.float16, torch.bfloat16, torch.float32} else cost.dtype
    q_w = q.to(work_dtype)
    c_w = cost.to(work_dtype)
    b_w = budget.to(work_dtype)
    eps = torch.finfo(work_dtype).tiny
    c_w = c_w.clamp_min(eps)
    b_w = b_w.clamp_min(eps)

    device = q.device
    batch = q.shape[0]
    solver_status = torch.zeros(batch, device=device, dtype=torch.long)
    p = q_w.clone()
    nu = torch.zeros(batch, device=device, dtype=work_dtype)

    expected = (q_w * c_w).sum(dim=-1)
    feasible = expected <= b_w + tolerance
    if feasible.all():
        return p.to(q.dtype), nu.to(q.dtype), solver_status

    mask = ~feasible
    q_m = q_w[mask]
    c_m = c_w[mask]
    b_m = b_w[mask]
    n_infeas = mask.sum().item()
    min_cost = c_m.min(dim=-1).values

    budget_too_low = b_m < min_cost
    if budget_too_low.any():
        idx = c_m.argmin(dim=-1)
        dirac = torch.zeros_like(q_m)
        dirac.scatter_(1, idx.unsqueeze(-1), 1.0)
        q_m = torch.where(budget_too_low.unsqueeze(-1), dirac, q_m)
        b_m = torch.where(budget_too_low, min_cost, b_m)
        solver_status[mask] = torch.where(
            budget_too_low,
            torch.tensor(1, device=device, dtype=torch.long),
            solver_status[mask],
        )

    expected_m = (q_m * c_m).sum(dim=-1)
    feasible_m = expected_m <= b_m + tolerance
    if feasible_m.all():
        p[mask] = q_m
        return p.to(q.dtype), nu.to(q.dtype), solver_status

    bisect_mask = ~feasible_m
    q_b = q_m[bisect_mask]
    c_b = c_m[bisect_mask]
    b_b = b_m[bisect_mask]

    lo = torch.zeros(q_b.shape[0], device=device, dtype=work_dtype)
    hi = torch.full((q_b.shape[0],), 10.0, device=device, dtype=work_dtype)

    for _ in range(32):
        logits = q_b.clamp_min(eps).log() - hi.unsqueeze(-1) * c_b
        p_test = torch.softmax(logits, dim=-1)
        too_high = (p_test * c_b).sum(dim=-1) > b_b
        hi = torch.where(too_high, hi * 2.0, hi)
        if not too_high.any():
            break

    for _ in range(iterations):
        nu_mid = (lo + hi) / 2.0
        logits = q_b.clamp_min(eps).log() - nu_mid.unsqueeze(-1) * c_b
        p_mid = torch.softmax(logits, dim=-1)
        too_high = (p_mid * c_b).sum(dim=-1) > b_b
        lo = torch.where(too_high, nu_mid, lo)
        hi = torch.where(too_high, hi, nu_mid)

    nu_final = hi
    logits = q_b.clamp_min(eps).log() - nu_final.unsqueeze(-1) * c_b
    p_final = torch.softmax(logits, dim=-1)

    p_result = q_m.clone()
    nu_result = torch.zeros(n_infeas, device=device, dtype=work_dtype)
    p_result[bisect_mask] = p_final
    nu_result[bisect_mask] = nu_final

    p[mask] = p_result
    nu[mask] = nu_result

    expected_final = (p * c_w).sum(dim=-1)
    min_cost_all = c_w.min(dim=-1).values
    effective_budget = torch.max(b_w, min_cost_all)
    audit_violation = (expected_final - effective_budget).clamp_min(0.0)

    active_bisect = mask.nonzero(as_tuple=True)[0][~feasible_m]
    tol_viol = audit_violation[active_bisect] > max(tolerance + 1e-5, 1e-4)
    if tol_viol.any():
        solver_status[active_bisect[tol_viol]] = 2

    return p.to(q.dtype), nu.to(q.dtype), solver_status


class DifferentiableKLProjection(torch.autograd.Function):
    """Implicit-differentiation KL projection layer.

    Maps ``(logits, cost, budget)`` to ``(p, nu, solver_status)`` where
    ``p`` is the KL projection of ``softmax(logits)`` onto
    ``{p in simplex | E_p[cost] <= budget}``.

    The backward pass uses the implicit function theorem on the KKT
    conditions of the convex projection problem, giving exact gradients
    through the Lagrange multiplier.  This is the same principle used in
    OptNet (Amos & Kolter, ICML 2017) and differentiable convex
    optimization layers (Agrawal et al., NeurIPS 2019).
    """

    @staticmethod
    def forward(
        ctx,
        logits: Tensor,
        cost: Tensor,
        budget: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        q = torch.softmax(logits, dim=-1)
        with torch.no_grad():
            p, nu, solver_status = _solve_kl_projection(q, cost, budget)
        ctx.save_for_backward(logits, cost, budget, nu, p)
        ctx.solver_status = solver_status
        return p, nu, solver_status

    @staticmethod
    def backward(
        ctx,
        grad_p: Tensor,
        _grad_nu: Tensor,
        _grad_status: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        logits, cost, budget, nu, p = ctx.saved_tensors

        batch, N = p.shape
        g = grad_p

        dot_gp = (g * p).sum(dim=-1, keepdim=True)
        v = p * (g - dot_gp)

        E = (p * cost).sum(dim=-1)
        delta_cost = cost - E.unsqueeze(-1)
        Var = (p * delta_cost.square()).sum(dim=-1).clamp_min(1e-12)

        S = (v * cost).sum(dim=-1)

        grad_logits = v.clone()
        grad_cost = torch.zeros_like(cost)
        grad_budget = torch.zeros_like(budget)

        active = nu > 1e-12
        if active.any():
            p_a = p[active]
            c_a = cost[active]
            nu_a = nu[active].unsqueeze(-1)
            v_a = v[active]
            E_a = E[active]
            Var_a = Var[active]
            S_a = S[active]

            delta = c_a - E_a.unsqueeze(-1)
            grad_logits_a = v_a - (S_a / Var_a).unsqueeze(-1) * p_a * delta

            grad_cost_a = (
                -nu_a * v_a
                - (S_a / Var_a).unsqueeze(-1)
                * p_a
                * (1.0 + nu_a * (E_a.unsqueeze(-1) - c_a))
            )

            grad_budget_a = S_a / Var_a

            grad_logits[active] = grad_logits_a
            grad_cost[active] = grad_cost_a
            grad_budget[active] = grad_budget_a

        infeasible = ctx.solver_status == 1
        if infeasible.any():
            grad_logits[infeasible] = 0.0
            grad_cost[infeasible] = 0.0
            grad_budget[infeasible] = 0.0

        return grad_logits, grad_cost, grad_budget


def kl_project_linear(
    q: Tensor,
    cost: Tensor,
    budget: Tensor,
    *,
    iterations: int = 48,
    tolerance: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]:
    """Non-differentiable KL projection (backward-compatible wrapper)."""
    return _solve_kl_projection(q, cost, budget, iterations=iterations, tolerance=tolerance)


def policy_entropy(pi: Tensor) -> Tensor:
    return -(pi * pi.clamp_min(1e-9).log()).sum(dim=-1)


def normalize_policy(pi: Tensor) -> Tensor:
    eps = torch.finfo(pi.dtype).tiny
    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    row_sum = pi.sum(dim=-1, keepdim=True)
    uniform = torch.full_like(pi, 1.0 / pi.shape[-1])
    pi = torch.where(row_sum > 0, pi / row_sum.clamp_min(eps), uniform)
    pi = pi.clamp_min(eps)
    return pi / pi.sum(dim=-1, keepdim=True)
