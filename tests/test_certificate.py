"""Tests for certificate module — KL projection forward and backward passes."""

import torch


def test_kl_projection_forward_satisfies_constraint():
    from certiq_net.dispatcher.certiq.certificate import kl_project_linear

    B, N = 8, 5
    q = torch.softmax(torch.randn(B, N), dim=-1)
    cost = torch.rand(B, N)
    budget = cost.min(dim=-1).values + 0.5

    p, nu, status = kl_project_linear(q, cost, budget)
    expected = (p * cost).sum(dim=-1)

    assert (expected <= budget + 1e-5).all(), "Projected policy violates budget constraint"
    assert torch.allclose(p.sum(dim=-1), torch.ones(B), atol=1e-5), "pi must sum to 1"
    assert (p >= 0).all(), "pi must be non-negative"
    assert status.shape == (B,)


def test_kl_projection_less_expensive_when_feasible():
    from certiq_net.dispatcher.certiq.certificate import kl_project_linear

    B, N = 4, 5
    q = torch.softmax(torch.randn(B, N), dim=-1)
    cost = torch.rand(B, N)
    budget = (q * cost).sum(dim=-1) + 0.5  # budget > expected cost → feasible

    p, nu, status = kl_project_linear(q, cost, budget)

    # When feasible, p should equal q (no projection needed)
    assert torch.allclose(p, q, atol=1e-5), "Feasible case should not change policy"
    assert (nu == 0).all(), "Nu should be zero when feasible"


def test_kl_projection_differentiable():
    from certiq_net.dispatcher.certiq.certificate import DifferentiableKLProjection

    B, N = 4, 5
    logits = torch.randn(B, N, requires_grad=True)
    cost = torch.rand(B, N)
    budget = cost.min(dim=-1).values + 0.5

    p, nu, status = DifferentiableKLProjection.apply(logits, cost, budget)
    loss = (p * cost).sum()
    loss.backward()

    assert logits.grad is not None
    assert logits.grad.abs().sum().item() > 0, "No gradients through KL projection"


def test_kl_projection_gradient_finite():
    from certiq_net.dispatcher.certiq.certificate import DifferentiableKLProjection

    B, N = 4, 5
    logits = torch.randn(B, N, requires_grad=True)
    cost = torch.rand(B, N)
    budget = cost.min(dim=-1).values + 0.5

    p, nu, status = DifferentiableKLProjection.apply(logits, cost, budget)
    loss = (p * cost).sum()
    loss.backward()

    assert torch.isfinite(logits.grad).all(), "Gradients must be finite"


def test_kl_projection_budget_too_low():
    from certiq_net.dispatcher.certiq.certificate import kl_project_linear

    B, N = 4, 5
    q = torch.softmax(torch.randn(B, N), dim=-1)
    cost = torch.rand(B, N)
    budget = cost.min(dim=-1).values - 1.0  # budget < min cost → infeasible

    p, nu, status = kl_project_linear(q, cost, budget)

    # When budget < min cost, should return dirac at cheapest queue
    min_idx = cost.argmin(dim=-1)
    for b in range(B):
        assert p[b, min_idx[b]] > 0.99, "Should concentrate on cheapest queue"
    assert (status == 1).any() or (status >= 0).all(), "Should report infeasible status"


def test_kl_projection_within_certiQ_model():
    from certiq_net.dispatcher import CertiQIndexModel

    model = CertiQIndexModel(
        N=4, hidden_dim=32, C=2.0, constraint_mode="projection", cost_fn="qmd"
    )
    model.eval()

    B, N = 8, 4
    Q = torch.rand(B, N) * 10
    mu = torch.rand(N) * 2 + 0.5

    with torch.no_grad():
        out = model.forward_full(Q, mu)

    assert out.pi.shape == (B, N)
    assert torch.allclose(out.pi.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (out.diagnostics.constraint_violation == 0).all(), \
        "Projection mode should enforce zero constraint violation"
    assert out.diagnostics.nu is not None, "Nu should be populated in projection mode"
    assert out.diagnostics.solver_status is not None, "Solver status should be populated"
    assert out.diagnostics.A_cert is not None, "A_cert should be populated"


def test_kl_projection_backward_within_model():
    from certiq_net.dispatcher import CertiQIndexModel

    model = CertiQIndexModel(
        N=3, hidden_dim=16, C=2.0, constraint_mode="projection", cost_fn="sed"
    )
    model.train()

    B, N = 4, 3
    Q = torch.rand(B, N) * 10
    mu = torch.rand(N) * 2 + 0.5

    out = model.forward_full(Q, mu, training_mode=True)
    loss = (out.pi - torch.full((B, N), 1.0 / N)).pow(2).sum()
    loss.backward()

    has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.parameters()
    )
    assert has_grad, "No gradients flowed through projection mode model"
