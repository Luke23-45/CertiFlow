"""Smoke tests for CertiQIndexModel — instantiation and forward pass."""

import torch


def test_model_instantiation():
    from certiq_net.dispatcher import CertiQIndexModel

    model = CertiQIndexModel(
        N=4,
        hidden_dim=32,
        tau=1.0,
        C=2.0,
        cost_fn="sed",
        encoder_layers=1,
        num_heads=2,
        num_inducing_points=2,
        dropout=0.0,
        constraint_mode="lagrangian",
    )
    assert model.N == 4
    assert model.tau == 1.0
    assert model.C == 2.0
    assert model.cost_fn == "sed"


def test_model_forward_lagrangian():
    from certiq_net.dispatcher import CertiQIndexModel

    model = CertiQIndexModel(
        N=4,
        hidden_dim=32,
        tau=1.0,
        C=2.0,
        cost_fn="sed",
        encoder_layers=1,
        num_heads=2,
        num_inducing_points=2,
    )
    model.eval()

    B, N = 8, 4
    Q = torch.rand(B, N) * 10
    mu = torch.rand(N) * 2 + 0.5

    with torch.no_grad():
        pi, diagnostics = model(Q, mu)

    assert pi.shape == (B, N)
    assert diagnostics.certificate_slack.shape == (B,)
    assert diagnostics.constraint_violation.shape == (B,)
    assert diagnostics.policy_entropy.shape == (B,)

    # pi should be a valid probability distribution
    assert torch.allclose(pi.sum(dim=-1), torch.ones(B), atol=1e-5)
    assert (pi >= 0).all()
    assert (pi <= 1).all()


def test_model_forward_full():
    from certiq_net.dispatcher import CertiQIndexModel

    model = CertiQIndexModel(
        N=4,
        hidden_dim=32,
        tau=1.0,
        C=2.0,
        cost_fn="qmd",
        encoder_layers=1,
        num_heads=2,
        num_inducing_points=0,
    )
    model.eval()

    B, N = 8, 4
    Q = torch.rand(B, N) * 10
    mu = torch.rand(B, N) * 2 + 0.5

    with torch.no_grad():
        out = model.forward_full(Q, mu)

    assert out.pi.shape == (B, N)
    assert out.value.shape == (B,)
    assert out.diagnostics is not None
    assert out.proposal_logits.shape == (B, N)


def test_model_different_cost_fns():
    from certiq_net.dispatcher import CertiQIndexModel

    B, N = 4, 3
    Q = torch.rand(B, N) * 10
    mu = torch.rand(N) * 2 + 0.5

    for cost_fn in ("sed", "qmd"):
        model = CertiQIndexModel(
            N=N,
            hidden_dim=16,
            tau=1.0,
            C=2.0,
            cost_fn=cost_fn,
            encoder_layers=1,
            num_heads=2,
            num_inducing_points=2,
        )
        model.eval()
        with torch.no_grad():
            pi, _ = model(Q, mu)
        assert pi.shape == (B, N)
        assert torch.allclose(pi.sum(dim=-1), torch.ones(B), atol=1e-5)


def test_model_gradient_flow():
    from certiq_net.dispatcher import CertiQIndexModel

    model = CertiQIndexModel(
        N=3,
        hidden_dim=16,
        tau=1.0,
        C=2.0,
        cost_fn="sed",
        encoder_layers=1,
        num_heads=2,
        num_inducing_points=2,
    )

    B, N = 4, 3
    Q = torch.rand(B, N) * 10
    mu = torch.rand(N) * 2 + 0.5
    target = torch.full((B, N), 1.0 / N)

    pi, _ = model(Q, mu)
    loss = (pi - target).pow(2).sum()
    loss.backward()

    has_grad = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum().item() > 0:
            has_grad = True
            break
    assert has_grad, "No gradients flowed through the model"


def test_import_top_level():
    from certiq_net import CertiQIndexModel, DispatcherDiagnostics, DispatcherForward
    assert CertiQIndexModel is not None
    assert DispatcherDiagnostics is not None
    assert DispatcherForward is not None


def test_dispatcher_module_exports():
    from certiq_net.dispatcher import (
        CertifiedGeometry,
        CertiQIndexModel,
        DispatcherDiagnostics,
        DispatcherForward,
        DispatchInteractionEncoder,
        MarginalIndexHead,
        delay_arrival_coordinate,
        delay_envelope,
        index_token_features,
        quadratic_drift_index,
        sed_hard_policy,
        sed_index,
        sed_soft_policy,
    )
    assert all(x is not None for x in [
        CertiQIndexModel, CertifiedGeometry,
        DispatcherDiagnostics, DispatcherForward, DispatchInteractionEncoder,
        MarginalIndexHead, delay_arrival_coordinate, delay_envelope,
        index_token_features, quadratic_drift_index, sed_hard_policy,
        sed_index, sed_soft_policy,
    ])
