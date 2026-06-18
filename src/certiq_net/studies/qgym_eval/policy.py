"""CertiQ policy wrapper for QGym evaluation."""

from __future__ import annotations

import torch


class CertiQPolicy:
    """Wraps a CertiQIndexModel for QGym's discrete-event simulator.

    Converts CertiQ's per-queue allocation (B, Q) into QGym's
    (B, S, Q) priority matrix by expanding across servers via the
    routing topology.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model
        self.device = device

    def test_forward(
        self,
        step: int,
        batch_queue: torch.Tensor,
        batch_time: torch.Tensor,
        repeated_queue: torch.Tensor,
        repeated_network: torch.Tensor,
        repeated_mu: torch.Tensor,
        repeated_h: torch.Tensor,
    ) -> torch.Tensor:
        # repeated_network: (B, S, Q), repeated_mu: (B, S, Q)
        # Compute effective service rate per queue (B, Q)
        mu_eff = (repeated_network * repeated_mu).sum(dim=1)

        with torch.no_grad():
            pi, diagnostics = self.model(batch_queue, mu_eff)

        # Expand per-queue allocation to per-server priority (B, S, Q)
        priority = pi.unsqueeze(1) * repeated_network

        return priority
