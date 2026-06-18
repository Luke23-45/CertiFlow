"""Shared interaction encoders for learned CertiQ dispatch policies."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def mlp(in_dim: int, hidden_dim: int, out_dim: int, layers: int, *, dropout: float = 0.0) -> nn.Sequential:
    """Build a compact GELU MLP."""
    assert layers >= 1, "layers must be positive."
    dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
    modules: list[nn.Module] = []
    for idx in range(len(dims) - 1):
        modules.append(nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            modules.append(nn.GELU())
            if dropout > 0:
                modules.append(nn.Dropout(dropout))
    return nn.Sequential(*modules)


class _FeedForward(nn.Module):
    """Pre-norm feed-forward block."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.ff(self.norm(x))


class _SelfAttentionBlock(nn.Module):
    """Permutation-equivariant self-attention block."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff = _FeedForward(d_model, hidden_dim=4 * d_model, dropout=dropout)

    def forward(self, x: Tensor) -> Tensor:
        y, _ = self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)
        x = x + y
        return self.ff(x)


class _InducedSetAttentionBlock(nn.Module):
    """Set Transformer style induced attention block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_inducing_points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_inducing_points = int(num_inducing_points)
        self.inducing_points = nn.Parameter(torch.randn(self.num_inducing_points, d_model) * 0.02)
        self.norm_x1 = nn.LayerNorm(d_model)
        self.norm_z1 = nn.LayerNorm(d_model)
        self.norm_z2 = nn.LayerNorm(d_model)
        self.norm_x2 = nn.LayerNorm(d_model)
        self.cross_in = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.self_latent = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_out = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ff_z = _FeedForward(d_model, hidden_dim=4 * d_model, dropout=dropout)
        self.ff_x = _FeedForward(d_model, hidden_dim=4 * d_model, dropout=dropout)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch = x.shape[0]
        z = self.inducing_points.unsqueeze(0).expand(batch, -1, -1)
        z_attn, _ = self.cross_in(self.norm_z1(z), self.norm_x1(x), self.norm_x1(x), need_weights=False)
        z = self.ff_z(z + z_attn)
        z_attn, _ = self.self_latent(self.norm_z2(z), self.norm_z2(z), self.norm_z2(z), need_weights=False)
        z = self.ff_z(z + z_attn)
        x_attn, _ = self.cross_out(self.norm_x2(x), self.norm_z2(z), self.norm_z2(z), need_weights=False)
        x = self.ff_x(x + x_attn)
        return x, z


class DispatchInteractionEncoder(nn.Module):
    """Shared interaction encoder for dispatch policies."""

    def __init__(
        self,
        feature_dim: int,
        *,
        d_model: int,
        d_global: int,
        num_layers: int,
        num_heads: int,
        num_inducing_points: int,
        dropout: float = 0.0,
        global_feature_dim: int = 0,
    ) -> None:
        super().__init__()
        assert feature_dim >= 1, "feature_dim must be positive."
        assert d_model >= 1 and d_global >= 1, "representation dims must be positive."
        self.feature_dim = int(feature_dim)
        self.d_model = int(d_model)
        self.d_global = int(d_global)
        self.global_feature_dim = int(global_feature_dim)
        self.input = mlp(feature_dim, d_model, d_model, layers=2, dropout=dropout)
        self.global_adapter = mlp(
            4 * d_model + self.global_feature_dim,
            d_model,
            2 * d_model,
            layers=2,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            if num_inducing_points > 0:
                self.blocks.append(
                    _InducedSetAttentionBlock(
                        d_model=d_model,
                        num_heads=num_heads,
                        num_inducing_points=num_inducing_points,
                        dropout=dropout,
                    )
                )
            else:
                self.blocks.append(_SelfAttentionBlock(d_model=d_model, num_heads=num_heads, dropout=dropout))
        self.global_head = mlp(
            3 * d_model + self.global_feature_dim,
            d_global,
            d_global,
            layers=2,
            dropout=dropout,
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_features: Tensor, global_features: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Return ``(token_embeddings, global_embedding)``."""
        x = self.input(token_features)
        pooled_mean = x.mean(dim=1)
        pooled_max = x.amax(dim=1)
        if global_features is None:
            global_features = x.new_zeros(x.shape[0], 0)
        adapter_in = torch.cat([pooled_mean, pooled_max, pooled_mean - pooled_max, pooled_mean.abs()], dim=-1)
        if self.global_feature_dim > 0:
            adapter_in = torch.cat([adapter_in, global_features], dim=-1)
        shift, scale = self.global_adapter(adapter_in).chunk(2, dim=-1)
        x = x * (1.0 + torch.tanh(scale).unsqueeze(1)) + shift.unsqueeze(1)
        latent_summary = None
        for block in self.blocks:
            if isinstance(block, _InducedSetAttentionBlock):
                x, latent_summary = block(x)
            else:
                x = block(x)
        x = self.norm(x)
        pooled_mean = x.mean(dim=1)
        pooled_max = x.amax(dim=1)
        global_in = [pooled_mean, pooled_max]
        if latent_summary is not None:
            global_in.append(latent_summary.mean(dim=1))
        else:
            global_in.append(x.new_zeros(x.shape[0], self.d_model))
        if self.global_feature_dim > 0:
            global_in.append(global_features)
        global_emb = self.global_head(torch.cat(global_in, dim=-1))
        return x, global_emb


def index_token_features(
    Q: Tensor,
    mu: Tensor,
    xi: Tensor | None = None,
    *,
    d_xi: int = 0,
) -> tuple[Tensor, Tensor]:
    """Return token and global features for the index model."""
    mu_safe = mu.clamp_min(torch.finfo(mu.dtype).tiny)
    qmd = (2.0 * Q + 1.0) / mu_safe
    token_parts = [
        Q.unsqueeze(-1),
        mu.unsqueeze(-1),
        torch.log1p(Q).unsqueeze(-1),
        torch.log(mu_safe).unsqueeze(-1),
        (Q / mu_safe).unsqueeze(-1),
        qmd.unsqueeze(-1),
    ]
    if xi is not None:
        assert d_xi > 0, "xi was provided but the model was built without context features."
        assert xi.shape[:2] == Q.shape, "xi must match Q batch and resource dimensions."
        if d_xi > 0:
            assert xi.shape[-1] == d_xi, "xi feature width must match d_xi."
        token_parts.append(xi)
    elif d_xi > 0:
        token_parts.append(torch.zeros(Q.shape[0], Q.shape[1], d_xi, device=Q.device, dtype=Q.dtype))
    token_features = torch.cat(token_parts, dim=-1)
    global_features = torch.cat(
        [
            Q.mean(dim=-1, keepdim=True),
            Q.amax(dim=-1, keepdim=True),
            mu_safe.mean(dim=-1, keepdim=True),
            mu_safe.amin(dim=-1, keepdim=True),
            qmd.mean(dim=-1, keepdim=True),
            qmd.amin(dim=-1, keepdim=True),
            qmd.amax(dim=-1, keepdim=True),
            qmd.std(dim=-1, keepdim=True, unbiased=False),
        ],
        dim=-1,
    )
    return token_features, global_features
