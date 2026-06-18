"""SB3 ActorCriticPolicy wrapping CertiQIndexModel for QGym's PPO pipeline."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.distributions.one_hot_categorical as one_hot_categorical
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import PyTorchObs
from torch import Tensor


class _DummyMLPExtractor(nn.Module):
    """Minimal zero-param MLP extractor for SB3 compatibility.

    Provides the `policy_net`, `value_net`, `latent_dim_pi`,
    `latent_dim_vf` attributes that CustomPPOTrainer expects.
    """

    def __init__(self, features_dim: int) -> None:
        super().__init__()
        self.policy_net = nn.Identity()
        self.value_net = nn.Identity()
        self.latent_dim_pi = features_dim
        self.latent_dim_vf = features_dim

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        return features, features


class CertiQSB3Policy(ActorCriticPolicy):
    """Wraps CertiQ's MarginalIndexHead as an SB3 ActorCriticPolicy.

    The policy accepts raw queue-length observations ``obs: (B, Q)``,
    produces a priority matrix ``action: (B, S, Q)``, and returns
    value estimates ``(B, 1)`` with log-probabilities ``(B,)``.

    Designed to be consumed by QGym's ``CustomPPOTrainer`` without
    any modifications to ``extern/QGym``.
    """

    def __init__(
        self,
        *args: Any,
        network: Tensor,
        mu: Tensor,
        alpha: Tensor,
        D: Tensor,
        tau: float = 1.0,
        randomize: bool = True,
        scale: int = 20,
        rescale_v: bool = False,
        time_f: bool = False,
        net_arch: dict | None = None,
        **kwargs: Any,
    ) -> None:
        self.certiq_network = network  # (S, Q) routing topology
        self.certiq_mu = mu  # (S, Q) per-server service rates
        self.certiq_alpha = alpha
        self.certiq_D = D
        self.tau = tau
        self._randomize = randomize
        self._rescale_v = rescale_v
        self.q = network.shape[1]
        self.s = network.shape[0]
        self.returns_mean = 0.0
        self.returns_std = 1.0
        self.mean_queue_length = 0.0
        self.std_queue_length = 1.0
        self._certiq_head_params: dict | None = None

        # net_arch must be passed or default to tiny
        net_arch = net_arch or dict(pi=[], vf=[])
        super().__init__(*args, net_arch=net_arch, **kwargs)

    @staticmethod
    def _load_certiq_model_config() -> dict:
        """Load certiq_index.yaml from the package config directory."""
        from pathlib import Path

        import yaml

        import certiq_net

        cfg_path = (
            Path(certiq_net.__file__).resolve().parents[2]
            / "configs" / "model" / "certiq_index.yaml"
        )
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            return cfg.get("model", cfg)
        return {}

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        features_dim = self.observation_space.shape[0]

        # Load certiq head params from config if not already provided
        if self._certiq_head_params is None:
            self._certiq_head_params = self._load_certiq_model_config()

        from certiq_net.dispatcher.certiq.index_model import MarginalIndexHead

        # Build the core CertiQ model
        self.marginal_index_head = MarginalIndexHead(
            N=self.q,
            hidden_dim=self._certiq_head_params.get("hidden_dim", 128),
            d_xi=self._certiq_head_params.get("d_xi", 0),
            encoder_layers=self._certiq_head_params.get("encoder_layers", 2),
            num_heads=self._certiq_head_params.get("num_heads", 4),
            num_inducing_points=self._certiq_head_params.get(
                "num_inducing_points", 4
            ),
            dropout=self._certiq_head_params.get("dropout", 0.0),
        ).to(self.device)

        # Effective per-queue service rate
        self.register_buffer(
            "_mu_eff",
            (self.certiq_network.to(self.device) * self.certiq_mu.to(self.device)).sum(
                dim=0
            ),
        )

        # SB3-expected attributes (zero-param identity modules)
        self.features_extractor = nn.Identity()
        self.features_dim = features_dim

        # Put encoder in pi_features_extractor so its params go to the policy optimizer
        self.pi_features_extractor = self.marginal_index_head.encoder
        self.vf_features_extractor = nn.Identity()

        self.mlp_extractor = _DummyMLPExtractor(features_dim)

        # CertiQ's output heads become SB3's action_net and value_net
        self.action_net = self.marginal_index_head.index_head
        self.value_net = self.marginal_index_head.value_head

        self._initialize_parameters()

    def _expand_pi_to_priority(self, pi: Tensor) -> Tensor:
        """Convert per-queue probs ``(B, Q)`` to per-server priority ``(B, S, Q)``."""
        device = pi.device
        network = self.certiq_network.to(device)
        priority = pi.unsqueeze(1) * network.unsqueeze(0)
        zero_mask = priority.sum(dim=-1, keepdim=True) == 0
        priority = priority + zero_mask * network.unsqueeze(0)
        return priority / priority.sum(dim=-1, keepdim=True).clamp(min=1e-10)

    def _run_certiq(
        self, obs: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run CertiQ forward, return ``(pi, value, logits)``."""
        batch_size = obs.shape[0]
        mu_batch = self._mu_eff.unsqueeze(0).expand(batch_size, -1)
        logits, value = self.marginal_index_head(obs, mu_batch, xi=None)
        proposal_logits = -logits / self.tau
        pi = F.softmax(proposal_logits.clamp(-20, 20), dim=-1)
        return pi, value, logits

    def forward(
        self, obs: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        obs = obs.view(-1, self.q)

        pi, value, _ = self._run_certiq(obs)
        action_probs = self._expand_pi_to_priority(pi)

        if deterministic:
            action_indices = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_indices, num_classes=self.q)
        else:
            dist = one_hot_categorical.OneHotCategorical(probs=action_probs)
            action = dist.sample()

        selected_probs = (action * action_probs).sum(dim=-1)
        log_prob = torch.log(selected_probs.clamp(min=1e-10)).sum(dim=1)

        return action.float(), value.unsqueeze(-1), log_prob

    def _get_prob_act(
        self, obs: Tensor, deterministic: bool = False
    ) -> tuple[Tensor, Tensor]:
        obs = obs.view(-1, self.q)
        pi, _, _ = self._run_certiq(obs)
        action_probs = self._expand_pi_to_priority(pi)

        if deterministic:
            action_indices = torch.argmax(action_probs, dim=-1)
            action = F.one_hot(action_indices, num_classes=self.q)
        else:
            dist = one_hot_categorical.OneHotCategorical(probs=action_probs)
            action = dist.sample()

        return action.float(), action_probs

    def evaluate_actions(
        self, obs: Tensor, actions: Tensor
    ) -> tuple[Tensor, Tensor | None]:
        obs = obs.view(-1, self.q)
        pi, _, _ = self._run_certiq(obs)
        action_probs = self._expand_pi_to_priority(pi)
        actions = actions.reshape((-1, self.s, self.q))

        selected_probs = (actions * action_probs).sum(dim=-1)
        log_prob = torch.log(selected_probs.clamp(min=1e-10)).sum(dim=1)
        return log_prob, None

    def predict_values(self, obs: PyTorchObs) -> Tensor:
        obs = obs.view(-1, self.q)
        _, value, _ = self._run_certiq(obs)
        return value.unsqueeze(-1)

    def evaluate_values(self, obs: PyTorchObs) -> Tensor:
        return self.predict_values(obs)

    def predict(
        self,
        observation: np.ndarray | dict[str, np.ndarray],
        state: tuple[np.ndarray, ...] | None = None,
        episode_start: np.ndarray | None = None,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
        self.set_training_mode(False)
        obs_tensor, vectorized_env = self.obs_to_tensor(observation)
        obs_tensor = obs_tensor.view(-1, self.q)
        with torch.no_grad():
            action, action_probs = self._get_prob_act(
                obs_tensor, deterministic
            )
        return action.cpu().numpy(), action_probs.cpu().numpy()

    def update_mean_std(
        self, mean_queue_length: float, std_queue_length: float
    ) -> None:
        self.mean_queue_length = mean_queue_length
        self.std_queue_length = max(std_queue_length, 1e-8)

    def update_rollout_stats(
        self, returns_mean: float, returns_std: float
    ) -> None:
        self.returns_mean = returns_mean
        self.returns_std = max(returns_std, 1e-8)

    def compute_cost_and_budget(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute cost, budget, and per-queue probs for the constraint.

        Uses the same cost function (QMD by default) and budget
        (``C = 20.0``) as the CertiQIndexModel config.

        Returns ``(cost, budget, pi)`` — each ``(B, Q)`` except
        ``budget`` which is ``(B,)``.
        """
        batch_size = obs.shape[0]
        C = self._certiq_head_params.get("C", 20.0)
        cost_fn = self._certiq_head_params.get("cost_fn", "qmd")

        mu_eff = self._mu_eff.to(obs.device).unsqueeze(0).expand(batch_size, -1)

        if cost_fn == "sed":
            cost = (obs + 1) / mu_eff
        else:
            cost = (2 * obs + 1) / mu_eff

        cost_min = cost.min(dim=-1).values
        budget = cost_min + C

        pi, _, _ = self._run_certiq(obs)
        return cost, budget, pi

    def compute_constraint_violation(self, obs: Tensor) -> Tensor:
        """Return the constraint violation ``(B,)`` for the Lagrangian loss.

        ``violation = max(E_pi[cost] - budget, 0)``

        The result retains the graph so gradients flow to the policy
        network through the Lagrangian penalty.
        """
        cost, budget, pi = self.compute_cost_and_budget(obs)
        a_final = (pi * cost).sum(dim=-1)
        return (a_final - budget).clamp(min=0.0)
