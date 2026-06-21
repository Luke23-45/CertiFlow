"""CustomPPOTrainer subclass that adds the CertiQ Lagrangian constraint loss."""

from __future__ import annotations

import time
from typing import Any

import torch as th

from certiq_net.studies.qgym_eval.train.qgym_import import CustomPPOTrainer

# Scaling factor to balance Lagrangian constraint loss against the
# normalized policy-gradient loss.  The raw cost values from the
# environment are O(1000) larger than the normalized advantages, so
# the Lagrangian penalty must be divided by this factor to prevent
# the constraint gradient from dominating and destroying the policy.
LAGR_SCALE = 1000.0
NU_MAX = 10.0
NU_DELTA_MAX = 0.5
FROZEN_RATIO_EPS = 1e-3
VALUE_RESIDUAL_CLIP = 10.0
LAG_MAX_FRAC = 0.5


class CertiqPPOTrainer(CustomPPOTrainer):
    """Extends QGym's ``CustomPPOTrainer`` with the CertiQ Lagrangian
    constraint penalty on the policy loss.

    The dual variable ``nu`` is updated via manual gradient ascent
    (not part of the Adam optimizers).
    """

    def __init__(self, *args: Any, lr_nu: float = 1e-3, **kwargs: Any) -> None:
        self.lr_nu = lr_nu
        self._nu_val = 0.0  # dual variable as Python float
        super().__init__(*args, **kwargs)

    def _robust_value_loss(
        self, returns: th.Tensor, values_pred: th.Tensor
    ) -> th.Tensor:
        """Residual-clipped Huber value loss.

        The residual clip bounds the gradient from large transient queue
        spikes. The Huber term keeps the loss smooth around zero while
        down-weighting outliers.
        """
        value_diff = th.clamp(
            values_pred - returns, -VALUE_RESIDUAL_CLIP, VALUE_RESIDUAL_CLIP
        )
        return th.nn.functional.smooth_l1_loss(
            value_diff, th.zeros_like(value_diff), beta=1.0
        )

    def train(self) -> None:
        """Override ``CustomPPOTrainer.train`` to add the Lagrangian
        constraint penalty ``nu * violation.mean()`` to the policy loss.

        The dual variable ``nu`` is updated via gradient ascent on the
        Lagrangian dual: ``nu = max(0, nu + lr_nu * excess)`` where
        ``excess = E_pi[cost] - budget`` (can be negative when constraint
        is satisfied).
        """
        policy = self.policy
        if not hasattr(policy, "compute_cost_and_budget"):
            raise AttributeError(
                "CertiqPPOTrainer requires a policy with "
                "compute_cost_and_budget(obs) method (CertiQSB3Policy)."
            )

        self.policy.set_training_mode(True)
        training_time_start = time.time()
        self.training_iteration += 1

        self._update_learning_rate(self.optimizer_policy, self.optimizer_value)

        clip_range = self.clip_range(self._current_progress_remaining)
        clipping_alpha = 1.0 - self.training_iteration / self.num_epochs
        clip_range = max(0.01, clipping_alpha * clip_range)

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(
                self._current_progress_remaining
            )

        pg_losses, value_losses, entropy_losses, lagrangian_losses = [], [], [], []
        clip_fractions = []
        excess_means = []
        ratio_devs = []
        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions

                log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations, actions
                )

                advantages = rollout_data.advantages
                if len(advantages.shape) == 1:
                    advantages = advantages.unsqueeze(-1)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(
                    ratio, 1 - clip_range, 1 + clip_range
                )
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                clip_fraction = th.mean(
                    (th.abs(ratio - 1) > clip_range).float()
                ).item()
                clip_fractions.append(clip_fraction)
                pg_losses.append(policy_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())
                policy_loss = policy_loss + self.ent_coef * entropy_loss

                with th.no_grad():
                    ratio_dev = th.mean(th.abs(ratio - 1)).item()
                ratio_devs.append(ratio_dev)

                # --- Lagrangian constraint ---
                policy_frozen = ratio_dev < FROZEN_RATIO_EPS
                if policy_frozen:
                    lag_loss = th.zeros((), device=policy_loss.device)
                else:
                    cost, budget, pi = policy.compute_cost_and_budget(
                        rollout_data.observations
                    )
                    a_final = (pi * cost).sum(dim=-1)
                    excess = a_final - budget  # can be negative
                    violation = excess.clamp(min=0.0)

                    lag_loss = self._nu_val * violation.mean() / LAGR_SCALE
                    lag_cap = LAG_MAX_FRAC * (policy_loss.detach().abs() + 1e-8)
                    lag_loss = th.minimum(lag_loss, lag_cap)
                    excess_means.append(excess.mean().item())

                lagrangian_losses.append(lag_loss.item())
                policy_loss = policy_loss + lag_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean(
                        (th.exp(log_ratio) - 1) - log_ratio
                    ).cpu().item()
                    approx_kl_divs.append(approx_kl_div)

                if (
                    self.target_kl is not None
                    and approx_kl_div > 1.5 * self.target_kl
                ):
                    continue_training = False
                    if self.verbose >= 1:
                        print(
                            f"Early stopping at step {epoch} "
                            f"due to reaching max kl: {approx_kl_div:.2f}"
                        )
                    break

                self.optimizer_policy.zero_grad()
                policy_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer_policy.step()

                # Reset gradients after policy phase so the shared encoder
                # does not accumulate policy-phase gradients into the value phase.
                self.policy.zero_grad()

            # Value phase (unchanged from parent)
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                values = self.policy.evaluate_values(
                    rollout_data.observations
                )
                values = values.flatten()
                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values,
                        -clip_range_vf,
                        clip_range_vf,
                    )

                value_loss = self.vf_coef * self._robust_value_loss(
                    rollout_data.returns, values_pred
                )
                value_losses.append(value_loss.item())

                self.optimizer_value.zero_grad()
                value_loss.backward()
                th.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer_value.step()

            if not continue_training:
                break

        # --- Dual variable update ---
        if excess_means:
            avg_excess = th.tensor(excess_means).mean().item()
            nu_delta = self.lr_nu * avg_excess / LAGR_SCALE
            nu_delta = max(-NU_DELTA_MAX, min(NU_DELTA_MAX, nu_delta))
            self._nu_val = max(0.0, min(NU_MAX, self._nu_val + nu_delta))

        training_time_end = time.time()
        print(f"training_time: {training_time_end - training_time_start}")

        self.logger.record(
            "train/policy_gradient_loss",
            th.tensor(pg_losses).mean().item(),
        )
        self.logger.record(
            "train/value_loss", th.tensor(value_losses).mean().item()
        )
        self.logger.record(
            "train/approx_kl", th.tensor(approx_kl_divs).mean().item()
        )
        self.logger.record(
            "train/clip_fraction", th.tensor(clip_fractions).mean().item()
        )
        self.logger.record("train/ratio_dev", th.tensor(ratio_devs).mean().item())
        self.logger.record(
            "train/lagrangian_loss",
            th.tensor(lagrangian_losses).mean().item(),
        )
        self.logger.record("train/entropy_loss", th.tensor(entropy_losses).mean().item())
        self.logger.record("train/nu", self._nu_val)
