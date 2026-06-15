from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from scipy.stats import beta as beta_dist

from src.voicegraph.observability.metrics import update_bandit_metrics

logger = logging.getLogger(__name__)


class BanditOptimizer:
    def __init__(self, alpha_prior: float = 1.0, beta_prior: float = 1.0):
        self.alpha_prior: float = alpha_prior
        self.beta_prior: float = beta_prior
        self.arm_alpha: Dict[str, float] = {}
        self.arm_beta: Dict[str, float] = {}

    def register_arm(self, arm_id: str) -> None:
        if arm_id not in self.arm_alpha:
            self.arm_alpha[arm_id] = self.alpha_prior
            self.arm_beta[arm_id] = self.beta_prior

    def update_arm(self, arm_id: str, reward: float, campaign_id: str = "default") -> None:
        if arm_id not in self.arm_alpha:
            self.register_arm(arm_id)
        if reward > 0:
            self.arm_alpha[arm_id] += reward
        else:
            self.arm_beta[arm_id] += 1.0

        update_bandit_metrics(
            campaign_id=campaign_id,
            weights={arm_id: {"alpha": self.arm_alpha[arm_id], "beta": self.arm_beta[arm_id]}},
        )

    def select_arm(self, arm_ids: List[str]) -> str:
        best_arm = arm_ids[0]
        best_score = -float("inf")

        for arm_id in arm_ids:
            if arm_id not in self.arm_alpha:
                self.register_arm(arm_id)
            a = self.arm_alpha[arm_id]
            b = self.arm_beta[arm_id]
            theta = float(beta_dist.rvs(a=a, b=b, size=1)[0])
            if theta > best_score:
                best_score = theta
                best_arm = arm_id

        return best_arm

    def select_script(self) -> str:
        best_arm: str | None = None
        best_score = -float("inf")

        for arm_id in self.arm_alpha:
            a = self.arm_alpha[arm_id]
            b = self.arm_beta[arm_id]
            theta = float(beta_dist.rvs(a=a, b=b, size=1)[0])
            if theta > best_score:
                best_score = theta
                best_arm = arm_id

        if best_arm is None:
            msg = "Нет зарегистрированных сценариев"
            raise RuntimeError(msg)

        return best_arm

    def get_weights(self) -> Dict[str, float]:
        weights = {}
        for arm_id in self.arm_alpha:
            a = self.arm_alpha[arm_id]
            b = self.arm_beta[arm_id]
            weights[arm_id] = a / max(a + b, 1)
        return weights

    def get_params(self, arm_id: str) -> Tuple[float, float]:
        return (self.arm_alpha.get(arm_id, self.alpha_prior), self.arm_beta.get(arm_id, self.beta_prior))
