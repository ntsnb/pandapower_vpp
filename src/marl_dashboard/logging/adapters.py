from __future__ import annotations

from typing import Any


class EnvironmentAdapter:
    """Read-only helper for turning env transitions into dashboard payloads."""

    def transition_payload(self, *, observation: Any = None, action: Any = None, reward: Any = None, info: dict[str, Any] | None = None) -> dict[str, Any]:
        return {"observation": observation, "action": action, "reward": reward, "info": info or {}}


class AlgorithmAdapter:
    """Read-only helper for normalizing learner metrics."""

    def loss_terms(self, metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in metrics.items()
            if key.endswith("_loss") or key in {"total_loss", "td_error", "q_loss"}
        }
