from __future__ import annotations

from typing import Any


class RuleV0EnvelopePolicy:
    policy_name = "rule_v0"

    def build(self, simulator, vpp, step: int, bid: dict[str, Any], fr, price: float, grid_state: dict[str, Any] | None = None) -> dict[str, Any]:
        return build_rule_v0_envelope(simulator, vpp, step, bid, fr, price, grid_state=grid_state)


def build_rule_v0_envelope(
    simulator,
    vpp,
    step: int,
    bid: dict[str, Any],
    fr,
    price: float,
    *,
    grid_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Adapter preserving the existing simulator rule-based envelope behavior."""

    envelope = simulator._build_dso_operating_envelope(vpp, step, bid, fr, price, grid_state=grid_state)
    return {**envelope, "source_policy": "rule_v0"}
