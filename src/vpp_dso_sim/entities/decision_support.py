from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DecisionSupportInterface:
    """Placeholder interface for future LLM-backed operator support."""

    def summarize_grid_state(self, network_state: dict[str, Any]) -> dict[str, Any]:
        return {"summary": "grid_state", "data": network_state}

    def generate_operator_report(self, network_state: dict[str, Any], violations: list[dict]) -> str:
        return (
            f"Grid converged={network_state.get('converged')}; "
            f"min_vm_pu={network_state.get('min_vm_pu')}; "
            f"max_line_loading_percent={network_state.get('max_line_loading_percent')}; "
            f"violations={len(violations)}"
        )

    def explain_constraint_violations(self, violations: list[dict]) -> list[str]:
        return [f"{item.get('kind')} at {item.get('element')} magnitude={item.get('magnitude')}" for item in violations]

    def suggest_safe_actions(self, representative_data: dict[str, Any]) -> dict[str, Any]:
        return {
            "status": "rule_placeholder",
            "suggestion": "reduce injection near high-voltage PCCs or increase injection near low-voltage PCCs",
            "representative_data": representative_data,
        }

