from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class RewardTermContract:
    name: str
    source: str
    sign: str
    weight: float
    visible_to: tuple[str, ...]
    notes: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["visible_to"] = list(self.visible_to)
        return payload


@dataclass(frozen=True)
class AgentRewardContract:
    agent_role: str
    objective: str
    time_scale: str
    raw_global_dso_reward_weight: float
    terms: tuple[RewardTermContract, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["terms"] = [term.to_dict() for term in self.terms]
        return payload


DISPATCH_RAW_DSO_REWARD_WEIGHT = 0.0
PORTFOLIO_RAW_DSO_REWARD_WEIGHT = 0.0
PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT = 1.0
PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS = 24

FLEXIBILITY_SERVICE_PRICE_MULTIPLIER = 1.00
AVAILABILITY_PAYMENT_RATE = 0.02
DISPATCH_PRIVATE_PROFIT_WEIGHT = 0.02
DISPATCH_TRACKING_PENALTY_WEIGHT = 25.0
DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT = 5.0
DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT = 10.0
DISPATCH_COMFORT_SOC_PENALTY_WEIGHT = 0.001
DISPATCH_PREFERRED_REGION_BONUS_WEIGHT = 0.50
SHIELD_INTERVENTION_LINEAR_PENALTY_WEIGHT = 5.0
SHIELD_INTERVENTION_QUADRATIC_PENALTY_WEIGHT = 10.0


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def shield_intervention_metrics(reward_components: dict[str, Any]) -> dict[str, float]:
    """Return learning-layer metrics for raw actions repaired by local/AC shields."""

    action_gap = max(0.0, _safe_float(reward_components.get("action_projection_gap_mw")))
    local_gap = max(0.0, _safe_float(reward_components.get("local_bounds_projection_gap_mw")))
    ac_gap = max(0.0, _safe_float(reward_components.get("ac_aware_projection_gap_mw")))
    ac_certified_gap = max(0.0, _safe_float(reward_components.get("ac_certified_projection_gap_mw")))
    fallback_gap = max(action_gap, local_gap, ac_gap, ac_certified_gap)
    shield_gap = (
        local_gap + ac_gap + ac_certified_gap
        if (ac_gap > 0.0 or local_gap > 0.0 or ac_certified_gap > 0.0)
        else fallback_gap
    )
    penalty = (
        SHIELD_INTERVENTION_LINEAR_PENALTY_WEIGHT * shield_gap
        + SHIELD_INTERVENTION_QUADRATIC_PENALTY_WEIGHT * shield_gap * shield_gap
    )
    return {
        "action_projection_gap_mw": float(action_gap),
        "local_bounds_projection_gap_mw": float(local_gap),
        "ac_aware_projection_gap_mw": float(ac_gap),
        "ac_certified_projection_gap_mw": float(ac_certified_gap),
        "shield_intervention_gap_mw": float(shield_gap),
        "shield_intervention_penalty": float(penalty),
        "shield_intervention_count": float(shield_gap > 1e-9),
    }


def default_reward_contracts() -> tuple[AgentRewardContract, ...]:
    return (
        AgentRewardContract(
            agent_role="dso_global_guidance",
            objective="minimize grid security cost, procurement cost and infeasible envelope guidance",
            time_scale="fast_step",
            raw_global_dso_reward_weight=1.0,
            terms=(
                RewardTermContract("scaled_reward", "normalized and clipped grid reward components", "maximize", 1.0, ("dso",), "Learning reward; raw component values remain logged for audit."),
                RewardTermContract("post_ac_security_penalty", "post-dispatch pandapower AC constraint audit", "minimize", 1.0, ("dso",), "Voltage, line, transformer and powerflow violations after AC power flow."),
                RewardTermContract("shield_intervention_penalty", "raw DSO/VPP action repaired by local or AC-aware projection", "minimize", SHIELD_INTERVENTION_LINEAR_PENALTY_WEIGHT, ("dso",), "Learning-layer penalty so upper policies cannot rely on the safety shield as a free fallback."),
            ),
        ),
        AgentRewardContract(
            agent_role="vpp_dispatch",
            objective="maximize the VPP's own settlement/profit proxy while respecting the DSO envelope",
            time_scale="fast_step",
            raw_global_dso_reward_weight=DISPATCH_RAW_DSO_REWARD_WEIGHT,
            terms=(
                RewardTermContract("private_profit_proxy", "own VPP DER settlement", "maximize", DISPATCH_PRIVATE_PROFIT_WEIGHT, ("own_vpp",), "Primary private incentive."),
                RewardTermContract("target_tracking_penalty", "own envelope delivery audit", "minimize", DISPATCH_TRACKING_PENALTY_WEIGHT, ("own_vpp", "dso"), "Contract delivery signal, not raw global reward."),
                RewardTermContract("local_bounds_projection_penalty", "own local bounds projection audit", "minimize", DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT, ("own_vpp", "dso"), "Teaches local FR/DOE bounds feasibility; AC security is audited separately."),
                RewardTermContract("shield_intervention_penalty", "global local/AC-aware projection audit", "minimize", SHIELD_INTERVENTION_LINEAR_PENALTY_WEIGHT, ("own_vpp", "dso"), "Extra learning-layer cost when raw continuous actions are only feasible after shield repair."),
            ),
        ),
        AgentRewardContract(
            agent_role="vpp_portfolio",
            objective="adapt the VPP portfolio slowly using long-horizon profit, reliability and localized DSO-alignment signals",
            time_scale=f"slow_loop_every_{PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS}_steps_by_default",
            raw_global_dso_reward_weight=PORTFOLIO_RAW_DSO_REWARD_WEIGHT,
            terms=(
                RewardTermContract("long_horizon_profit_proxy", "own dispatch settlement", "maximize", 0.10, ("own_vpp",), "Aggregated private incentive."),
                RewardTermContract("localized_dso_alignment_reward", "settlement-compatible DSO service signals", "maximize", PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT, ("own_vpp", "dso"), "A localized variant, not the raw global DSO reward."),
                RewardTermContract("switching_cost", "own portfolio action", "minimize", 1.0, ("own_vpp",), "Prevents unrealistic frequent reconfiguration."),
            ),
        ),
    )


__all__ = [
    "AVAILABILITY_PAYMENT_RATE",
    "DISPATCH_COMFORT_SOC_PENALTY_WEIGHT",
    "DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT",
    "DISPATCH_PREFERRED_REGION_BONUS_WEIGHT",
    "DISPATCH_PRIVATE_PROFIT_WEIGHT",
    "DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT",
    "DISPATCH_RAW_DSO_REWARD_WEIGHT",
    "DISPATCH_TRACKING_PENALTY_WEIGHT",
    "FLEXIBILITY_SERVICE_PRICE_MULTIPLIER",
    "PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS",
    "PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT",
    "PORTFOLIO_RAW_DSO_REWARD_WEIGHT",
    "AgentRewardContract",
    "RewardTermContract",
    "SHIELD_INTERVENTION_LINEAR_PENALTY_WEIGHT",
    "SHIELD_INTERVENTION_QUADRATIC_PENALTY_WEIGHT",
    "default_reward_contracts",
    "shield_intervention_metrics",
]
