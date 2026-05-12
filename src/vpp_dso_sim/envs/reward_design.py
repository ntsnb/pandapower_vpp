from __future__ import annotations

from typing import Any

from vpp_dso_sim.learning.reward_contracts import (
    AVAILABILITY_PAYMENT_RATE,
    DISPATCH_COMFORT_SOC_PENALTY_WEIGHT,
    DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT,
    DISPATCH_PREFERRED_REGION_BONUS_WEIGHT,
    DISPATCH_PRIVATE_PROFIT_WEIGHT,
    DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT,
    DISPATCH_RAW_DSO_REWARD_WEIGHT,
    DISPATCH_TRACKING_PENALTY_WEIGHT,
    FLEXIBILITY_SERVICE_PRICE_MULTIPLIER,
    PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT,
    PORTFOLIO_RAW_DSO_REWARD_WEIGHT,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bounded_penalty(value: float, scale: float, clip: float) -> float:
    return min(float(clip), max(0.0, float(value)) / max(1e-9, float(scale)))


def _preferred_region_score(p_mw: float, envelope: dict[str, Any]) -> float:
    low = _safe_float(envelope.get("preferred_p_min_mw"), _safe_float(envelope.get("p_min_mw"), p_mw))
    high = _safe_float(envelope.get("preferred_p_max_mw"), _safe_float(envelope.get("p_max_mw"), p_mw))
    if low > high:
        low, high = high, low
    span = max(1e-6, high - low)
    if low <= p_mw <= high:
        return 1.0
    distance = min(abs(p_mw - low), abs(p_mw - high))
    return max(0.0, 1.0 - distance / span)


def _service_quantity_mw(p_mw: float, envelope: dict[str, Any]) -> float:
    request = str(envelope.get("service_request", "")).lower()
    if "absorb" in request or "charge" in request or "import" in request:
        return max(0.0, -p_mw)
    if "export" in request or "reduce" in request or "inject" in request:
        return max(0.0, p_mw)
    return abs(p_mw)


def _portfolio_switching_cost(action: str) -> float:
    normalized = str(action or "keep").lower()
    if normalized == "reweight":
        return 0.02
    if normalized == "propose_membership_change":
        return 0.08
    return 0.0


def dso_reward_from_components(reward_components: dict[str, Any]) -> dict[str, float]:
    dso_reward = _safe_float(
        reward_components.get("dso_reward"),
        _safe_float(reward_components.get("reward"), -_safe_float(reward_components.get("total_cost"))),
    )
    return {
        "reward_type_code": 0.0,
        "dso_reward": dso_reward,
        "scaled_reward": _safe_float(reward_components.get("scaled_reward"), dso_reward),
        "scaled_total_cost": _safe_float(reward_components.get("scaled_total_cost")),
        "system_reward": dso_reward,
        "grid_security_cost": _safe_float(reward_components.get("total_cost")),
        "post_ac_violation_count": _safe_float(reward_components.get("post_ac_violation_count")),
        "post_ac_violation_magnitude": _safe_float(reward_components.get("post_ac_violation_magnitude")),
        "feasibility_bonus": _safe_float(reward_components.get("feasibility_bonus")),
        "tracking_bonus": _safe_float(reward_components.get("tracking_bonus")),
        "action_projection_penalty": _safe_float(reward_components.get("action_projection_penalty")),
        "local_bounds_projection_gap_mw": _safe_float(reward_components.get("local_bounds_projection_gap_mw")),
    }


def vpp_dispatch_reward_components(
    *,
    vpp,
    envelope: dict[str, Any],
    audit: dict[str, Any],
    dt_hours: float,
    t: int,
) -> dict[str, float]:
    """Fast self-interested VPP dispatch reward.

    This is intentionally local: it uses the owning VPP's delivered power,
    local operating cost, local comfort/SOC penalties, and the DSO signal
    addressed to this VPP. It does not include raw global DSO reward.
    """

    price = _safe_float(envelope.get("price"), 80.0)
    delivered_p_mw = float(vpp.current_power_mw())
    target_p_mw = _safe_float(
        audit.get("projected_target_p_mw"),
        _safe_float(envelope.get("preferred_target_p_mw"), delivered_p_mw),
    )
    tracking_gap_mw = abs(delivered_p_mw - target_p_mw)
    projection_gap_mw = _safe_float(
        audit.get("local_bounds_projection_gap_mw"),
        _safe_float(audit.get("projection_gap_mw"), 0.0),
    )
    p_min = _safe_float(envelope.get("p_min_mw"), delivered_p_mw)
    p_max = _safe_float(envelope.get("p_max_mw"), delivered_p_mw)
    flex_span_mw = max(0.0, p_max - p_min)

    energy_market_revenue = price * delivered_p_mw * dt_hours
    flexibility_service_payment = (
        FLEXIBILITY_SERVICE_PRICE_MULTIPLIER
        * price
        * _service_quantity_mw(delivered_p_mw, envelope)
        * dt_hours
    )
    availability_payment = AVAILABILITY_PAYMENT_RATE * price * flex_span_mw * dt_hours
    der_operation_cost = float(vpp.operating_cost()) * dt_hours
    raw_comfort_soc_penalty = float(vpp.comfort_penalty(t) + vpp.soc_violation_penalty(t))
    scaled_comfort_soc_penalty = _bounded_penalty(raw_comfort_soc_penalty, scale=100.0, clip=5.0)
    target_tracking_penalty = DISPATCH_TRACKING_PENALTY_WEIGHT * tracking_gap_mw * tracking_gap_mw
    envelope_projection_penalty = (
        DISPATCH_LINEAR_PROJECTION_PENALTY_WEIGHT * projection_gap_mw
        + DISPATCH_QUADRATIC_PROJECTION_PENALTY_WEIGHT * projection_gap_mw * projection_gap_mw
    )
    preferred_region_bonus = DISPATCH_PREFERRED_REGION_BONUS_WEIGHT * _preferred_region_score(delivered_p_mw, envelope)

    private_profit_proxy = (
        energy_market_revenue
        + flexibility_service_payment
        + availability_payment
        - der_operation_cost
    )
    reward = (
        DISPATCH_PRIVATE_PROFIT_WEIGHT * private_profit_proxy
        + preferred_region_bonus
        - target_tracking_penalty
        - envelope_projection_penalty
        - DISPATCH_COMFORT_SOC_PENALTY_WEIGHT * scaled_comfort_soc_penalty
    )
    return {
        "reward_type_code": 1.0,
        "vpp_dispatch_reward": float(reward),
        "private_profit_proxy": float(private_profit_proxy),
        "energy_market_revenue": float(energy_market_revenue),
        "flexibility_service_payment": float(flexibility_service_payment),
        "flexibility_service_price_multiplier": float(FLEXIBILITY_SERVICE_PRICE_MULTIPLIER),
        "availability_payment": float(availability_payment),
        "availability_payment_rate": float(AVAILABILITY_PAYMENT_RATE),
        "raw_dso_reward_shared": float(DISPATCH_RAW_DSO_REWARD_WEIGHT),
        "der_operation_cost": float(der_operation_cost),
        "target_tracking_penalty": float(target_tracking_penalty),
        "envelope_projection_penalty": float(envelope_projection_penalty),
        "raw_comfort_soc_penalty": float(raw_comfort_soc_penalty),
        "scaled_comfort_soc_penalty": float(scaled_comfort_soc_penalty),
        "comfort_soc_penalty": float(scaled_comfort_soc_penalty),
        "comfort_soc_penalty_scale": 100.0,
        "comfort_soc_penalty_clip": 5.0,
        "preferred_region_bonus": float(preferred_region_bonus),
        "private_profit_weight": float(DISPATCH_PRIVATE_PROFIT_WEIGHT),
        "delivered_p_mw": delivered_p_mw,
        "target_p_mw": target_p_mw,
        "flex_span_mw": flex_span_mw,
    }


def vpp_portfolio_reward_components(
    *,
    vpp,
    envelope: dict[str, Any],
    audit: dict[str, Any],
    dso_components: dict[str, Any],
    dispatch_components: dict[str, float],
    action: str,
) -> dict[str, float]:
    """Slow VPP portfolio reward with localized DSO-intent feedback.

    The portfolio agent receives a long-horizon proxy, not the raw DSO global
    reward. The DSO term is localized into service alignment, reliability and
    availability signals that can plausibly be settled or contracted.
    """

    delivered_p_mw = _safe_float(dispatch_components.get("delivered_p_mw"))
    tracking_gap_mw = abs(delivered_p_mw - _safe_float(dispatch_components.get("target_p_mw"), delivered_p_mw))
    projection_gap_mw = _safe_float(
        audit.get("local_bounds_projection_gap_mw"),
        _safe_float(audit.get("projection_gap_mw"), 0.0),
    )
    flex_span_mw = _safe_float(dispatch_components.get("flex_span_mw"), 0.0)
    preferred_score = _preferred_region_score(delivered_p_mw, envelope)
    reliability_bonus = 0.50 * max(0.0, 1.0 - tracking_gap_mw / max(1e-6, flex_span_mw))
    availability_quality = min(1.0, flex_span_mw / 0.50)

    network_penalty = (
        _safe_float(dso_components.get("voltage_violation_penalty"))
        + _safe_float(dso_components.get("line_overload_penalty"))
        + _safe_float(dso_components.get("transformer_overload_penalty"))
        + _safe_float(dso_components.get("powerflow_penalty"))
        + _safe_float(dso_components.get("post_ac_violation_magnitude_penalty"))
    )
    localized_dso_alignment_reward = (
        PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT
        * 0.35
        * preferred_score
        + 0.25 * _safe_float(dso_components.get("feasibility_bonus"))
        + 0.15 * availability_quality
        + 0.10 * reliability_bonus
        - 0.001 * network_penalty
    )
    switching_cost = _portfolio_switching_cost(action)
    delivery_risk_penalty = 0.50 * projection_gap_mw + 0.20 * tracking_gap_mw
    long_horizon_profit_proxy = 0.10 * _safe_float(dispatch_components.get("private_profit_proxy"))
    reward = (
        long_horizon_profit_proxy
        + localized_dso_alignment_reward
        + reliability_bonus
        - switching_cost
        - delivery_risk_penalty
    )
    return {
        "reward_type_code": 2.0,
        "vpp_portfolio_reward": float(reward),
        "long_horizon_profit_proxy": float(long_horizon_profit_proxy),
        "localized_dso_alignment_reward": float(localized_dso_alignment_reward),
        "reliability_bonus": float(reliability_bonus),
        "availability_quality": float(availability_quality),
        "switching_cost": float(switching_cost),
        "delivery_risk_penalty": float(delivery_risk_penalty),
        "global_reward_variant_weight": float(PORTFOLIO_LOCALIZED_DSO_ALIGNMENT_WEIGHT),
        "raw_dso_reward_shared": float(PORTFOLIO_RAW_DSO_REWARD_WEIGHT),
    }


def build_role_reward_maps(
    *,
    vpps,
    envelopes_by_vpp: dict[str, dict[str, Any]],
    dispatch_audit: dict[str, dict[str, Any]],
    portfolio_actions_by_vpp: dict[str, str],
    dso_components: dict[str, Any],
    dt_hours: float,
    t: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rewards: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}

    dso_components_local = dso_reward_from_components(dso_components)
    rewards["dso_global_guidance"] = dso_components_local["dso_reward"]
    components["dso_global_guidance"] = dso_components_local

    for vpp in vpps:
        envelope = envelopes_by_vpp.get(vpp.id, {})
        audit = dispatch_audit.get(vpp.id, {})
        dispatch = vpp_dispatch_reward_components(
            vpp=vpp,
            envelope=envelope,
            audit=audit,
            dt_hours=dt_hours,
            t=t,
        )
        dispatch_agent = f"{vpp.id}_dispatch"
        rewards[dispatch_agent] = dispatch["vpp_dispatch_reward"]
        components[dispatch_agent] = dispatch

        portfolio_action = portfolio_actions_by_vpp.get(vpp.id, "keep")
        portfolio = vpp_portfolio_reward_components(
            vpp=vpp,
            envelope=envelope,
            audit=audit,
            dso_components=dso_components,
            dispatch_components=dispatch,
            action=portfolio_action,
        )
        portfolio_agent = f"{vpp.id}_portfolio"
        rewards[portfolio_agent] = portfolio["vpp_portfolio_reward"]
        components[portfolio_agent] = portfolio

    return rewards, components
