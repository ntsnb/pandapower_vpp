from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vpp_dso_sim.learning.reward_config import (
    DSORewardConfig,
    RewardConfig,
    ShieldRewardConfig,
    VPPDispatchRewardConfig,
    VPPPortfolioRewardConfig,
    VPPRewardConfig,
)
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
    PORTFOLIO_DECISION_MASK_DEFAULT_INTERVAL_STEPS,
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


ACTION_LANDING_DROP_REASON_CODES = {
    "landed": 0.0,
    "local_physical_limit": 1.0,
    "dso_envelope_clip": 2.0,
    "ac_shield_projection": 3.0,
    "baseline_override": 4.0,
    "not_applied_bug": 5.0,
    "logging_bug": 6.0,
}


def _action_landing_drop_reason_code(reason: str) -> float:
    return ACTION_LANDING_DROP_REASON_CODES.get(str(reason), ACTION_LANDING_DROP_REASON_CODES["not_applied_bug"])


def _classify_action_landing_drop(
    *,
    decoded_delta_p_mw: float,
    actual_delta_p_mw: float,
    raw_to_device_gap_mw: float,
    device_to_pre_ac_gap_mw: float,
    pre_ac_to_ac_gap_mw: float,
    ac_to_actual_gap_mw: float,
    accepted_to_actual_gap_mw: float,
    eps: float = 1e-9,
) -> str:
    if abs(decoded_delta_p_mw) <= eps or abs(actual_delta_p_mw) > eps:
        return "landed"
    if raw_to_device_gap_mw > eps:
        return "local_physical_limit"
    if device_to_pre_ac_gap_mw > eps:
        return "dso_envelope_clip"
    if pre_ac_to_ac_gap_mw > eps or ac_to_actual_gap_mw > eps:
        return "ac_shield_projection"
    if accepted_to_actual_gap_mw > eps:
        return "baseline_override"
    return "not_applied_bug"


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


def _preferred_bonus_gates(
    *,
    p_mw: float,
    envelope: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, float]:
    low = _safe_float(envelope.get("preferred_p_min_mw"), _safe_float(envelope.get("p_min_mw"), p_mw))
    high = _safe_float(envelope.get("preferred_p_max_mw"), _safe_float(envelope.get("p_max_mw"), p_mw))
    if low > high:
        low, high = high, low
    hard_low = _safe_float(envelope.get("p_min_mw"), low)
    hard_high = _safe_float(envelope.get("p_max_mw"), high)
    if hard_low > hard_high:
        hard_low, hard_high = hard_high, hard_low
    preferred_span = max(0.0, high - low)
    hard_span = max(0.0, hard_high - hard_low)
    inside = float(low <= p_mw <= high)
    lambda_gate = max(0.0, min(1.0, _safe_float(envelope.get("guidance_strength_lambda"), 1.0)))
    width_gate = max(0.0, min(1.0, 1.0 - preferred_span / hard_span)) if hard_span > 1e-9 else 0.0
    if "effective_response_score" in envelope:
        effectiveness_gate = max(0.0, min(1.0, _safe_float(envelope.get("effective_response_score"), 1.0)))
    else:
        projection_gap_mw = max(
            0.0,
            _safe_float(audit.get("local_bounds_projection_gap_mw"), _safe_float(audit.get("projection_gap_mw"), 0.0)),
        )
        effectiveness_gate = max(0.0, min(1.0, 1.0 - projection_gap_mw / max(1e-9, hard_span)))
    return {
        "preferred_inside_range": inside,
        "preferred_bonus_lambda_gate": lambda_gate,
        "preferred_bonus_width_gate": width_gate,
        "preferred_bonus_effectiveness_gate": effectiveness_gate,
        "preferred_region_score": _preferred_region_score(p_mw, envelope),
    }


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


def _portfolio_switching_cost_from_config(action: str, config: VPPPortfolioRewardConfig) -> float:
    normalized = str(action or "keep").lower()
    if normalized == "reweight":
        return float(config.switching_reweight_cost)
    if normalized == "propose_membership_change":
        return float(config.switching_membership_change_cost)
    return float(config.switching_keep_cost)


def _signed_clip(value: float, lower: float, upper: float) -> float:
    return min(float(upper), max(float(lower), float(value)))


def _battery_degradation_cost(vpp, *, dt_hours: float, price_per_mwh: float) -> float:
    total = 0.0
    for der in getattr(vpp, "der_list", []):
        if hasattr(der, "capacity_mwh") or "storage" in der.__class__.__name__.lower():
            total += abs(_safe_float(getattr(der, "p_mw", 0.0))) * float(dt_hours) * float(price_per_mwh)
    return float(total)


def _storage_potential_shaping_components(
    *,
    vpp,
    envelope: dict[str, Any],
    audit: dict[str, Any],
    dt_hours: float,
    config: VPPDispatchRewardConfig,
    current_price: float,
) -> dict[str, float]:
    if float(config.storage_potential_shaping_weight) == 0.0:
        return {
            "storage_potential_raw": 0.0,
            "storage_potential_shaping_reward": 0.0,
            "storage_value_spread_per_mwh": 0.0,
            "storage_charge_mwh": 0.0,
            "storage_discharge_mwh": 0.0,
            "storage_anti_hoarding_pass": 1.0,
        }

    forecast = audit.get("future_price_forecast", envelope.get("future_price_forecast", []))
    if not isinstance(forecast, (list, tuple)):
        forecast = []
    horizon = max(1, int(config.storage_future_value_window_steps))
    window = [_safe_float(price, current_price) for price in list(forecast)[:horizon]]
    future_price = max(window) if window else float(current_price)
    roundtrip_eff = float(config.storage_charge_efficiency) * float(config.storage_discharge_efficiency)
    value_spread = (
        float(config.storage_discount) * roundtrip_eff * future_price
        - float(current_price)
        - float(config.degradation_price_per_mwh)
    )
    charge_mwh = 0.0
    discharge_mwh = 0.0
    potential_raw = 0.0
    for der in getattr(vpp, "der_list", []):
        is_storage = hasattr(der, "capacity_mwh") or "storage" in der.__class__.__name__.lower()
        if not is_storage:
            continue
        p_mw = _safe_float(getattr(der, "p_mw", 0.0))
        if p_mw < 0.0:
            energy = -p_mw * float(dt_hours)
            charge_mwh += energy
            if value_spread > 0.0:
                potential_raw += value_spread * energy
            else:
                potential_raw += value_spread * energy
        elif p_mw > 0.0:
            energy = p_mw * float(dt_hours)
            discharge_mwh += energy
            potential_raw -= max(0.0, value_spread) * energy

    shaping = float(config.storage_potential_shaping_weight) * potential_raw
    anti_hoarding_pass = 1.0
    if charge_mwh > 1.0e-9 and value_spread <= 0.0 and shaping > 1.0e-9:
        anti_hoarding_pass = 0.0
    return {
        "storage_potential_raw": float(potential_raw),
        "storage_potential_shaping_reward": float(shaping),
        "storage_value_spread_per_mwh": float(value_spread),
        "storage_charge_mwh": float(charge_mwh),
        "storage_discharge_mwh": float(discharge_mwh),
        "storage_anti_hoarding_pass": float(anti_hoarding_pass),
    }


def attribute_projection_gaps(
    *,
    action_projection_gap_mw: float = 0.0,
    local_bounds_projection_gap_mw: float = 0.0,
    ac_aware_projection_gap_mw: float = 0.0,
    ac_certified_projection_gap_mw: float = 0.0,
    dispatch_audit: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    dispatch_responsibility: dict[str, float] = {}
    portfolio_responsibility: dict[str, float] = {}
    dispatch_total = 0.0
    for vpp_id, audit in dict(dispatch_audit or {}).items():
        gap = max(0.0, _safe_float(audit.get("local_bounds_projection_gap_mw"), audit.get("projection_gap_mw", 0.0)))
        scope = str(audit.get("projection_gap_scope", "")).lower()
        if gap > 0.0 and ("local" in scope or "der" in scope or "bounds" in scope):
            dispatch_responsibility[str(vpp_id)] = gap
            dispatch_total += gap
        else:
            dispatch_responsibility[str(vpp_id)] = 0.0
        portfolio_responsibility[str(vpp_id)] = max(0.0, _safe_float(audit.get("portfolio_projection_gap_mw"), 0.0))

    dso_gap = max(0.0, float(ac_aware_projection_gap_mw)) + max(0.0, float(ac_certified_projection_gap_mw))
    known = dispatch_total + dso_gap + sum(portfolio_responsibility.values())
    exogenous = max(0.0, max(0.0, float(action_projection_gap_mw)) - known)
    return {
        "dso_responsible_projection_gap_mw": float(dso_gap),
        "dispatch_responsible_projection_gap_mw": dispatch_responsibility,
        "portfolio_responsible_projection_gap_mw": portfolio_responsibility,
        "exogenous_projection_gap_mw": float(exogenous),
        "attribution_method": "heuristic_scope_and_ac_gap",
        "local_bounds_projection_gap_mw": max(0.0, float(local_bounds_projection_gap_mw)),
    }


def dso_reward_from_components(reward_components: dict[str, Any]) -> dict[str, float]:
    dso_reward_env = _safe_float(
        reward_components.get("dso_reward_env"),
        _safe_float(reward_components.get("dso_reward"), _safe_float(reward_components.get("reward"), -_safe_float(reward_components.get("total_cost")))),
    )
    dso_reward_train = _safe_float(
        reward_components.get("dso_reward_train"),
        dso_reward_env,
    )
    return {
        "reward_type_code": 0.0,
        "dso_reward": dso_reward_env,
        "dso_reward_env": dso_reward_env,
        "dso_reward_train": dso_reward_train,
        "dso_reward_critic_scaled": _safe_float(reward_components.get("dso_reward_critic_scaled")),
        "scaled_reward": _safe_float(reward_components.get("scaled_reward"), dso_reward_env),
        "scaled_total_cost": _safe_float(reward_components.get("scaled_total_cost")),
        "system_reward": dso_reward_env,
        "grid_security_cost": _safe_float(reward_components.get("total_cost")),
        "post_ac_violation_count": _safe_float(reward_components.get("post_ac_violation_count")),
        "post_ac_violation_magnitude": _safe_float(reward_components.get("post_ac_violation_magnitude")),
        "feasibility_bonus": _safe_float(reward_components.get("feasibility_bonus")),
        "tracking_bonus": _safe_float(reward_components.get("tracking_bonus")),
        "action_projection_penalty": _safe_float(reward_components.get("action_projection_penalty")),
        "local_bounds_projection_gap_mw": _safe_float(reward_components.get("local_bounds_projection_gap_mw")),
        "dso_responsible_projection_gap_mw": _safe_float(reward_components.get("dso_responsible_projection_gap_mw")),
        "dso_responsible_projection_penalty": _safe_float(reward_components.get("dso_responsible_projection_penalty")),
    }


def vpp_dispatch_reward_components(
    *,
    vpp,
    envelope: dict[str, Any],
    audit: dict[str, Any],
    dt_hours: float,
    t: int,
    reward_config: RewardConfig | None = None,
) -> dict[str, float]:
    """Fast self-interested VPP dispatch reward.

    This is intentionally local: it uses the owning VPP's delivered power,
    local operating cost, local comfort/SOC penalties, and the DSO signal
    addressed to this VPP. It does not include raw global DSO reward.
    """

    reward_config = reward_config or RewardConfig()
    dispatch_config = reward_config.vpp.dispatch
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

    baseline_source = "audit_baseline_p_mw"
    if "baseline_p_mw" in audit:
        baseline_p_mw = _safe_float(audit.get("baseline_p_mw"), delivered_p_mw)
    elif "baseline_p_mw" in envelope:
        baseline_p_mw = _safe_float(envelope.get("baseline_p_mw"), delivered_p_mw)
        baseline_source = "envelope_baseline_p_mw"
    elif "last_safe_p_mw" in audit:
        baseline_p_mw = _safe_float(audit.get("last_safe_p_mw"), delivered_p_mw)
        baseline_source = "last_safe_p_mw"
    else:
        baseline_p_mw = delivered_p_mw
        baseline_source = "current_delivered_fallback"
    requested_delta_p_mw = _safe_float(envelope.get("preferred_target_p_mw"), target_p_mw) - baseline_p_mw
    accepted_delta_p_mw = target_p_mw - baseline_p_mw
    actual_delta_p_mw = delivered_p_mw - baseline_p_mw
    raw_target_p_mw = _safe_float(audit.get("raw_target_p_mw"), target_p_mw)
    decoded_target_p_mw = _safe_float(
        audit.get("decoded_target_p_mw"),
        _safe_float(audit.get("dispatch_adjusted_target_p_mw"), raw_target_p_mw),
    )
    device_feasible_target_p_mw = _safe_float(
        audit.get("device_feasible_target_p_mw"),
        _safe_float(audit.get("projected_target_p_mw"), target_p_mw),
    )
    pre_ac_target_p_mw = _safe_float(audit.get("pre_ac_target_p_mw"), target_p_mw)
    ac_projected_target_p_mw = _safe_float(audit.get("ac_projected_target_p_mw"), pre_ac_target_p_mw)
    ac_certified_target_p_mw = _safe_float(audit.get("ac_certified_target_p_mw"), ac_projected_target_p_mw)
    actual_target_p_mw = _safe_float(audit.get("actual_target_p_mw"), delivered_p_mw)
    raw_delta_p_mw = raw_target_p_mw - baseline_p_mw
    decoded_delta_p_mw = decoded_target_p_mw - baseline_p_mw
    device_feasible_delta_p_mw = device_feasible_target_p_mw - baseline_p_mw
    pre_ac_delta_p_mw = pre_ac_target_p_mw - baseline_p_mw
    ac_projected_delta_p_mw = ac_projected_target_p_mw - baseline_p_mw
    ac_certified_delta_p_mw = ac_certified_target_p_mw - baseline_p_mw
    raw_to_device_gap_mw = _safe_float(
        audit.get("raw_to_device_gap_mw"),
        abs(device_feasible_target_p_mw - raw_target_p_mw),
    )
    device_to_pre_ac_gap_mw = abs(pre_ac_target_p_mw - device_feasible_target_p_mw)
    pre_ac_to_ac_gap_mw = abs(ac_projected_target_p_mw - pre_ac_target_p_mw)
    device_to_ac_gap_mw = _safe_float(
        audit.get("device_to_ac_gap_mw"),
        abs(ac_projected_target_p_mw - device_feasible_target_p_mw),
    )
    ac_to_actual_gap_mw = _safe_float(
        audit.get("ac_to_actual_gap_mw"),
        abs(actual_target_p_mw - ac_projected_target_p_mw),
    )
    accepted_to_actual_gap_mw = _safe_float(
        audit.get("accepted_to_actual_gap_mw"),
        abs(actual_delta_p_mw - accepted_delta_p_mw),
    )
    actual_delta_nonzero_flag = float(abs(actual_delta_p_mw) > 1e-9)
    action_landing_ratio = _safe_float(
        audit.get("action_landing_ratio"),
        abs(actual_delta_p_mw) / (abs(decoded_delta_p_mw) + 1e-9),
    )
    action_landing_drop_reason = str(
        audit.get(
            "action_landing_drop_reason",
            _classify_action_landing_drop(
                decoded_delta_p_mw=decoded_delta_p_mw,
                actual_delta_p_mw=actual_delta_p_mw,
                raw_to_device_gap_mw=raw_to_device_gap_mw,
                device_to_pre_ac_gap_mw=device_to_pre_ac_gap_mw,
                pre_ac_to_ac_gap_mw=pre_ac_to_ac_gap_mw,
                ac_to_actual_gap_mw=ac_to_actual_gap_mw,
                accepted_to_actual_gap_mw=accepted_to_actual_gap_mw,
            ),
        )
    )
    if abs(accepted_delta_p_mw) > 1e-9:
        directional_delivery = (1.0 if accepted_delta_p_mw > 0.0 else -1.0) * actual_delta_p_mw
        verified_delivery_mw = _signed_clip(directional_delivery, 0.0, abs(accepted_delta_p_mw))
    else:
        directional_delivery = 0.0
        verified_delivery_mw = 0.0
    contract_shortfall_mw = max(0.0, abs(accepted_delta_p_mw) - verified_delivery_mw)

    energy_market_revenue = price * delivered_p_mw * dt_hours
    if reward_config.is_v2_minimal and bool(dispatch_config.use_baseline_service_payment):
        flexibility_service_payment = price * verified_delivery_mw * dt_hours
        service_quantity_mw = verified_delivery_mw
    else:
        service_quantity_mw = _service_quantity_mw(delivered_p_mw, envelope)
        flexibility_service_payment = (
            FLEXIBILITY_SERVICE_PRICE_MULTIPLIER
            * price
            * service_quantity_mw
            * dt_hours
        )
    availability_payment = AVAILABILITY_PAYMENT_RATE * price * flex_span_mw * dt_hours
    der_operation_cost = float(vpp.operating_cost()) * dt_hours
    raw_comfort_soc_penalty = float(vpp.comfort_penalty(t) + vpp.soc_violation_penalty(t))
    scaled_comfort_soc_penalty = _bounded_penalty(raw_comfort_soc_penalty, scale=100.0, clip=5.0)
    target_tracking_penalty = dispatch_config.contract_delivery_weight * tracking_gap_mw * tracking_gap_mw
    envelope_projection_penalty = (
        dispatch_config.projection_linear_weight * projection_gap_mw
        + dispatch_config.projection_quadratic_weight * projection_gap_mw * projection_gap_mw
    )
    dispatch_responsible_gap = _safe_float(
        audit.get("dispatch_responsible_projection_gap_mw"),
        projection_gap_mw,
    )
    dispatch_projection_penalty = (
        dispatch_config.projection_linear_weight * dispatch_responsible_gap
        + dispatch_config.projection_quadratic_weight * dispatch_responsible_gap * dispatch_responsible_gap
    )
    preferred_gates = _preferred_bonus_gates(
        p_mw=delivered_p_mw,
        envelope=envelope,
        audit=audit,
    )
    preferred_region_bonus = (
        dispatch_config.preferred_region_bonus_weight
        * preferred_gates["preferred_inside_range"]
        * preferred_gates["preferred_bonus_lambda_gate"]
        * preferred_gates["preferred_bonus_width_gate"]
        * preferred_gates["preferred_bonus_effectiveness_gate"]
    )
    battery_degradation_cost = _battery_degradation_cost(
        vpp,
        dt_hours=dt_hours,
        price_per_mwh=dispatch_config.degradation_price_per_mwh,
    )
    service_quality_penalty_total = 0.0
    quality_adjusted_operational_surplus = 0.0

    if reward_config.is_v3_market_safety:
        settlement_complete = _safe_float(audit.get("settlement_audit_complete"), 0.0)
        settlement_power_balance_ok = _safe_float(audit.get("settlement_power_balance_ok"), 0.0)
        if (
            dispatch_config.require_per_der_settlement_audit
            and dispatch_config.paper_long_fail_on_incomplete_settlement_audit
            and (settlement_complete < 1.0 or settlement_power_balance_ok < 1.0)
        ):
            raise ValueError(
                "incomplete DER-level settlement audit for reward-v3 paper-long: "
                f"settlement_audit_complete={settlement_complete}, "
                f"settlement_power_balance_ok={settlement_power_balance_ok}"
            )
        operational_surplus = _safe_float(
            audit.get("economic_operational_surplus", audit.get("operational_surplus")),
            energy_market_revenue - der_operation_cost,
        )
        service_quality_penalty_total = _safe_float(
            audit.get(
                "service_quality_penalty_total",
                _safe_float(audit.get("comfort_cost_total")) + _safe_float(audit.get("unserved_penalty_total")),
            )
        )
        quality_adjusted_operational_surplus = _safe_float(
            audit.get("quality_adjusted_operational_surplus"),
            operational_surplus - service_quality_penalty_total,
        )
        if str(dispatch_config.service_payment_source).lower() == "disabled":
            flexibility_service_payment = 0.0
        else:
            flexibility_service_payment = _safe_float(audit.get("service_payment"), flexibility_service_payment)
        if str(dispatch_config.availability_payment_source).lower() == "disabled":
            availability_payment = 0.0
        else:
            availability_payment = _safe_float(audit.get("availability_payment"), availability_payment)
        if str(dispatch_config.contract_settlement_source).lower() == "disabled":
            contract_delivery_penalty = 0.0
        else:
            contract_delivery_penalty = _safe_float(audit.get("contract_penalty"), target_tracking_penalty)

        storage_components = _storage_potential_shaping_components(
            vpp=vpp,
            envelope=envelope,
            audit=audit,
            dt_hours=dt_hours,
            config=dispatch_config,
            current_price=price,
        )
        private_profit_proxy = (
            operational_surplus
            + float(dispatch_config.service_payment_weight) * flexibility_service_payment
            + float(dispatch_config.availability_payment_weight) * availability_payment
            - float(dispatch_config.contract_delivery_weight) * contract_delivery_penalty
        )
        reward = (
            dispatch_config.private_profit_weight * operational_surplus
            + float(dispatch_config.service_payment_weight) * flexibility_service_payment
            + float(dispatch_config.availability_payment_weight) * availability_payment
            + preferred_region_bonus
            + storage_components["storage_potential_shaping_reward"]
            - float(dispatch_config.contract_delivery_weight) * contract_delivery_penalty
            - dispatch_projection_penalty
            - dispatch_config.comfort_soc_weight * scaled_comfort_soc_penalty
            - dispatch_config.battery_degradation_weight * battery_degradation_cost
        )
    elif reward_config.is_v2_minimal:
        private_profit_proxy = energy_market_revenue - der_operation_cost
        contract_delivery_penalty = dispatch_config.contract_delivery_weight * contract_shortfall_mw * contract_shortfall_mw
        settlement_complete = 0.0
        settlement_power_balance_ok = 0.0
        operational_surplus = private_profit_proxy
        quality_adjusted_operational_surplus = operational_surplus
        storage_components = {
            "storage_potential_raw": 0.0,
            "storage_potential_shaping_reward": 0.0,
            "storage_value_spread_per_mwh": 0.0,
            "storage_charge_mwh": 0.0,
            "storage_discharge_mwh": 0.0,
            "storage_anti_hoarding_pass": 1.0,
        }
        reward = (
            dispatch_config.private_profit_weight * private_profit_proxy
            + dispatch_config.service_payment_weight * flexibility_service_payment
            + dispatch_config.availability_payment_weight * availability_payment
            + preferred_region_bonus
            - contract_delivery_penalty
            - dispatch_projection_penalty
            - dispatch_config.comfort_soc_weight * scaled_comfort_soc_penalty
            - dispatch_config.battery_degradation_weight * battery_degradation_cost
        )
    else:
        private_profit_proxy = (
            energy_market_revenue
            + flexibility_service_payment
            + availability_payment
            - der_operation_cost
        )
        contract_delivery_penalty = target_tracking_penalty
        settlement_complete = 0.0
        settlement_power_balance_ok = 0.0
        operational_surplus = private_profit_proxy
        quality_adjusted_operational_surplus = operational_surplus
        storage_components = {
            "storage_potential_raw": 0.0,
            "storage_potential_shaping_reward": 0.0,
            "storage_value_spread_per_mwh": 0.0,
            "storage_charge_mwh": 0.0,
            "storage_discharge_mwh": 0.0,
            "storage_anti_hoarding_pass": 1.0,
        }
        reward = (
            dispatch_config.private_profit_weight * private_profit_proxy
            + preferred_region_bonus
            - target_tracking_penalty
            - envelope_projection_penalty
            - dispatch_config.comfort_soc_weight * scaled_comfort_soc_penalty
        )
    visible_energy_minus_operation_cost = float(energy_market_revenue - der_operation_cost)
    if reward_config.is_v3_market_safety:
        market_energy_margin_total = float(
            _safe_float(audit.get("export_revenue_total"))
            + _safe_float(audit.get("evcs_user_revenue_total"))
            - _safe_float(audit.get("import_energy_cost_total"))
        )
    else:
        market_energy_margin_total = visible_energy_minus_operation_cost
    return {
        "reward_type_code": 1.0,
        "vpp_dispatch_reward": float(reward),
        "vpp_dispatch_reward_env": float(reward),
        "private_profit_proxy": float(private_profit_proxy),
        "vpp_operational_surplus_ex_transfer": float(operational_surplus),
        "economic_operational_surplus": float(operational_surplus),
        "quality_adjusted_operational_surplus": float(quality_adjusted_operational_surplus),
        "service_quality_penalty_total": float(service_quality_penalty_total),
        "settlement_audit_complete": float(settlement_complete),
        "settlement_power_balance_ok": float(settlement_power_balance_ok),
        "settlement_power_balance_error_mw": _safe_float(audit.get("power_balance_error_mw"), 0.0),
        "settlement_audit_required": 1.0 if dispatch_config.require_per_der_settlement_audit else 0.0,
        "dispatch_private_profit_reward": float(
            dispatch_config.private_profit_weight * private_profit_proxy
        ),
        "market_price": float(price),
        "dt_hours": float(dt_hours),
        "energy_market_revenue": float(energy_market_revenue),
        "visible_energy_minus_operation_cost": visible_energy_minus_operation_cost,
        "market_energy_margin_total": market_energy_margin_total,
        "flexibility_service_payment": float(flexibility_service_payment),
        "service_payment": float(flexibility_service_payment),
        "service_quantity_mw": float(service_quantity_mw),
        "flexibility_service_price_multiplier": float(FLEXIBILITY_SERVICE_PRICE_MULTIPLIER),
        "availability_payment": float(availability_payment),
        "availability_payment_rate": float(AVAILABILITY_PAYMENT_RATE),
        "raw_dso_reward_shared": float(dispatch_config.raw_dso_reward_weight),
        "export_revenue_total": _safe_float(audit.get("export_revenue_total")),
        "pv_export_revenue_total": _safe_float(audit.get("pv_export_revenue_total")),
        "mt_export_revenue_total": _safe_float(audit.get("mt_export_revenue_total")),
        "storage_discharge_revenue_total": _safe_float(audit.get("storage_discharge_revenue_total")),
        "evcs_user_revenue_total": _safe_float(audit.get("evcs_user_revenue_total")),
        "import_energy_cost_total": _safe_float(audit.get("import_energy_cost_total")),
        "evcs_wholesale_cost_total": _safe_float(audit.get("evcs_wholesale_cost_total")),
        "storage_charge_cost_total": _safe_float(audit.get("storage_charge_cost_total")),
        "hvac_energy_cost_total": _safe_float(audit.get("hvac_energy_cost_total")),
        "flex_energy_cost_total": _safe_float(audit.get("flex_energy_cost_total")),
        "unclassified_import_cost_total": _safe_float(audit.get("unclassified_import_cost_total")),
        "der_operating_cost_total": _safe_float(audit.get("der_operating_cost_total")),
        "battery_degradation_cost_total": _safe_float(audit.get("battery_degradation_cost_total")),
        "comfort_cost_total": _safe_float(audit.get("comfort_cost_total")),
        "unserved_penalty_total": _safe_float(audit.get("unserved_penalty_total")),
        "legacy_operational_surplus_with_service_quality": _safe_float(
            audit.get("legacy_operational_surplus_with_service_quality"),
            _safe_float(audit.get("quality_adjusted_operational_surplus"), operational_surplus),
        ),
        "der_operation_cost": float(der_operation_cost),
        "target_tracking_penalty": float(target_tracking_penalty),
        "envelope_projection_penalty": float(envelope_projection_penalty),
        "dispatch_projection_penalty": float(dispatch_projection_penalty),
        "dispatch_responsible_projection_gap_mw": float(dispatch_responsible_gap),
        "raw_comfort_soc_penalty": float(raw_comfort_soc_penalty),
        "scaled_comfort_soc_penalty": float(scaled_comfort_soc_penalty),
        "comfort_soc_penalty": float(scaled_comfort_soc_penalty),
        "comfort_penalty": float(vpp.comfort_penalty(t)),
        "soc_penalty": float(vpp.soc_violation_penalty(t)),
        "comfort_soc_penalty_scale": 100.0,
        "comfort_soc_penalty_clip": 5.0,
        "preferred_region_bonus": float(preferred_region_bonus),
        **{key: float(value) for key, value in preferred_gates.items()},
        "private_profit_weight": float(dispatch_config.private_profit_weight),
        "service_payment_weight": float(dispatch_config.service_payment_weight),
        "availability_payment_weight": float(dispatch_config.availability_payment_weight),
        "contract_delivery_weight": float(dispatch_config.contract_delivery_weight),
        "projection_linear_weight": float(dispatch_config.projection_linear_weight),
        "projection_quadratic_weight": float(dispatch_config.projection_quadratic_weight),
        "comfort_soc_weight": float(dispatch_config.comfort_soc_weight),
        "battery_degradation_weight": float(dispatch_config.battery_degradation_weight),
        "storage_potential_shaping_weight": float(dispatch_config.storage_potential_shaping_weight),
        "baseline_p_mw": float(baseline_p_mw),
        "baseline_source_code": float({
            "audit_baseline_p_mw": 0,
            "envelope_baseline_p_mw": 1,
            "last_safe_p_mw": 2,
            "current_delivered_fallback": 3,
        }.get(baseline_source, 9)),
        "raw_action_norm": _safe_float(audit.get("raw_action_norm"), 0.0),
        "raw_target_p_mw": float(raw_target_p_mw),
        "decoded_target_p_mw": float(decoded_target_p_mw),
        "device_feasible_target_p_mw": float(device_feasible_target_p_mw),
        "pre_ac_target_p_mw": float(pre_ac_target_p_mw),
        "ac_projected_target_p_mw": float(ac_projected_target_p_mw),
        "ac_certified_target_p_mw": float(ac_certified_target_p_mw),
        "actual_target_p_mw": float(actual_target_p_mw),
        "raw_delta_p_mw": float(raw_delta_p_mw),
        "decoded_delta_p_mw": float(decoded_delta_p_mw),
        "device_feasible_delta_p_mw": float(device_feasible_delta_p_mw),
        "pre_ac_delta_p_mw": float(pre_ac_delta_p_mw),
        "ac_projected_delta_p_mw": float(ac_projected_delta_p_mw),
        "ac_certified_delta_p_mw": float(ac_certified_delta_p_mw),
        "raw_to_device_gap_mw": float(raw_to_device_gap_mw),
        "device_to_ac_gap_mw": float(device_to_ac_gap_mw),
        "ac_to_actual_gap_mw": float(ac_to_actual_gap_mw),
        "accepted_to_actual_gap_mw": float(accepted_to_actual_gap_mw),
        "actual_delta_nonzero_flag": float(actual_delta_nonzero_flag),
        "action_landing_ratio": float(action_landing_ratio),
        "action_landing_drop_reason": action_landing_drop_reason,
        "action_landing_drop_reason_code": _action_landing_drop_reason_code(action_landing_drop_reason),
        "requested_delta_p_mw": float(requested_delta_p_mw),
        "accepted_delta_p_mw": float(accepted_delta_p_mw),
        "actual_delta_p_mw": float(actual_delta_p_mw),
        "directional_delivery_mw": float(directional_delivery),
        "verified_delivery_mw": float(verified_delivery_mw),
        "contract_shortfall_mw": float(contract_shortfall_mw),
        "contract_delivery_penalty": float(contract_delivery_penalty),
        "battery_degradation_cost": float(battery_degradation_cost),
        **storage_components,
        "delivered_p_mw": delivered_p_mw,
        "target_p_mw": target_p_mw,
        "flex_span_mw": flex_span_mw,
    }


@dataclass
class PortfolioWindowTracker:
    reward_config: RewardConfig
    stats: dict[str, dict[str, float]] = field(default_factory=dict)

    def reset(self) -> None:
        self.stats.clear()

    def _window(self, vpp_id: str) -> dict[str, float]:
        if vpp_id not in self.stats:
            self.stats[vpp_id] = {
                "sum_private_profit": 0.0,
                "sum_contract_shortfall": 0.0,
                "sum_shield_intervention": 0.0,
                "sum_projection_gap": 0.0,
                "sum_comfort_soc_penalty": 0.0,
                "sum_verified_capacity": 0.0,
                "n_steps": 0.0,
            }
        return self.stats[vpp_id]

    def update(
        self,
        vpp_id: str,
        dispatch_components: dict[str, float],
        *,
        shield_intervention_gap_mw: float = 0.0,
    ) -> None:
        window = self._window(vpp_id)
        window["sum_private_profit"] += _safe_float(dispatch_components.get("private_profit_proxy"))
        window["sum_contract_shortfall"] += _safe_float(dispatch_components.get("contract_shortfall_mw"))
        window["sum_shield_intervention"] += max(0.0, float(shield_intervention_gap_mw))
        window["sum_projection_gap"] += _safe_float(dispatch_components.get("dispatch_responsible_projection_gap_mw"))
        window["sum_comfort_soc_penalty"] += _safe_float(dispatch_components.get("scaled_comfort_soc_penalty"))
        window["sum_verified_capacity"] += _safe_float(dispatch_components.get("verified_delivery_mw"))
        window["n_steps"] += 1.0

    def settle_if_decision(self, vpp_id: str, *, step: int, action: str) -> dict[str, float]:
        config = self.reward_config.vpp.portfolio
        is_decision = int(step) % max(1, int(config.decision_interval_steps)) == 0
        window = self._window(vpp_id)
        if not is_decision:
            return self._empty_result(action=action, decision_step=False)

        n_steps = max(1.0, window["n_steps"])
        mean_profit = window["sum_private_profit"] / n_steps
        mean_shortfall = window["sum_contract_shortfall"] / n_steps
        mean_shield = window["sum_shield_intervention"] / n_steps
        mean_projection = window["sum_projection_gap"] / n_steps
        mean_comfort_soc = window["sum_comfort_soc_penalty"] / n_steps
        mean_verified_capacity = window["sum_verified_capacity"] / n_steps
        switching_cost = _portfolio_switching_cost_from_config(action, config)
        reward = (
            float(config.long_horizon_profit_weight) * mean_profit
            + float(config.verified_capacity_weight) * mean_verified_capacity
            - float(config.delivery_reliability_weight) * mean_shortfall
            - float(config.future_shield_penalty_weight) * mean_shield
            - float(config.future_projection_penalty_weight) * mean_projection
            - float(config.future_comfort_soc_weight) * mean_comfort_soc
            - switching_cost
        )
        result = {
            "reward_type_code": 2.0,
            "vpp_portfolio_reward": float(reward),
            "vpp_portfolio_reward_env": float(reward),
            "portfolio_decision_step": 1.0,
            "portfolio_window_profit": float(mean_profit),
            "portfolio_window_contract_shortfall": float(mean_shortfall),
            "portfolio_window_shield_intervention": float(mean_shield),
            "portfolio_window_projection_gap": float(mean_projection),
            "portfolio_window_comfort_soc_penalty": float(mean_comfort_soc),
            "portfolio_window_verified_capacity": float(mean_verified_capacity),
            "portfolio_switching_cost": float(switching_cost),
            "portfolio_action_type_code": float({"keep": 0, "reweight": 1, "propose_membership_change": 2}.get(str(action), 9)),
            "raw_dso_reward_shared": float(config.raw_dso_reward_weight),
        }
        self.stats[vpp_id] = {
            "sum_private_profit": 0.0,
            "sum_contract_shortfall": 0.0,
            "sum_shield_intervention": 0.0,
            "sum_projection_gap": 0.0,
            "sum_comfort_soc_penalty": 0.0,
            "sum_verified_capacity": 0.0,
            "n_steps": 0.0,
        }
        return result

    def _empty_result(self, *, action: str, decision_step: bool) -> dict[str, float]:
        return {
            "reward_type_code": 2.0,
            "vpp_portfolio_reward": 0.0,
            "vpp_portfolio_reward_env": 0.0,
            "portfolio_decision_step": 1.0 if decision_step else 0.0,
            "portfolio_window_profit": 0.0,
            "portfolio_window_contract_shortfall": 0.0,
            "portfolio_window_shield_intervention": 0.0,
            "portfolio_window_projection_gap": 0.0,
            "portfolio_window_comfort_soc_penalty": 0.0,
            "portfolio_window_verified_capacity": 0.0,
            "portfolio_switching_cost": _portfolio_switching_cost_from_config(action, self.reward_config.vpp.portfolio),
            "portfolio_action_type_code": float({"keep": 0, "reweight": 1, "propose_membership_change": 2}.get(str(action), 9)),
            "raw_dso_reward_shared": float(self.reward_config.vpp.portfolio.raw_dso_reward_weight),
        }


def vpp_portfolio_reward_components(
    *,
    vpp,
    envelope: dict[str, Any],
    audit: dict[str, Any],
    dso_components: dict[str, Any],
    dispatch_components: dict[str, float],
    action: str,
    reward_config: RewardConfig | None = None,
) -> dict[str, float]:
    """Slow VPP portfolio reward with localized DSO-intent feedback.

    The portfolio agent receives a long-horizon proxy, not the raw DSO global
    reward. The DSO term is localized into service alignment, reliability and
    availability signals that can plausibly be settled or contracted.
    """

    reward_config = reward_config or RewardConfig()
    portfolio_config = reward_config.vpp.portfolio
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
        portfolio_config.localized_dso_alignment_weight
        * 0.35
        * preferred_score
        + 0.25 * _safe_float(dso_components.get("feasibility_bonus"))
        + 0.15 * availability_quality
        + 0.10 * reliability_bonus
        - 0.001 * network_penalty
    )
    switching_cost = _portfolio_switching_cost_from_config(action, portfolio_config)
    delivery_risk_penalty = 0.50 * projection_gap_mw + 0.20 * tracking_gap_mw
    long_horizon_profit_proxy = (
        float(portfolio_config.long_horizon_profit_weight)
        * _safe_float(dispatch_components.get("private_profit_proxy"))
    )
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
        "global_reward_variant_weight": float(portfolio_config.localized_dso_alignment_weight),
        "raw_dso_reward_shared": float(portfolio_config.raw_dso_reward_weight),
        "portfolio_switching_cost": float(switching_cost),
        "portfolio_action_type_code": float({"keep": 0, "reweight": 1, "propose_membership_change": 2}.get(str(action), 9)),
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
    reward_config: RewardConfig | None = None,
    portfolio_tracker: PortfolioWindowTracker | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    rewards: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    reward_config = reward_config or RewardConfig()

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
            reward_config=reward_config,
        )
        dispatch_agent = f"{vpp.id}_dispatch"
        rewards[dispatch_agent] = dispatch["vpp_dispatch_reward"]
        components[dispatch_agent] = dispatch

        portfolio_action = portfolio_actions_by_vpp.get(vpp.id, "keep")
        if reward_config.is_v2_minimal and reward_config.vpp.portfolio.mode == "window_return":
            if portfolio_tracker is None:
                portfolio_tracker = PortfolioWindowTracker(reward_config)
            portfolio_tracker.update(
                vpp.id,
                dispatch,
                shield_intervention_gap_mw=_safe_float(
                    dso_components.get("shield_intervention_gap_mw"),
                    _safe_float(dso_components.get("action_projection_gap_mw"), 0.0),
                ),
            )
            portfolio = portfolio_tracker.settle_if_decision(vpp.id, step=t, action=portfolio_action)
        else:
            portfolio = vpp_portfolio_reward_components(
                vpp=vpp,
                envelope=envelope,
                audit=audit,
                dso_components=dso_components,
                dispatch_components=dispatch,
                action=portfolio_action,
                reward_config=reward_config,
            )
        portfolio_agent = f"{vpp.id}_portfolio"
        rewards[portfolio_agent] = portfolio["vpp_portfolio_reward"]
        components[portfolio_agent] = portfolio

    return rewards, components
