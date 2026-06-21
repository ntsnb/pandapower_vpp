from __future__ import annotations

from typing import Any


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def dispatch_private_profit_trace_rows(
    *,
    episode: int,
    step: int,
    algorithm: str,
    vpp_ids: list[str],
    dispatch_components: list[dict[str, Any]],
    raw_dispatch_rewards: Any | None = None,
    train_dispatch_rewards: Any | None = None,
) -> list[dict[str, Any]]:
    """Expand dispatch reward components into per-agent private-profit traces."""

    rows: list[dict[str, Any]] = []
    settlement_keys = (
        "economic_operational_surplus",
        "quality_adjusted_operational_surplus",
        "service_quality_penalty_total",
        "export_revenue_total",
        "pv_export_revenue_total",
        "mt_export_revenue_total",
        "storage_discharge_revenue_total",
        "evcs_user_revenue_total",
        "import_energy_cost_total",
        "evcs_wholesale_cost_total",
        "storage_charge_cost_total",
        "hvac_energy_cost_total",
        "flex_energy_cost_total",
        "unclassified_import_cost_total",
        "der_operating_cost_total",
        "battery_degradation_cost_total",
        "comfort_cost_total",
        "unserved_penalty_total",
        "legacy_operational_surplus_with_service_quality",
    )
    action_landing_keys = (
        "raw_action_norm",
        "raw_target_p_mw",
        "decoded_target_p_mw",
        "device_feasible_target_p_mw",
        "pre_ac_target_p_mw",
        "ac_projected_target_p_mw",
        "ac_certified_target_p_mw",
        "actual_target_p_mw",
        "raw_delta_p_mw",
        "decoded_delta_p_mw",
        "device_feasible_delta_p_mw",
        "pre_ac_delta_p_mw",
        "ac_projected_delta_p_mw",
        "ac_certified_delta_p_mw",
        "raw_to_device_gap_mw",
        "device_to_ac_gap_mw",
        "ac_to_actual_gap_mw",
        "accepted_to_actual_gap_mw",
        "actual_delta_nonzero_flag",
        "action_landing_ratio",
        "action_landing_drop_reason_code",
    )
    storage_shaping_keys = (
        "storage_potential_raw",
        "storage_potential_shaping_reward",
        "storage_potential_shaping_weight",
        "storage_value_spread_per_mwh",
        "storage_charge_mwh",
        "storage_discharge_mwh",
        "storage_anti_hoarding_pass",
    )
    for index, vpp_id in enumerate(vpp_ids):
        component = dispatch_components[index] if index < len(dispatch_components) else {}
        private_profit_proxy = _safe_float(component.get("private_profit_proxy"))
        private_profit_weight = _safe_float(component.get("private_profit_weight"))
        service_payment_weight = _safe_float(component.get("service_payment_weight"))
        availability_payment_weight = _safe_float(component.get("availability_payment_weight"))
        contract_delivery_weight = _safe_float(component.get("contract_delivery_weight"))
        comfort_soc_weight = _safe_float(component.get("comfort_soc_weight"))
        battery_degradation_weight = _safe_float(component.get("battery_degradation_weight"))
        market_price = _safe_float(component.get("market_price"), _safe_float(component.get("price")))
        delivered_p_mw = _safe_float(component.get("delivered_p_mw"))
        dt_hours = _safe_float(component.get("dt_hours"))
        energy_market_revenue = _safe_float(component.get("energy_market_revenue"))
        der_operation_cost = _safe_float(component.get("der_operation_cost"))
        visible_energy_minus_operation = energy_market_revenue - der_operation_cost
        settlement_breakdown = {key: _safe_float(component.get(key)) for key in settlement_keys}
        market_energy_margin_total = (
            settlement_breakdown["export_revenue_total"]
            + settlement_breakdown["evcs_user_revenue_total"]
            - settlement_breakdown["import_energy_cost_total"]
        )
        contract_delivery_penalty = _safe_float(component.get("contract_delivery_penalty"))
        dispatch_projection_penalty = _safe_float(component.get("dispatch_projection_penalty"))
        scaled_comfort_soc_penalty = _safe_float(component.get("scaled_comfort_soc_penalty"))
        battery_degradation_cost = _safe_float(component.get("battery_degradation_cost"))
        dispatch_reward_env = _indexed_float(raw_dispatch_rewards, index)
        dispatch_reward_train = _indexed_float(train_dispatch_rewards, index)
        training_projection_penalty = max(0.0, float(dispatch_reward_env - dispatch_reward_train))
        row = {
            "episode": int(episode),
            "step": int(step),
            "algorithm": str(algorithm),
            "agent_id": f"{vpp_id}_dispatch",
            "vpp_id": str(vpp_id),
            "market_price": market_price,
            "delivered_p_mw": delivered_p_mw,
            "dt_hours": dt_hours,
            "energy_market_revenue": energy_market_revenue,
            "der_operation_cost": der_operation_cost,
            "visible_energy_minus_operation_cost": float(visible_energy_minus_operation),
            "market_energy_margin_total": float(market_energy_margin_total),
            **settlement_breakdown,
            "private_profit_vs_visible_energy_residual": float(
                private_profit_proxy - visible_energy_minus_operation
            ),
            "economic_surplus_vs_market_margin_residual": float(
                settlement_breakdown["economic_operational_surplus"] - market_energy_margin_total
            ),
            "private_profit_proxy": private_profit_proxy,
            "private_profit_weight": private_profit_weight,
            "dispatch_private_profit_reward": _safe_float(
                component.get("dispatch_private_profit_reward"),
                private_profit_weight * private_profit_proxy,
            ),
            "dispatch_reward_env": dispatch_reward_env,
            "dispatch_reward_train": dispatch_reward_train,
            "flexibility_service_payment": _safe_float(component.get("flexibility_service_payment")),
            "service_payment": _safe_float(component.get("service_payment")),
            "service_payment_weight": service_payment_weight,
            "availability_payment": _safe_float(component.get("availability_payment")),
            "availability_payment_weight": availability_payment_weight,
            "preferred_region_bonus": _safe_float(component.get("preferred_region_bonus")),
            **{key: _safe_float(component.get(key)) for key in storage_shaping_keys},
            "baseline_p_mw": _safe_float(component.get("baseline_p_mw")),
            "requested_delta_p_mw": _safe_float(component.get("requested_delta_p_mw")),
            "accepted_delta_p_mw": _safe_float(component.get("accepted_delta_p_mw")),
            "actual_delta_p_mw": _safe_float(component.get("actual_delta_p_mw")),
            **{key: _safe_float(component.get(key)) for key in action_landing_keys},
            "action_landing_drop_reason": str(component.get("action_landing_drop_reason", "")),
            "verified_delivery_mw": _safe_float(component.get("verified_delivery_mw")),
            "contract_shortfall_mw": _safe_float(component.get("contract_shortfall_mw")),
            "contract_delivery_weight": contract_delivery_weight,
            "contract_delivery_penalty": contract_delivery_penalty,
            "dispatch_responsible_projection_gap_mw": _safe_float(
                component.get("dispatch_responsible_projection_gap_mw")
            ),
            "dispatch_projection_penalty": dispatch_projection_penalty,
            "scaled_comfort_soc_penalty": scaled_comfort_soc_penalty,
            "comfort_soc_weight": comfort_soc_weight,
            "battery_degradation_cost": battery_degradation_cost,
            "battery_degradation_weight": battery_degradation_weight,
            "reward_scaled_contract_delivery_penalty": float(
                contract_delivery_weight * contract_delivery_penalty
            ),
            "reward_scaled_dispatch_projection_penalty": float(dispatch_projection_penalty),
            "reward_scaled_training_projection_penalty": float(training_projection_penalty),
            "reward_scaled_total_projection_penalty": float(
                dispatch_projection_penalty + training_projection_penalty
            ),
            "reward_scaled_comfort_soc_penalty": float(comfort_soc_weight * scaled_comfort_soc_penalty),
            "reward_scaled_battery_degradation_penalty": float(
                battery_degradation_weight * battery_degradation_cost
            ),
            "energy_market_revenue_formula": "market_price * delivered_p_mw * dt_hours",
            "der_operation_cost_formula": "vpp.operating_cost() * dt_hours",
            "economic_operational_surplus_formula": (
                "export_revenue_total + evcs_user_revenue_total - import_energy_cost_total "
                "- der_operating_cost_total - battery_degradation_cost_total"
            ),
            "quality_adjusted_surplus_formula": (
                "economic_operational_surplus - service_quality_penalty_total"
            ),
            "service_quality_penalty_formula": "comfort_cost_total + unserved_penalty_total",
            "private_profit_proxy_formula": (
                "economic_operational_surplus + enabled transfers - contract_penalty"
            ),
            "dispatch_private_profit_reward_formula": "private_profit_weight * private_profit_proxy",
            "action_landing_ratio_formula": (
                "abs(actual_delta_p_mw) / (abs(decoded_delta_p_mw) + epsilon)"
            ),
        }
        rows.append(row)
    return rows


def _indexed_float(values: Any | None, index: int) -> float:
    if values is None:
        return 0.0
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return 0.0
