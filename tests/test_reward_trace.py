from __future__ import annotations

import pytest

from vpp_dso_sim.learning.reward_trace import dispatch_private_profit_trace_rows


def test_dispatch_private_profit_trace_includes_full_settlement_breakdown() -> None:
    rows = dispatch_private_profit_trace_rows(
        episode=3,
        step=9,
        algorithm="happo",
        vpp_ids=["vpp_a"],
        dispatch_components=[
            {
                "private_profit_proxy": 7.5,
                "economic_operational_surplus": 7.5,
                "quality_adjusted_operational_surplus": -92.5,
                "service_quality_penalty_total": 100.0,
                "export_revenue_total": 12.0,
                "evcs_user_revenue_total": 3.0,
                "import_energy_cost_total": 4.0,
                "der_operating_cost_total": 2.0,
                "battery_degradation_cost_total": 1.5,
                "comfort_cost_total": 80.0,
                "unserved_penalty_total": 20.0,
                "energy_market_revenue": 8.0,
                "der_operation_cost": 0.5,
                "private_profit_weight": 1.0,
            }
        ],
    )

    row = rows[0]
    assert row["economic_operational_surplus"] == pytest.approx(7.5)
    assert row["quality_adjusted_operational_surplus"] == pytest.approx(-92.5)
    assert row["service_quality_penalty_total"] == pytest.approx(100.0)
    assert row["export_revenue_total"] == pytest.approx(12.0)
    assert row["evcs_user_revenue_total"] == pytest.approx(3.0)
    assert row["import_energy_cost_total"] == pytest.approx(4.0)
    assert row["der_operating_cost_total"] == pytest.approx(2.0)
    assert row["battery_degradation_cost_total"] == pytest.approx(1.5)
    assert row["comfort_cost_total"] == pytest.approx(80.0)
    assert row["unserved_penalty_total"] == pytest.approx(20.0)
    assert row["private_profit_proxy_formula"] == (
        "economic_operational_surplus + enabled transfers - contract_penalty"
    )
    assert row["quality_adjusted_surplus_formula"] == (
        "economic_operational_surplus - service_quality_penalty_total"
    )


def test_dispatch_private_profit_trace_includes_action_landing_audit() -> None:
    rows = dispatch_private_profit_trace_rows(
        episode=4,
        step=11,
        algorithm="happo",
        vpp_ids=["vpp_a"],
        dispatch_components=[
            {
                "private_profit_proxy": 1.0,
                "private_profit_weight": 1.0,
                "baseline_p_mw": -0.20,
                "raw_action_norm": 0.50,
                "raw_target_p_mw": 0.10,
                "decoded_target_p_mw": 0.05,
                "device_feasible_target_p_mw": 0.04,
                "pre_ac_target_p_mw": 0.03,
                "ac_projected_target_p_mw": 0.02,
                "ac_certified_target_p_mw": 0.01,
                "actual_target_p_mw": 0.00,
                "raw_delta_p_mw": 0.30,
                "decoded_delta_p_mw": 0.25,
                "device_feasible_delta_p_mw": 0.24,
                "pre_ac_delta_p_mw": 0.23,
                "ac_projected_delta_p_mw": 0.22,
                "ac_certified_delta_p_mw": 0.21,
                "actual_delta_p_mw": 0.20,
                "raw_to_device_gap_mw": 0.06,
                "device_to_ac_gap_mw": 0.02,
                "ac_to_actual_gap_mw": 0.02,
                "accepted_to_actual_gap_mw": 0.03,
                "actual_delta_nonzero_flag": 1.0,
                "action_landing_ratio": 0.80,
                "action_landing_drop_reason": "landed",
            }
        ],
    )

    row = rows[0]
    assert row["raw_action_norm"] == pytest.approx(0.50)
    assert row["decoded_target_p_mw"] == pytest.approx(0.05)
    assert row["device_feasible_target_p_mw"] == pytest.approx(0.04)
    assert row["pre_ac_target_p_mw"] == pytest.approx(0.03)
    assert row["ac_projected_target_p_mw"] == pytest.approx(0.02)
    assert row["ac_certified_target_p_mw"] == pytest.approx(0.01)
    assert row["actual_target_p_mw"] == pytest.approx(0.00)
    assert row["action_landing_ratio"] == pytest.approx(0.80)
    assert row["action_landing_drop_reason"] == "landed"
    assert row["action_landing_ratio_formula"] == (
        "abs(actual_delta_p_mw) / (abs(decoded_delta_p_mw) + epsilon)"
    )


def test_dispatch_private_profit_trace_includes_reward_scaled_cost_and_weight_audit() -> None:
    rows = dispatch_private_profit_trace_rows(
        episode=5,
        step=12,
        algorithm="happo",
        vpp_ids=["vpp_a"],
        dispatch_components=[
            {
                "private_profit_proxy": 10.0,
                "private_profit_weight": 1.0,
                "service_payment": 7.0,
                "service_payment_weight": 0.0,
                "availability_payment": 3.0,
                "availability_payment_weight": 0.0,
                "contract_delivery_penalty": 9.0,
                "contract_delivery_weight": 0.0,
                "dispatch_projection_penalty": 0.7,
                "scaled_comfort_soc_penalty": 3.0,
                "comfort_soc_weight": 0.02,
                "battery_degradation_cost": 4.0,
                "battery_degradation_weight": 0.01,
                "storage_potential_raw": 12.5,
                "storage_potential_shaping_reward": 0.25,
                "storage_potential_shaping_weight": 0.02,
            }
        ],
        raw_dispatch_rewards=[10.85],
        train_dispatch_rewards=[10.15],
    )

    row = rows[0]
    assert row["service_payment_weight"] == pytest.approx(0.0)
    assert row["availability_payment_weight"] == pytest.approx(0.0)
    assert row["storage_potential_raw"] == pytest.approx(12.5)
    assert row["storage_potential_shaping_reward"] == pytest.approx(0.25)
    assert row["storage_potential_shaping_weight"] == pytest.approx(0.02)
    assert row["reward_scaled_contract_delivery_penalty"] == pytest.approx(0.0)
    assert row["reward_scaled_dispatch_projection_penalty"] == pytest.approx(0.7)
    assert row["reward_scaled_training_projection_penalty"] == pytest.approx(0.7)
    assert row["reward_scaled_total_projection_penalty"] == pytest.approx(1.4)
    assert row["reward_scaled_comfort_soc_penalty"] == pytest.approx(0.06)
    assert row["reward_scaled_battery_degradation_penalty"] == pytest.approx(0.04)
