from __future__ import annotations

import pandas as pd

from scripts.analyze_dispatch_absorption_rewards import analyze
from scripts.watch_dispatch_profit_episode_report import generate_report


def test_dispatch_absorption_report_keeps_full_settlement_attribution(tmp_path) -> None:
    trace_path = tmp_path / "happo_dispatch_private_profit_trace_episode_0002.csv"
    pd.DataFrame(
        [
            {
                "episode": 2,
                "step": 0,
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 80.0,
                "delivered_p_mw": -0.10,
                "dt_hours": 0.25,
                "energy_market_revenue": -2.0,
                "der_operation_cost": 0.4,
                "visible_energy_minus_operation_cost": -2.4,
                "market_energy_margin_total": 4.0,
                "economic_operational_surplus": 3.5,
                "quality_adjusted_operational_surplus": -96.5,
                "service_quality_penalty_total": 100.0,
                "export_revenue_total": 8.0,
                "evcs_user_revenue_total": 2.0,
                "import_energy_cost_total": 6.0,
                "der_operating_cost_total": 0.5,
                "battery_degradation_cost_total": 0.0,
                "comfort_cost_total": 90.0,
                "unserved_penalty_total": 10.0,
                "private_profit_proxy": 3.5,
                "private_profit_weight": 1.0,
                "dispatch_private_profit_reward": 3.5,
            },
            {
                "episode": 2,
                "step": 1,
                "agent_id": "vpp_b_dispatch",
                "vpp_id": "vpp_b",
                "market_price": 80.0,
                "delivered_p_mw": 0.05,
                "dt_hours": 0.25,
                "energy_market_revenue": 1.0,
                "der_operation_cost": 0.2,
                "visible_energy_minus_operation_cost": 0.8,
                "market_energy_margin_total": 0.8,
                "economic_operational_surplus": -1.2,
                "quality_adjusted_operational_surplus": -1.2,
                "service_quality_penalty_total": 0.0,
                "export_revenue_total": 1.0,
                "evcs_user_revenue_total": 0.0,
                "import_energy_cost_total": 0.2,
                "der_operating_cost_total": 2.0,
                "battery_degradation_cost_total": 0.0,
                "comfort_cost_total": 0.0,
                "unserved_penalty_total": 0.0,
                "private_profit_proxy": -1.2,
                "private_profit_weight": 1.0,
                "dispatch_private_profit_reward": -1.2,
            },
        ]
    ).to_csv(trace_path, index=False)

    report_path, detail_path = analyze(trace_path, tmp_path)
    detail = pd.read_csv(detail_path)
    report_text = report_path.read_text(encoding="utf-8")

    assert "economic_operational_surplus" in detail.columns
    assert "service_quality_penalty_total" in detail.columns
    assert "private_profit_vs_visible_energy_residual" in detail.columns
    assert detail.loc[0, "settlement_trace_complete"] == 1.0
    assert detail.loc[0, "private_profit_vs_visible_energy_residual"] == 5.9
    assert "完整 settlement 分项" in report_text
    assert "经济运行盈余" in report_text


def test_dispatch_absorption_report_flags_legacy_trace_without_settlement_breakdown(tmp_path) -> None:
    trace_path = tmp_path / "happo_dispatch_private_profit_trace_episode_0003.csv"
    pd.DataFrame(
        [
            {
                "episode": 3,
                "step": 0,
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 80.0,
                "delivered_p_mw": -0.02,
                "dt_hours": 0.25,
                "energy_market_revenue": -0.4,
                "der_operation_cost": 0.3,
                "private_profit_proxy": -2000.0,
                "private_profit_weight": 1.0,
                "dispatch_private_profit_reward": -2000.0,
            }
        ]
    ).to_csv(trace_path, index=False)

    report_path, detail_path = analyze(trace_path, tmp_path)
    detail = pd.read_csv(detail_path)
    report_text = report_path.read_text(encoding="utf-8")

    assert detail.loc[0, "settlement_trace_complete"] == 0.0
    assert detail.loc[0, "private_profit_vs_visible_energy_residual"] == -1999.3
    assert "旧 trace 缺少完整 settlement 分项" in report_text


def test_dispatch_absorption_report_summarizes_action_landing_when_available(tmp_path) -> None:
    trace_path = tmp_path / "happo_dispatch_private_profit_trace_episode_0005.csv"
    pd.DataFrame(
        [
            {
                "episode": 5,
                "step": 0,
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 80.0,
                "delivered_p_mw": 0.05,
                "dt_hours": 0.25,
                "energy_market_revenue": 1.0,
                "der_operation_cost": 0.2,
                "economic_operational_surplus": 0.8,
                "quality_adjusted_operational_surplus": 0.8,
                "service_quality_penalty_total": 0.0,
                "export_revenue_total": 1.0,
                "evcs_user_revenue_total": 0.0,
                "import_energy_cost_total": 0.0,
                "der_operating_cost_total": 0.2,
                "battery_degradation_cost_total": 0.0,
                "comfort_cost_total": 0.0,
                "unserved_penalty_total": 0.0,
                "private_profit_proxy": 0.8,
                "dispatch_private_profit_reward": 0.8,
                "decoded_delta_p_mw": 0.10,
                "actual_delta_p_mw": 0.08,
                "action_landing_ratio": 0.80,
                "actual_delta_nonzero_flag": 1.0,
                "action_landing_drop_reason": "landed",
                "raw_to_device_gap_mw": 0.00,
                "device_to_ac_gap_mw": 0.00,
                "ac_to_actual_gap_mw": 0.02,
                "accepted_to_actual_gap_mw": 0.02,
            },
            {
                "episode": 5,
                "step": 1,
                "agent_id": "vpp_b_dispatch",
                "vpp_id": "vpp_b",
                "market_price": 80.0,
                "delivered_p_mw": 0.00,
                "dt_hours": 0.25,
                "energy_market_revenue": 0.0,
                "der_operation_cost": 0.0,
                "economic_operational_surplus": 0.0,
                "quality_adjusted_operational_surplus": 0.0,
                "service_quality_penalty_total": 0.0,
                "export_revenue_total": 0.0,
                "evcs_user_revenue_total": 0.0,
                "import_energy_cost_total": 0.0,
                "der_operating_cost_total": 0.0,
                "battery_degradation_cost_total": 0.0,
                "comfort_cost_total": 0.0,
                "unserved_penalty_total": 0.0,
                "private_profit_proxy": 0.0,
                "dispatch_private_profit_reward": 0.0,
                "decoded_delta_p_mw": 0.10,
                "actual_delta_p_mw": 0.00,
                "action_landing_ratio": 0.00,
                "actual_delta_nonzero_flag": 0.0,
                "action_landing_drop_reason": "dso_envelope_clip",
                "raw_to_device_gap_mw": 0.00,
                "device_to_ac_gap_mw": 0.10,
                "ac_to_actual_gap_mw": 0.00,
                "accepted_to_actual_gap_mw": 0.10,
            },
        ]
    ).to_csv(trace_path, index=False)

    report_path, detail_path = analyze(trace_path, tmp_path)
    detail = pd.read_csv(detail_path)
    report_text = report_path.read_text(encoding="utf-8")

    assert "action_landing_trace_complete" in detail.columns
    assert detail["action_landing_trace_complete"].mean() == 1.0
    assert "动作落地审计" in report_text
    assert "actual_delta_nonzero_rate" in report_text
    assert "dso_envelope_clip" in report_text


def test_watch_dispatch_profit_report_uses_settlement_formula_when_available(tmp_path) -> None:
    trace_path = tmp_path / "happo_dispatch_private_profit_trace_episode_0004.csv"
    pd.DataFrame(
        [
            {
                "episode": 4,
                "step": 0,
                "algorithm": "happo",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 80.0,
                "delivered_p_mw": -0.10,
                "dt_hours": 0.25,
                "energy_market_revenue": -2.0,
                "der_operation_cost": 0.4,
                "visible_energy_minus_operation_cost": -2.4,
                "market_energy_margin_total": 4.0,
                "economic_operational_surplus": 3.5,
                "quality_adjusted_operational_surplus": -96.5,
                "service_quality_penalty_total": 100.0,
                "export_revenue_total": 8.0,
                "evcs_user_revenue_total": 2.0,
                "import_energy_cost_total": 6.0,
                "der_operating_cost_total": 0.5,
                "battery_degradation_cost_total": 0.0,
                "comfort_cost_total": 90.0,
                "unserved_penalty_total": 10.0,
                "private_profit_proxy": 3.5,
                "private_profit_weight": 1.0,
                "dispatch_private_profit_reward": 3.5,
                "decoded_delta_p_mw": 0.10,
                "actual_delta_p_mw": 0.08,
                "action_landing_ratio": 0.80,
                "actual_delta_nonzero_flag": 1.0,
                "action_landing_drop_reason": "landed",
            }
        ]
    ).to_csv(trace_path, index=False)

    report_path = generate_report(trace_path, tmp_path)
    report_text = report_path.read_text(encoding="utf-8")
    summary_path = tmp_path / "reports" / "dispatch_private_profit_agent_summary_happo_episode_0004.csv"
    summary = pd.read_csv(summary_path)

    assert "经济运行盈余" in report_text
    assert "energy_market_revenue - der_operation_cost" not in report_text
    assert "动作落地审计" in report_text
    assert "total_economic_operational_surplus" in summary.columns
    assert "mean_action_landing_ratio" in summary.columns
    assert "actual_delta_nonzero_rate" in summary.columns
    assert summary.loc[0, "total_private_profit_vs_visible_residual"] == 5.9
    assert summary.loc[0, "mean_action_landing_ratio"] == 0.8
