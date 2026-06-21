from __future__ import annotations

import pandas as pd
import pytest

from vpp_dso_sim.visualization.reward_dynamic_report import (
    write_reward_dynamic_episode_report,
)


def test_reward_dynamic_episode_report_writes_html_and_abs_share_csv(tmp_path) -> None:
    step_metrics = pd.DataFrame(
        [
            {
                "episode": 0,
                "step": 0,
                "algorithm": "happo_sequential_ctde",
                "reward": 1.0,
                "dso_reward_train": 0.4,
                "mean_dispatch_reward_train": 0.5,
                "mean_portfolio_reward_train": 0.1,
                "dso_safety_gate": 0.92,
                "dso_vpp_welfare_raw": 10.0,
                "raw_action_safety_cost_norm": 0.05,
                "projected_action_safety_cost_norm": 0.0,
                "private_profit_proxy": 3.0,
                "vpp_operational_surplus_ex_transfer": 3.0,
                "service_payment": 0.0,
                "availability_payment": 0.0,
                "contract_delivery_penalty": 0.0,
                "dispatch_projection_penalty": 0.2,
                "storage_potential_shaping_reward": -0.03,
                "settlement_audit_complete": 1.0,
                "settlement_power_balance_ok": 1.0,
                "portfolio_window_profit": 0.1,
                "portfolio_switching_cost": 0.0,
            },
            {
                "episode": 0,
                "step": 1,
                "algorithm": "happo_sequential_ctde",
                "reward": 1.2,
                "dso_reward_train": 0.5,
                "mean_dispatch_reward_train": 0.6,
                "mean_portfolio_reward_train": 0.1,
                "dso_safety_gate": 0.95,
                "dso_vpp_welfare_raw": 12.0,
                "raw_action_safety_cost_norm": 0.0,
                "projected_action_safety_cost_norm": 0.0,
                "private_profit_proxy": 4.0,
                "vpp_operational_surplus_ex_transfer": 4.0,
                "service_payment": 0.0,
                "availability_payment": 0.0,
                "contract_delivery_penalty": 0.0,
                "dispatch_projection_penalty": 0.0,
                "storage_potential_shaping_reward": -0.02,
                "settlement_audit_complete": 1.0,
                "settlement_power_balance_ok": 1.0,
                "portfolio_window_profit": 0.2,
                "portfolio_switching_cost": 0.0,
            },
        ]
    )
    episode_metrics = pd.DataFrame(
        [
            {
                "episode": 0,
                "algorithm": "happo_sequential_ctde",
                "episode_reward": 2.2,
                "episode_cost": 1.0,
                "violation_count": 0,
                "critic_loss": 0.3,
                "dso_policy_loss": -0.1,
                "dispatch_policy_loss": -0.2,
                "portfolio_policy_loss": -0.05,
            }
        ]
    )
    dispatch_trace = pd.DataFrame(
        [
            {
                "episode": 0,
                "step": 0,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 80.0,
                "delivered_p_mw": -0.1,
                "dt_hours": 0.25,
                "private_profit_proxy": 3.0,
                "dispatch_private_profit_reward": 3.0,
                "energy_market_revenue_formula": "DER-level settlement audit",
                "private_profit_proxy_formula": "operational_surplus + enabled transfers - contract_penalty",
            }
        ]
    )

    path = write_reward_dynamic_episode_report(
        output_dir=tmp_path,
        algorithm="happo_sequential_ctde",
        episode=0,
        step_metrics=step_metrics,
        episode_metrics=episode_metrics,
        dispatch_trace=dispatch_trace,
    )

    assert path.exists()
    assert (tmp_path / "latest_reward_dynamic_cards.html").exists()
    assert (tmp_path / "reward_component_abs_share_happo_sequential_ctde_episode_0000.csv").exists()
    html = path.read_text(encoding="utf-8")
    assert "Reward 动态卡片看板" in html
    assert "happo_sequential_ctde" in html
    assert "Episode 0" in html
    assert "VPP Dispatch" in html
    assert "DSO 安全与福利" in html
    assert "Settlement 审计" in html
    assert "storage_potential_shaping_reward" in html
    assert "service/availability/contract 默认关闭" in html
    assert "同类智能体内绝对占比" in html

    share = pd.read_csv(tmp_path / "reward_component_abs_share_happo_sequential_ctde_episode_0000.csv")
    assert {"role_abs_total", "global_abs_share", "share_scope"}.issubset(share.columns)
    assert set(share["share_scope"]) == {"role_internal_abs_share"}

    role_share_sums = share.groupby("role")["share"].sum().to_dict()
    assert role_share_sums["DSO"] == pytest.approx(1.0)
    assert role_share_sums["dispatch"] == pytest.approx(1.0)

    dispatch = share.set_index("key")
    assert dispatch.at["private_profit_proxy", "share"] > 0.95
