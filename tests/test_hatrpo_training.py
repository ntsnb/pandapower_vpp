from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from vpp_dso_sim.learning.hatrpo import (
    HATRPOConfig,
    build_hatrpo_gaussian_policy,
    conjugate_gradient,
    evaluate_hatrpo_checkpoint,
    hatrpo_trust_region_update,
    torch_available,
    train_hatrpo,
)


PANDAPOWER_AVAILABLE = importlib.util.find_spec("pandapower") is not None

REWARD_V2_REQUIRED_STEP_COLUMNS = {
    "dso_reward_env",
    "dso_reward_train",
    "dso_reward_critic_scaled",
    "dso_safety_margin_penalty",
    "dso_voltage_guard_penalty",
    "dso_line_guard_penalty",
    "dso_trafo_guard_penalty",
    "dso_powerflow_failure_penalty",
    "dso_flex_procurement_cost",
    "dso_loss_cost",
    "dso_curtailment_cost",
    "dso_safe_capacity_utilization_reward",
    "dso_over_conservative_curtailment_penalty",
    "dso_responsible_projection_gap_mw",
    "dso_responsible_projection_penalty",
    "tracking_bonus_diagnostic",
    "effective_response_bonus_diagnostic",
    "target_tracking_error_to_raw_target",
    "target_tracking_error_to_projected_target",
    "mean_dispatch_reward_env",
    "mean_dispatch_reward_train",
    "min_dispatch_reward_train",
    "p05_dispatch_reward_train",
    "p95_dispatch_reward_train",
    "private_profit_proxy",
    "economic_operational_surplus",
    "quality_adjusted_operational_surplus",
    "service_quality_penalty_total",
    "settlement_audit_complete",
    "settlement_power_balance_ok",
    "settlement_power_balance_error_mw",
    "energy_market_revenue",
    "visible_energy_minus_operation_cost",
    "market_energy_margin_total",
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
    "baseline_p_mw",
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
    "requested_delta_p_mw",
    "accepted_delta_p_mw",
    "actual_delta_p_mw",
    "verified_delivery_mw",
    "contract_shortfall_mw",
    "contract_delivery_penalty",
    "availability_payment",
    "service_payment",
    "der_operation_cost",
    "battery_degradation_cost",
    "comfort_penalty",
    "soc_penalty",
    "dispatch_responsible_projection_gap_mw",
    "dispatch_projection_penalty",
    "mean_portfolio_reward_env",
    "mean_portfolio_reward_train",
    "portfolio_window_profit",
    "portfolio_window_contract_shortfall",
    "portfolio_window_shield_intervention",
    "portfolio_window_projection_gap",
    "portfolio_window_comfort_soc_penalty",
    "portfolio_window_verified_capacity",
    "portfolio_switching_cost",
    "portfolio_action_type",
    "raw_action_violation_rate",
    "post_ac_violation_count",
    "post_ac_security_penalty",
    "shield_intervention_gap_mw",
    "shield_intervention_penalty",
    "action_projection_gap_mw",
    "local_bounds_projection_gap_mw",
    "ac_aware_projection_gap_mw",
    "ac_certified_projection_gap_mw",
    "certificate_repair_rate",
}


def test_conjugate_gradient_solves_small_spd_system():
    torch = pytest.importorskip("torch")
    matrix = torch.tensor([[4.0, 1.0], [1.0, 3.0]], dtype=torch.float32)
    rhs = torch.tensor([1.0, 2.0], dtype=torch.float32)

    solution, diagnostics = conjugate_gradient(lambda vector: matrix @ vector, rhs, cg_iters=10)

    assert torch.allclose(solution, torch.linalg.solve(matrix, rhs), atol=1e-4)
    assert diagnostics["cg_iterations"] <= 10
    assert diagnostics["cg_residual"] < diagnostics["initial_residual"]


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_hatrpo_trust_region_update_uses_fvp_and_respects_kl():
    import torch

    torch.manual_seed(5)
    policy = build_hatrpo_gaussian_policy(obs_dim=3, action_dim=2, hidden_dim=8, torch_module=torch)
    obs = torch.randn(8, 3)
    with torch.no_grad():
        mean, log_std = policy(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        actions = dist.sample()
        old_log_probs = dist.log_prob(actions).sum(dim=-1)
    advantages = torch.tensor([1.0, 0.4, -0.7, 1.3, -0.2, 0.8, -1.1, 0.5], dtype=torch.float32)

    diagnostics = hatrpo_trust_region_update(
        policy=policy,
        obs=obs,
        actions=actions,
        old_log_probs=old_log_probs,
        advantages=advantages,
        max_kl=0.05,
        cg_iters=8,
        cg_damping=0.10,
        line_search_steps=6,
        line_search_backtrack=0.5,
        distribution="gaussian",
        torch_module=torch,
        role="unit_test_gaussian",
    )

    assert diagnostics["conjugate_gradient"] is True
    assert diagnostics["fisher_vector_product"] is True
    assert diagnostics["update_accepted"] is True
    assert diagnostics["mean_kl"] <= 0.05001
    assert diagnostics["param_delta_l2"] > 0.0


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
@pytest.mark.skipif(not PANDAPOWER_AVAILABLE, reason="pandapower is not installed")
def test_hatrpo_training_runs_against_multi_agent_ctde_env():
    output_dir = Path("outputs") / "test_hatrpo_training"
    result = train_hatrpo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir,
        config=HATRPOConfig(
            episodes=1,
            horizon_steps=2,
            hidden_dim=16,
            seed=61,
            max_kl=0.05,
            cg_iters=5,
            line_search_steps=5,
            value_epochs=1,
            portfolio_decision_interval_steps=1,
            reward_scale=0.01,
        ),
    )

    summary = result["summary"]
    update_metrics = result["update_metrics"]

    assert summary["algorithm"] == "hatrpo_trust_region_ctde"
    assert summary["hatrpo_complete_core"] is True
    assert summary["trust_region_surrogate_update"] is True
    assert summary["conjugate_gradient"] is True
    assert summary["fisher_vector_product"] is True
    assert summary["centralized_critic_uses_global_state"] is True
    assert summary["critic_visible_to_decentralized_actors"] is False
    assert summary["actor_privacy_scope"] == "local_actor_observation"
    assert summary["ctde_observation_encoding"] is True
    assert summary["dso_policy_distribution"] == "gaussian"
    assert summary["dispatch_policy_distribution"] == "gaussian"
    assert summary["portfolio_policy_distribution"] == "categorical"
    assert summary["actor_update_attempts"] == 3
    assert summary["critic_updates"] > 0
    assert summary["reward_scale"] == pytest.approx(0.01)
    assert summary["requested_device"] == "auto"
    assert summary["resolved_device"] in {"cpu", "cuda", "cuda:0"}
    assert isinstance(summary["cuda_available"], bool)
    assert "device_meta" in summary
    assert summary["param_delta_l2"] > 0.0
    assert result["checkpoint"].exists()
    assert (output_dir / "hatrpo_training_summary.json").exists()
    assert (output_dir / "hatrpo_update_metrics.csv").exists()
    assert not result["step_metrics"].empty
    assert not update_metrics.empty
    assert {"dso_global_guidance", "shared_vpp_dispatch", "shared_vpp_portfolio"}.issubset(
        set(update_metrics["role"])
    )
    assert set(result["step_metrics"]["privacy_scope"]) == {"own_vpp_local_observation_only"}
    observed_kl = update_metrics["mean_kl"].dropna()
    assert observed_kl.empty or float(observed_kl.max()) <= float(summary["max_kl"]) + 1e-5

    eval_result = evaluate_hatrpo_checkpoint(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        checkpoint_path=result["checkpoint"],
        output_dir=output_dir / "eval",
        horizon_steps=2,
        seed=62,
    )
    assert eval_result["summary"]["evaluation_mode"] == "frozen_mean_argmax_actor"
    assert eval_result["summary"]["total_violation_count"] >= 0
    assert not eval_result["step_metrics"].empty
    assert (output_dir / "eval" / "hatrpo_frozen_eval_summary.json").exists()


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
@pytest.mark.skipif(not PANDAPOWER_AVAILABLE, reason="pandapower is not installed")
def test_hatrpo_applies_reward_shield_coefficients_from_yaml(tmp_path):
    config_path = tmp_path / "shield_coeff_hatrpo.yaml"
    config_path.write_text(
        "\n".join(
            [
                "extends: configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
                "reward:",
                "  shield:",
                "    dso_penalty_coef: 0.30",
                "    dispatch_penalty_coef: 0.40",
            ]
        ),
        encoding="utf-8",
    )

    result = train_hatrpo(
        config_path=config_path,
        output_dir=tmp_path / "shield_coeff_hatrpo",
        config=HATRPOConfig(
            episodes=1,
            horizon_steps=1,
            hidden_dim=16,
            seed=71,
            max_kl=0.05,
            cg_iters=3,
            line_search_steps=3,
            value_epochs=1,
            portfolio_decision_interval_steps=1,
            device="cpu",
        ),
    )

    assert result["summary"]["dso_shield_intervention_penalty_coef"] == pytest.approx(0.30)
    assert result["summary"]["dispatch_shield_intervention_penalty_coef"] == pytest.approx(0.40)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
@pytest.mark.skipif(not PANDAPOWER_AVAILABLE, reason="pandapower is not installed")
def test_hatrpo_reward_v2_step_metrics_cover_reward_and_security_columns(tmp_path):
    output_dir = tmp_path / "hatrpo_reward_v2_columns"
    result = train_hatrpo(
        config_path=Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
        output_dir=output_dir,
        config=HATRPOConfig(
            episodes=1,
            horizon_steps=1,
            hidden_dim=16,
            seed=92,
            max_kl=0.05,
            cg_iters=3,
            line_search_steps=3,
            value_epochs=1,
            portfolio_decision_interval_steps=1,
            device="cpu",
        ),
    )

    missing = REWARD_V2_REQUIRED_STEP_COLUMNS.difference(result["step_metrics"].columns)
    assert not missing

    trace_path = output_dir / "hatrpo_dispatch_private_profit_trace.csv"
    episode_trace_path = output_dir / "hatrpo_dispatch_private_profit_trace_episode_0000.csv"
    reward_cards_dir = output_dir / "reports" / "reward_dynamic_cards"
    assert trace_path.exists()
    assert episode_trace_path.exists()
    assert (reward_cards_dir / "reward_dynamic_cards_hatrpo_trust_region_ctde_episode_0000.html").exists()
    assert (reward_cards_dir / "latest_reward_dynamic_cards.html").exists()
    trace = pd.read_csv(trace_path)
    episode_trace = pd.read_csv(episode_trace_path)
    required_trace_columns = {
        "episode",
        "step",
        "algorithm",
        "agent_id",
        "vpp_id",
        "market_price",
        "delivered_p_mw",
        "dt_hours",
        "energy_market_revenue",
        "der_operation_cost",
        "visible_energy_minus_operation_cost",
        "market_energy_margin_total",
        "economic_operational_surplus",
        "quality_adjusted_operational_surplus",
        "service_quality_penalty_total",
        "export_revenue_total",
        "evcs_user_revenue_total",
        "import_energy_cost_total",
        "der_operating_cost_total",
        "battery_degradation_cost_total",
        "comfort_cost_total",
        "unserved_penalty_total",
        "private_profit_proxy",
        "private_profit_weight",
        "dispatch_private_profit_reward",
        "energy_market_revenue_formula",
        "private_profit_proxy_formula",
        "dispatch_private_profit_reward_formula",
    }
    assert not required_trace_columns.difference(trace.columns)
    assert set(trace["agent_id"].astype(str).str.endswith("_dispatch")) == {True}
    assert (
        trace["energy_market_revenue"]
        - trace["market_price"] * trace["delivered_p_mw"] * trace["dt_hours"]
    ).abs().max() == pytest.approx(0.0)
    assert (
        trace["private_profit_proxy"]
        - (trace["energy_market_revenue"] - trace["der_operation_cost"])
    ).abs().max() == pytest.approx(0.0)
    assert (
        trace["visible_energy_minus_operation_cost"]
        - (trace["energy_market_revenue"] - trace["der_operation_cost"])
    ).abs().max() == pytest.approx(0.0)
    assert (
        trace["economic_operational_surplus"]
        - trace["private_profit_proxy"]
    ).abs().max() == pytest.approx(0.0)
    assert (
        trace["dispatch_private_profit_reward"]
        - trace["private_profit_weight"] * trace["private_profit_proxy"]
    ).abs().max() == pytest.approx(0.0)
    assert len(episode_trace) == len(trace)
    assert set(episode_trace["episode"]) == {0}
