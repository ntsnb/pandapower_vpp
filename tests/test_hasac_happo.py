from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from vpp_dso_sim.learning.advanced_marl import (
    HAPPOConfig,
    HASACConfig,
    MultiHeadValueCriticSpec,
    build_multi_head_value_critic,
    evaluate_hasac_checkpoint,
    torch_available,
    train_happo,
    train_hasac,
)

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


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_multi_head_value_critic_exposes_named_heads():
    import torch

    spec = MultiHeadValueCriticSpec(
        state_dim=5,
        joint_action_dim=3,
        head_names=("dso_global_guidance", "vpp_a_dispatch", "vpp_a_portfolio"),
        hidden_dims=(8, 8),
    )

    critic = build_multi_head_value_critic(spec, require_torch=True)
    values = critic(torch.zeros(2, 5), torch.zeros(2, 3))
    heads = critic.forward_heads(torch.zeros(2, 5), torch.zeros(2, 3))

    assert values.shape == (2, 3)
    assert set(heads) == {"dso_global_guidance", "vpp_a_dispatch", "vpp_a_portfolio"}
    assert heads["dso_global_guidance"].shape == (2,)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_training_writes_sequential_update_artifacts():
    output_dir = Path("outputs") / "test_happo_training"
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir,
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=2,
            hidden_dim=16,
            ppo_epochs=1,
            seed=51,
        ),
    )

    summary = result["summary"]

    assert summary["algorithm"] == "happo_sequential_ctde"
    assert summary["sequential_role_update"] is True
    assert summary["importance_correction"] is True
    assert summary["role_update_steps"] > 0
    assert summary["value_head_count"] > 1
    assert result["checkpoint"].exists()
    assert (output_dir / "happo_training_summary.json").exists()
    assert not result["update_metrics"].empty
    roles = set(result["update_metrics"]["role"])
    assert "dso_global_guidance" in roles
    assert any(role.endswith("_dispatch") for role in roles)
    assert any(role.endswith("_portfolio") for role in roles)
    assert summary["per_vpp_dispatch_actors"] is True
    assert summary["per_vpp_portfolio_actors"] is True
    assert summary["shared_dispatch_parameters"] is False
    assert summary["shared_portfolio_parameters"] is False
    assert summary["shield_intervention_penalty_in_role_rewards"] is True
    assert summary["portfolio_agent_timescale"] == "slow_loop"
    assert summary["portfolio_decision_step_count"] >= 1
    assert "correction_mean" in result["update_metrics"].columns
    assert "shield_intervention_penalty" in result["step_metrics"].columns
    assert result["episode_metrics"]["critic_loss"].notna().all()


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_shared_rollout_multi_worker_completes_one_update(tmp_path):
    progress_events: list[dict[str, object]] = []
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "happo_shared_rollout",
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=8,
            hidden_dim=16,
            ppo_epochs=1,
            seed=53,
            device="cpu",
            shared_rollout_enabled=True,
            shared_rollout_workers=2,
            shared_rollout_backend="serial",
            rollout_fragment_steps=2,
            critic_use_action_summary=False,
        ),
        progress_callback=progress_events.append,
        progress_step_interval=1,
    )

    summary = result["summary"]
    assert summary["shared_rollout_enabled"] is True
    assert summary["shared_rollout_workers"] == 2
    assert summary["shared_rollout_backend"] == "serial"
    assert summary["rollout_fragment_steps"] == 2
    assert summary["shared_rollout_batches"] == 1
    assert summary["shared_rollout_total_samples"] == 4
    assert summary["shared_rollout_fragment_cut_count"] == 2
    assert summary["shared_rollout_policy_version_mismatch_count"] == 0
    assert summary["shared_rollout_worker_start_offsets"] == {"0": 0, "1": 2}
    assert "shared_rollout_bootstrap_value_mean" in summary
    assert set(result["step_metrics"]["worker_index"]) == {0, 1}
    assert not result["update_metrics"].empty
    assert "policy_version" in result["update_metrics"].columns
    assert "worker_count" in result["update_metrics"].columns
    assert "num_workers" in result["update_metrics"].columns
    assert result["update_metrics"]["worker_count"].eq(2).all()
    assert result["update_metrics"]["num_workers"].eq(2).all()
    assert result["update_metrics"]["ratio_mean"].notna().all()
    assert {event["phase"] for event in progress_events} == {"shared_rollout_step", "train_update"}
    rollout_events = [event for event in progress_events if event["phase"] == "shared_rollout_step"]
    update_events = [event for event in progress_events if event["phase"] == "train_update"]
    assert {event["worker_index"] for event in rollout_events} == {0, 1}
    assert all(event["policy_version"] == "happo_sequential_ctde:shared_rollout:episode=0" for event in rollout_events)
    assert len(update_events) == 1
    update = update_events[0]
    assert update["gradient_step"] == 1
    assert update["global_step"] == 8
    assert {"critic_loss", "dso_policy_loss", "dispatch_policy_loss", "portfolio_policy_loss"}.issubset(update)
    trace = result["dispatch_private_profit_trace"]
    assert {
        "worker_index",
        "worker_start_step",
        "policy_version",
        "policy_normalized_aggregate_action",
        "policy_normalized_der_action_mean",
        "policy_normalized_der_action_std",
    }.issubset(trace.columns)
    assert set(trace["worker_index"]) == {0, 1}
    assert set(trace["worker_start_step"]) == {0, 2}
    assert set(trace["policy_version"]) == {"happo_sequential_ctde:shared_rollout:episode=0"}


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_shared_rollout_fragments_continue_worker_time_axis(tmp_path):
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "happo_shared_rollout_continuous_fragments",
        config=HAPPOConfig(
            episodes=2,
            horizon_steps=8,
            hidden_dim=16,
            ppo_epochs=1,
            seed=54,
            device="cpu",
            shared_rollout_enabled=True,
            shared_rollout_workers=2,
            shared_rollout_backend="serial",
            rollout_fragment_steps=2,
            critic_use_action_summary=False,
        ),
    )

    step_metrics = result["step_metrics"]
    first_worker_steps = {
        int(episode): list(group["step"].astype(int))
        for episode, group in step_metrics[step_metrics["worker_index"] == 0].groupby("episode")
    }
    second_worker_steps = {
        int(episode): list(group["step"].astype(int))
        for episode, group in step_metrics[step_metrics["worker_index"] == 1].groupby("episode")
    }

    assert first_worker_steps == {0: [0, 1], 1: [2, 3]}
    assert second_worker_steps == {0: [2, 3], 1: [4, 5]}


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_shared_rollout_workers_start_on_distinct_time_slices(tmp_path):
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "happo_shared_rollout_staggered_workers",
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=8,
            hidden_dim=16,
            ppo_epochs=1,
            seed=55,
            device="cpu",
            shared_rollout_enabled=True,
            shared_rollout_workers=2,
            shared_rollout_backend="serial",
            rollout_fragment_steps=2,
            critic_use_action_summary=False,
        ),
    )

    step_metrics = result["step_metrics"]
    first_update_steps = {
        int(worker_index): list(group["step"].astype(int))
        for worker_index, group in step_metrics.groupby("worker_index")
    }

    assert first_update_steps == {0: [0, 1], 1: [2, 3]}


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_shared_rollout_subprocess_backend_completes_one_update(tmp_path):
    result = train_happo(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=tmp_path / "happo_shared_rollout_subprocess",
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=8,
            hidden_dim=16,
            ppo_epochs=1,
            seed=56,
            device="cpu",
            shared_rollout_enabled=True,
            shared_rollout_workers=2,
            shared_rollout_backend="subprocess",
            rollout_fragment_steps=2,
            critic_use_action_summary=False,
        ),
    )

    summary = result["summary"]
    assert summary["shared_rollout_enabled"] is True
    assert summary["shared_rollout_backend"] == "subprocess"
    assert summary["shared_rollout_workers"] == 2
    assert summary["shared_rollout_worker_start_offsets"] == {"0": 0, "1": 2}
    assert summary["shared_rollout_subprocess_worker_pids"]
    assert len(summary["shared_rollout_subprocess_worker_pids"]) == 2
    assert summary["shared_rollout_subprocess_worker_exitcodes"] == {"0": 0, "1": 0}
    assert summary["shared_rollout_total_samples"] == 4

    step_metrics = result["step_metrics"]
    subprocess_steps = {
        int(worker_index): list(group["step"].astype(int))
        for worker_index, group in step_metrics.groupby("worker_index")
    }
    assert subprocess_steps == {0: [0, 1], 1: [2, 3]}

    update_metrics = result["update_metrics"]
    required_perf_columns = {
        "shared_rollout_backend",
        "num_workers",
        "rollout_collect_seconds",
        "policy_forward_seconds",
        "env_step_wall_seconds",
        "env_step_worker_mean_seconds",
        "env_step_worker_max_seconds",
        "wait_for_workers_seconds",
        "total_update_seconds",
        "samples_collected",
        "samples_per_second",
        "slowest_worker_id",
        "ratio_mean_before_first_update",
        "ratio_std_before_first_update",
        "old_log_prob_nan_count",
        "new_log_prob_nan_count",
        "advantage_mean",
        "advantage_std",
        "return_mean",
        "return_std",
    }
    assert required_perf_columns.issubset(update_metrics.columns)
    assert update_metrics["shared_rollout_backend"].eq("subprocess").all()
    assert update_metrics["num_workers"].eq(2).all()
    assert update_metrics["samples_collected"].eq(4).all()
    assert (update_metrics["rollout_collect_seconds"] > 0).all()
    assert (update_metrics["env_step_wall_seconds"] > 0).all()
    assert (update_metrics["samples_per_second"] > 0).all()
    assert update_metrics["ratio_mean_before_first_update"].iloc[0] == pytest.approx(1.0, abs=1e-5)
    assert update_metrics["ratio_std_before_first_update"].iloc[0] < 1e-4
    assert update_metrics["old_log_prob_nan_count"].eq(0).all()
    assert update_metrics["new_log_prob_nan_count"].eq(0).all()


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_reward_v2_step_metrics_cover_reward_and_security_columns(tmp_path):
    output_dir = tmp_path / "happo_reward_v2_columns"
    result = train_happo(
        config_path=Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
        output_dir=output_dir,
        config=HAPPOConfig(
            episodes=1,
            horizon_steps=1,
            hidden_dim=16,
            ppo_epochs=1,
            seed=91,
            device="cpu",
        ),
    )

    missing = REWARD_V2_REQUIRED_STEP_COLUMNS.difference(result["step_metrics"].columns)
    assert not missing

    trace_path = output_dir / "happo_dispatch_private_profit_trace.csv"
    episode_trace_path = output_dir / "happo_dispatch_private_profit_trace_episode_0000.csv"
    reward_cards_dir = output_dir / "reports" / "reward_dynamic_cards"
    assert trace_path.exists()
    assert episode_trace_path.exists()
    assert (reward_cards_dir / "reward_dynamic_cards_happo_sequential_ctde_episode_0000.html").exists()
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


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_hasac_training_and_frozen_eval_run():
    output_dir = Path("outputs") / "test_hasac_training"
    train = train_hasac(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir / "train",
        config=HASACConfig(
            episodes=1,
            horizon_steps=3,
            batch_size=2,
            warmup_steps=2,
            hidden_dim=16,
            replay_capacity=64,
            seed=61,
        ),
    )

    summary = train["summary"]

    assert summary["algorithm"] == "hasac_continuous_dispatch"
    assert summary["soft_actor_critic"] is True
    assert summary["twin_soft_q"] is True
    assert summary["off_policy_replay"] is True
    assert summary["automatic_entropy_tuning"] is True
    assert summary["shield_intervention_penalty_in_role_rewards"] is True
    assert summary["per_vpp_dispatch_actors"] is True
    assert summary["shared_dispatch_parameters"] is False
    assert summary["critic_updates"] > 0
    assert summary["actor_updates"] > 0
    assert summary["alpha_updates"] > 0
    assert summary["portfolio_scope"] == "held_keep_discrete_slow_loop_not_hasac"
    assert "alpha_dso" in train["update_metrics"].columns
    assert "alpha_dispatch" in train["update_metrics"].columns
    assert "shield_intervention_penalty_mean" in train["update_metrics"].columns

    eval_result = evaluate_hasac_checkpoint(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        checkpoint_path=train["checkpoint"],
        output_dir=output_dir / "eval",
        horizon_steps=2,
        seed=62,
    )

    assert eval_result["summary"]["evaluation_mode"] == "frozen_mean_actor"
    assert eval_result["summary"]["total_violation_count"] >= 0
    assert not eval_result["step_metrics"].empty
    assert (output_dir / "eval" / "hasac_frozen_eval_summary.json").exists()


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_hasac_training_records_resolved_device_and_reward_artifacts():
    output_dir = Path("outputs") / "test_hasac_device_reward_artifacts"
    train = train_hasac(
        config_path=Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
        output_dir=output_dir,
        config=HASACConfig(
            episodes=1,
            horizon_steps=2,
            batch_size=2,
            warmup_steps=2,
            hidden_dim=16,
            replay_capacity=64,
            seed=63,
            device="cpu",
        ),
    )

    summary = train["summary"]

    assert summary["requested_device"] == "cpu"
    assert summary["resolved_device"] == "cpu"
    assert summary["device_meta"]["resolved_device"] == "cpu"
    assert summary["reward_version"] == "v2_minimal"
    assert summary["critic_reward_scale"] == pytest.approx(0.01)
    assert (output_dir / "resolved_reward_config.yaml").exists()
    assert (output_dir / "reward_config_hash.txt").exists()
    assert summary["reward_config_hash"]
