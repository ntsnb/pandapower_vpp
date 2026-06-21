from __future__ import annotations

from pathlib import Path

import pytest

from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
from vpp_dso_sim.learning.ctde_networks import split_vpp_dispatch_vector
from vpp_dso_sim.learning.deep_rl import (
    DeepRLConfig,
    PrivacySeparatedCTDEConfig,
    _gae_returns_advantages,
    _gae_returns_advantages_bootstrap,
    _ppo_clipped_policy_loss,
    encode_joint_action_summary,
    evaluate_privacy_separated_ctde_checkpoint,
    joint_action_summary_dim,
    torch_available,
    train_deep_rl_actor_critic,
    train_privacy_separated_ctde,
)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_deep_rl_smoke_updates_weights_and_writes_artifacts():
    output_dir = Path("outputs") / "test_deep_rl_training"
    result = train_deep_rl_actor_critic(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir,
        config=DeepRLConfig(episodes=1, horizon_steps=2, hidden_dim=16, seed=7),
    )

    summary = result["summary"]

    assert summary["is_deep_rl"] is True
    assert summary["deep_learning_framework"] == "torch"
    assert summary["optimizer_steps"] > 0
    assert summary["param_delta_l2"] > 0.0
    assert summary["dso_actor_trainable"] is True
    assert summary["vpp_dispatch_trainable"] is True
    assert summary["vpp_der_disaggregation_trainable"] is True
    assert summary["dispatch_action_type"] == "der_level_normalized_setpoints"
    assert summary["portfolio_trainable"] is True
    assert summary["portfolio_action_type"] == "slow_loop_commercial_configuration_proposal"
    assert summary["portfolio_physical_change_gated"] is True
    assert "portfolio_proxy_reward" in result["episode_metrics"].columns
    assert "portfolio_action" in result["trajectory"].columns
    assert result["checkpoint"].exists()
    assert (output_dir / "deep_rl_episode_metrics.csv").exists()
    assert (output_dir / "deep_rl_step_metrics.csv").exists()
    assert (output_dir / "deep_rl_loss_metrics.csv").exists()
    assert not result["loss_metrics"].empty


def test_vpp_dispatch_action_changes_decoded_simulator_targets():
    env = MultiAgentVPPDSOEnv(config_path=Path("configs") / "european_lv_mixed_vpp.yaml", horizon_steps=1)
    observations, _ = env.reset(seed=3)
    vpp_id = env.scenario.vpps[0].id
    report = observations["dso_global_guidance"]["vpp_reports"][vpp_id]
    base_target = 0.5 * (float(report["p_min_mw"]) + float(report["p_max_mw"]))

    _, _, _, _, infos = env.step(
        {
            "dso_global_guidance": {"targets": {vpp_id: base_target}},
            f"{vpp_id}_dispatch": {"normalized_setpoint_bias": 1.0},
        }
    )

    dso_info = infos["dso_global_guidance"]
    adjusted = dso_info["decoded_vpp_dispatch_adjustments"][vpp_id]

    assert dso_info["decoded_dso_targets"][vpp_id] == base_target
    assert dso_info["decoded_simulator_targets"][vpp_id] != base_target
    assert adjusted["p_min_mw"] <= adjusted["projected_target_p_mw"] <= adjusted["p_max_mw"]


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_privacy_separated_ctde_trainer_uses_separate_modules_and_writes_artifacts():
    output_dir = Path("outputs") / "test_privacy_separated_ctde"
    result = train_privacy_separated_ctde(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir,
        config=PrivacySeparatedCTDEConfig(episodes=1, horizon_steps=2, hidden_dim=16, seed=11),
    )

    summary = result["summary"]

    assert summary["algorithm"] == "privacy_separated_ctde_actor_critic"
    assert summary["privacy_separated_execution"] is True
    assert summary["dso_vpp_shared_encoder"] is False
    assert summary["homogeneous_vpp_parameter_sharing"] is True
    assert summary["centralized_critic_uses_global_state"] is True
    assert summary["critic_visible_to_decentralized_actors"] is False
    assert summary["target_ctde_primary_trainer"] is True
    assert summary["architecture_version"] == "ctde_v2_deepsets_action_conditioned"
    assert summary["vpp_encoder_type"] == "deep_sets_shared_token_mlp"
    assert summary["critic_type"] == "centralized_action_conditioned_summary_critic"
    assert summary["critic_head_type"] == "role_multi_head_value_baselines"
    assert summary["critic_value_heads"] == "dso,dispatch,portfolio"
    assert summary["role_aware_value_baselines"] is True
    assert summary["policy_update_rule"] == "mappo_happo_lite_gae_single_epoch_clipped_surrogate"
    assert summary["gae_lambda"] == 0.95
    assert summary["ppo_clip_ratio"] == 0.20
    assert summary["action_conditioned_critic"] is True
    assert summary["critic_action_dim"] > 0
    assert summary["critic_action_summary_dim"] == summary["critic_action_dim"]
    assert summary["param_delta_l2"] > 0.0
    assert "dso_policy_loss" in result["loss_metrics"].columns
    assert "vpp_dispatch_policy_loss" in result["loss_metrics"].columns
    assert "portfolio_policy_loss" in result["loss_metrics"].columns
    assert "dso_value_loss" in result["loss_metrics"].columns
    assert "dispatch_value_loss" in result["loss_metrics"].columns
    assert "portfolio_value_loss" in result["loss_metrics"].columns
    assert "policy_update_rule" in result["loss_metrics"].columns
    assert "critic_action_summary_l2" in result["step_metrics"].columns
    assert "dso_value_estimate" in result["step_metrics"].columns
    assert "dispatch_value_estimate" in result["step_metrics"].columns
    assert "portfolio_value_estimate" in result["step_metrics"].columns
    assert "architecture_version" in result["step_metrics"].columns
    assert result["step_metrics"]["critic_action_summary_l2"].max() > 0.0
    assert set(result["step_metrics"]["vpp_encoder_type"]) == {"deep_sets_shared_token_mlp"}
    assert set(result["trajectory"]["privacy_scope"]) == {"own_vpp_local_observation_only"}
    assert result["checkpoint"].exists()
    assert (output_dir / "privacy_separated_ctde_checkpoint.pt").exists()
    assert (output_dir / "deep_rl_training_summary.csv").exists()
    import torch

    checkpoint = torch.load(result["checkpoint"], map_location="cpu")
    keys = set(checkpoint["model_state_dict"].keys())
    assert "vpp_dispatch_actor.encoder.token_mlp.0.weight" in keys
    assert "centralized_critic.action_encoder.net.1.weight" in keys
    assert "centralized_critic.dso_value.weight" in keys
    assert "centralized_critic.dispatch_value.weight" in keys
    assert "centralized_critic.portfolio_value.weight" in keys
    assert checkpoint["architecture_meta"]["critic_type"] == "centralized_action_conditioned_summary_critic"
    assert checkpoint["architecture_meta"]["critic_value_heads"] == "dso,dispatch,portfolio"


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_privacy_separated_ctde_checkpoint_frozen_eval_writes_artifacts():
    output_dir = Path("outputs") / "test_privacy_separated_ctde_eval"
    train = train_privacy_separated_ctde(
        config_path=Path("configs") / "european_lv_benchmark_v2.yaml",
        output_dir=output_dir / "train",
        config=PrivacySeparatedCTDEConfig(episodes=1, horizon_steps=2, hidden_dim=16, seed=13),
    )
    eval_result = evaluate_privacy_separated_ctde_checkpoint(
        config_path=Path("configs") / "european_lv_benchmark_v2.yaml",
        checkpoint_path=train["checkpoint"],
        output_dir=output_dir / "eval",
        horizon_steps=2,
        seed=14,
    )

    assert eval_result["summary"]["evaluation_mode"] == "frozen_deterministic_mean_policy"
    assert eval_result["summary"]["total_violation_count"] >= 0
    assert eval_result["summary"]["critic_value_heads"] == "dso,dispatch,portfolio"
    assert not eval_result["step_metrics"].empty
    assert (output_dir / "eval" / "frozen_eval_summary.csv").exists()
    assert (output_dir / "eval" / "simulator_results" / "summary.json").exists()


def test_joint_action_summary_shape_is_stable_for_critic():
    vector = encode_joint_action_summary(
        normalized_dso_action=[0.1, -0.2],
        vpp_ids=["vpp_a", "vpp_b"],
        normalized_aggregate_actions={"vpp_a": 0.3, "vpp_b": -0.4},
        normalized_der_actions={"vpp_a": [0.1, 0.2, 0.3], "vpp_b": [-0.1, -0.2, -0.3]},
        portfolio_action_indices={"vpp_a": 1, "vpp_b": 2},
        max_der_per_vpp=3,
        action_clip=1.0,
    )

    assert vector.shape == (joint_action_summary_dim(2),)
    assert vector.dtype.name == "float32"


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_gae_and_ppo_clip_helpers_are_finite_and_clipped():
    import torch

    values = torch.tensor([0.10, 0.20, 0.05], dtype=torch.float32)
    returns, advantages = _gae_returns_advantages(
        rewards=[1.0, 0.5, -0.25],
        values=values,
        gamma=0.90,
        gae_lambda=0.80,
        torch=torch,
    )

    assert returns.shape == values.shape
    assert advantages.shape == values.shape
    assert torch.isfinite(returns).all()
    assert torch.isfinite(advantages).all()

    log_probs = torch.log(torch.tensor([1.50, 0.20, 1.00], dtype=torch.float32))
    old_log_probs = torch.zeros(3, dtype=torch.float32)
    loss, ratios = _ppo_clipped_policy_loss(
        log_probs=log_probs,
        old_log_probs=old_log_probs,
        advantages=torch.tensor([1.0, -1.0, 0.5], dtype=torch.float32),
        clip_ratio=0.20,
        torch=torch,
    )

    assert torch.isfinite(loss)
    assert ratios.max() > 1.20
    clipped = torch.clamp(ratios, 0.80, 1.20)
    assert float(clipped.max()) == pytest.approx(1.20)
    assert float(clipped.min()) == pytest.approx(0.80)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_gae_bootstrap_fragment_cut_uses_next_value():
    import torch

    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    returns, advantages = _gae_returns_advantages_bootstrap(
        rewards=[0.0, 0.0],
        values=values,
        next_value=torch.tensor(10.0),
        terminals=[False, False],
        gamma=0.5,
        gae_lambda=1.0,
        torch=torch,
    )

    assert torch.allclose(returns, torch.tensor([2.5, 5.0]), atol=1e-6)
    assert torch.allclose(advantages, returns - values, atol=1e-6)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_gae_bootstrap_true_terminal_ignores_next_value():
    import torch

    values = torch.tensor([1.0, 2.0], dtype=torch.float32)
    returns, advantages = _gae_returns_advantages_bootstrap(
        rewards=[0.0, 0.0],
        values=values,
        next_value=torch.tensor(10.0),
        terminals=[False, True],
        gamma=0.5,
        gae_lambda=1.0,
        torch=torch,
    )

    assert torch.allclose(returns, torch.tensor([0.0, 0.0]), atol=1e-6)
    assert torch.allclose(advantages, returns - values, atol=1e-6)


def test_vpp_dispatch_vector_splits_into_context_and_der_tokens():
    vector = [float(index) for index in range(16 + 2 * 26)]
    context, tokens, mask = split_vpp_dispatch_vector(vector)

    assert context.shape == (1, 16)
    assert tokens.shape == (1, 2, 26)
    assert mask.shape == (1, 2)
    assert context[0, 0] == 0.0
    assert tokens[0, 0, 0] == 16.0
    assert tokens[0, 1, -1] == 67.0
    assert mask.all()


def test_vpp_dispatch_der_actions_are_accepted_by_env():
    env = MultiAgentVPPDSOEnv(config_path=Path("configs") / "european_lv_mixed_vpp.yaml", horizon_steps=1)
    observations, _ = env.reset(seed=4)
    vpp = env.scenario.vpps[0]
    envelope = observations[f"{vpp.id}_dispatch"]["operating_envelope"]
    der_actions = {der.id: 0.25 for der in vpp.der_list}

    _, _, _, _, infos = env.step(
        {
            f"{vpp.id}_dispatch": {
                "selected_p_mw": float(envelope["preferred_target_p_mw"]),
                "der_actions": der_actions,
            }
        }
    )

    dso_info = infos["dso_global_guidance"]
    payload = dso_info["decoded_simulator_action_payload"][vpp.id]
    adjusted = dso_info["decoded_vpp_dispatch_adjustments"][vpp.id]
    results = env.simulator.collect_results()
    rl_dispatch = results["vpp_rl_disaggregation"]
    selected = rl_dispatch[rl_dispatch["vpp_id"] == vpp.id]

    assert payload["action_mode"] == "learned_der_disaggregation"
    assert adjusted["uses_learned_der_actions"] is True
    assert adjusted["der_action_count"] == len(vpp.der_list)
    assert not selected.empty
    assert selected["is_learned_der_action"].all()
    assert selected["normalized_der_action"].notna().all()
    assert selected["projection_gap_mw"].max() < 1e-6
    env.close()
