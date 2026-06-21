from __future__ import annotations

from pathlib import Path

import pytest

from vpp_dso_sim.learning.matd3 import (
    MATD3Config,
    MATD3ReplayBuffer,
    _soft_update,
    evaluate_matd3_checkpoint,
    torch_available,
    train_matd3,
)


def test_matd3_replay_buffer_samples_expected_shapes():
    buffer = MATD3ReplayBuffer(capacity=4, seed=1)
    for index in range(4):
        buffer.add(
            {
                "dso_obs": [float(index), 0.0],
                "vpp_obs": [[float(index), 1.0]],
                "critic_state": [0.1, 0.2, float(index)],
                "joint_action": [0.0, 0.1],
                "next_dso_obs": [float(index + 1), 0.0],
                "next_vpp_obs": [[float(index + 1), 1.0]],
                "next_critic_state": [0.2, 0.3, float(index + 1)],
                "dso_reward": float(index),
                "dispatch_reward": -float(index),
                "done": 0.0,
            }
        )

    batch = buffer.sample(2)

    assert batch["dso_obs"].shape == (2, 2)
    assert batch["vpp_obs"].shape == (2, 1, 2)
    assert batch["joint_action"].shape == (2, 2)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_matd3_polyak_update_changes_targets_slowly():
    import torch

    source = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    for param in source.parameters():
        param.data.fill_(1.0)
    for param in target.parameters():
        param.data.fill_(0.0)

    _soft_update(source, target, tau=0.1)

    for param in target.parameters():
        assert torch.allclose(param.data, torch.full_like(param.data, 0.1))


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_matd3_training_writes_full_off_policy_artifacts():
    output_dir = Path("outputs") / "test_matd3_training"
    result = train_matd3(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir,
        config=MATD3Config(
            episodes=1,
            horizon_steps=4,
            batch_size=2,
            warmup_steps=2,
            hidden_dim=16,
            replay_capacity=64,
            seed=31,
        ),
    )

    summary = result["summary"]

    assert summary["algorithm"] == "matd3_continuous_dispatch"
    assert summary["matd3_complete_core"] is True
    assert summary["twin_critics"] is True
    assert summary["target_networks"] is True
    assert summary["target_policy_smoothing"] is True
    assert summary["delayed_actor_updates"] is True
    assert summary["shield_intervention_penalty_in_role_rewards"] is True
    assert summary["mean_dispatch_reward_critic"] is False
    assert summary["per_vpp_dispatch_q_heads"] is True
    assert summary["per_vpp_dispatch_actors"] is True
    assert summary["shared_dispatch_parameters"] is False
    assert summary["general_sum_reward_heads"] is True
    assert summary["critic_head_type"] == "role_multi_head_twin_q"
    critic_head_names = summary["critic_head_names"].split(",")
    assert summary["critic_head_count"] == len(critic_head_names)
    assert critic_head_names[0] == "dso_global_guidance"
    assert all(name.endswith("_dispatch") for name in critic_head_names[1:])
    assert summary["critic_updates"] > 0
    assert summary["actor_updates"] > 0
    assert summary["critic_updates"] > summary["actor_updates"]
    assert summary["joint_action_dim"] > 0
    assert summary["portfolio_scope"] == "held_keep_discrete_slow_loop_not_matd3"
    assert result["checkpoint"].exists()
    assert (output_dir / "matd3_update_metrics.csv").exists()
    assert not result["update_metrics"].empty
    assert {"dso_critic_loss", "dispatch_critic_loss", "policy_delay"}.issubset(
        result["update_metrics"].columns
    )
    assert "shield_intervention_penalty_mean" in result["update_metrics"].columns
    assert "shield_intervention_penalty" in result["step_metrics"].columns
    assert result["update_metrics"]["per_vpp_dispatch_q_heads"].eq(True).all()
    assert result["update_metrics"]["critic_head_count"].iloc[-1] == summary["critic_head_count"]
    assert set(result["trajectory"]["privacy_scope"]) == {"own_vpp_local_observation_only"}
    assert result["step_metrics"]["portfolio_policy"].eq("held_keep_not_matd3_discrete").all()


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_matd3_training_records_resolved_device_and_reward_artifacts():
    output_dir = Path("outputs") / "test_matd3_device_summary"
    result = train_matd3(
        config_path=Path("configs") / "european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml",
        output_dir=output_dir,
        config=MATD3Config(
            episodes=1,
            horizon_steps=2,
            batch_size=2,
            warmup_steps=2,
            hidden_dim=16,
            replay_capacity=64,
            seed=33,
            device="cpu",
        ),
    )

    summary = result["summary"]

    assert summary["requested_device"] == "cpu"
    assert summary["resolved_device"] == "cpu"
    assert "cuda_available" in summary
    assert "cuda_device_count" in summary
    assert "cuda_device_name" in summary
    assert result["summary"]["device_meta"]["resolved_device"] == "cpu"
    assert summary["reward_version"] == "v2_minimal"
    assert summary["critic_reward_scale"] == pytest.approx(0.01)
    assert (output_dir / "resolved_reward_config.yaml").exists()
    assert (output_dir / "reward_config_hash.txt").exists()
    assert summary["reward_config_hash"]


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_matd3_checkpoint_frozen_eval_runs():
    output_dir = Path("outputs") / "test_matd3_eval"
    train = train_matd3(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        output_dir=output_dir / "train",
        config=MATD3Config(
            episodes=1,
            horizon_steps=3,
            batch_size=2,
            warmup_steps=2,
            hidden_dim=16,
            replay_capacity=64,
            seed=41,
        ),
    )
    eval_result = evaluate_matd3_checkpoint(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        checkpoint_path=train["checkpoint"],
        output_dir=output_dir / "eval",
        horizon_steps=2,
        seed=42,
    )

    assert eval_result["summary"]["evaluation_mode"] == "frozen_deterministic_actor"
    assert eval_result["summary"]["total_violation_count"] >= 0
    assert not eval_result["step_metrics"].empty
    assert (output_dir / "eval" / "matd3_frozen_eval_summary.json").exists()
