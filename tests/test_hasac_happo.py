from __future__ import annotations

from pathlib import Path

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
