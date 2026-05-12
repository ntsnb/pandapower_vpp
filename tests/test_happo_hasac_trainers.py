from __future__ import annotations

import pytest

from vpp_dso_sim.learning.advanced_trainers import (
    HASACReplayBuffer,
    advanced_algorithm_capability_rows,
    build_hasac_twin_soft_q,
    build_squashed_gaussian_actor,
    happo_sequential_surrogate_loss,
    hasac_actor_alpha_loss,
    hasac_soft_critic_loss,
    torch_available,
)
from vpp_dso_sim.learning.reward_contracts import shield_intervention_metrics


def test_shield_intervention_metrics_prioritize_local_and_ac_gaps():
    metrics = shield_intervention_metrics(
        {
            "action_projection_gap_mw": 9.0,
            "local_bounds_projection_gap_mw": 0.25,
            "ac_aware_projection_gap_mw": 0.75,
        }
    )

    assert metrics["shield_intervention_gap_mw"] == pytest.approx(1.0)
    assert metrics["shield_intervention_penalty"] > 0.0
    assert metrics["shield_intervention_count"] == 1.0


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_happo_sequential_update_applies_importance_correction():
    import torch

    role_order = ("dso_global_guidance", "vpp_dispatch", "vpp_portfolio")
    old = {role: torch.zeros(4) for role in role_order}
    new = {
        "dso_global_guidance": torch.log(torch.full((4,), 1.10)),
        "vpp_dispatch": torch.log(torch.full((4,), 0.90)),
        "vpp_portfolio": torch.log(torch.full((4,), 1.05)),
    }
    advantages = {role: torch.ones(4) for role in role_order}

    loss, diagnostics = happo_sequential_surrogate_loss(
        new_log_probs_by_role=new,
        old_log_probs_by_role=old,
        advantages_by_role=advantages,
        role_order=role_order,
        clip_ratio=0.20,
        torch_module=torch,
    )

    assert torch.isfinite(loss)
    assert diagnostics["sequential_update"] is True
    assert diagnostics["importance_correction"] is True
    assert diagnostics["dso_global_guidance_correction_mean"].item() == pytest.approx(1.0)
    assert diagnostics["vpp_dispatch_correction_mean"].item() == pytest.approx(1.10)
    assert diagnostics["vpp_portfolio_correction_mean"].item() == pytest.approx(0.99)


def test_hasac_replay_buffer_samples_role_rewards():
    buffer = HASACReplayBuffer(capacity=4, seed=3)
    for index in range(4):
        buffer.add(
            {
                "state": [float(index), 0.0],
                "action": [0.1],
                "reward": [1.0, -0.5, 0.25],
                "next_state": [float(index + 1), 0.0],
                "done": 0.0,
            }
        )

    batch = buffer.sample(2)

    assert batch["state"].shape == (2, 2)
    assert batch["reward"].shape == (2, 3)


@pytest.mark.skipif(not torch_available(), reason="PyTorch is not installed")
def test_hasac_soft_actor_critic_losses_are_finite_and_multi_head():
    import torch

    state_dim = 5
    action_dim = 3
    head_count = 4
    actor = build_squashed_gaussian_actor(obs_dim=state_dim, action_dim=action_dim, hidden_dim=16, torch_module=torch)
    critic = build_hasac_twin_soft_q(
        state_dim=state_dim,
        joint_action_dim=action_dim,
        output_dim=head_count,
        hidden_dim=16,
        torch_module=torch,
    )
    target_critic = build_hasac_twin_soft_q(
        state_dim=state_dim,
        joint_action_dim=action_dim,
        output_dim=head_count,
        hidden_dim=16,
        torch_module=torch,
    )
    state = torch.zeros(6, state_dim)
    next_state = torch.ones(6, state_dim) * 0.1
    action, log_prob, _ = actor.sample(state)
    next_action, next_log_prob, _ = actor.sample(next_state)
    reward = torch.ones(6, head_count) * 0.2
    done = torch.zeros(6)
    log_alpha = torch.tensor(-1.0, requires_grad=True)

    critic_loss, target_q = hasac_soft_critic_loss(
        critic=critic,
        target_critic=target_critic,
        state=state,
        action=action,
        reward=reward,
        next_state=next_state,
        next_action=next_action,
        next_log_prob=next_log_prob,
        done=done,
        alpha=log_alpha.exp(),
        gamma=0.95,
        torch_module=torch,
    )
    actor_loss, alpha_loss, alpha = hasac_actor_alpha_loss(
        critic=critic,
        state=state,
        sampled_action=action,
        log_prob=log_prob,
        log_alpha=log_alpha,
        target_entropy=-float(action_dim),
        torch_module=torch,
    )

    assert target_q.shape == (6, head_count)
    assert torch.isfinite(critic_loss)
    assert torch.isfinite(actor_loss)
    assert torch.isfinite(alpha_loss)
    assert alpha.item() > 0.0


def test_advanced_algorithm_capability_rows_expose_ui_metadata():
    rows = advanced_algorithm_capability_rows()
    by_algorithm = {row["algorithm"]: row for row in rows}

    assert "matd3_continuous_dispatch" in by_algorithm
    assert "happo_sequential_ctde" in by_algorithm
    assert "hatrpo_trust_region_ctde" in by_algorithm
    assert "hasac_soft_actor_critic" in by_algorithm
    assert "per_vpp_q_heads" in by_algorithm["matd3_continuous_dispatch"]["implemented_mechanisms"]
    assert "importance_correction" in by_algorithm["happo_sequential_ctde"]["implemented_mechanisms"]
    assert "fisher_vector_product" in by_algorithm["hatrpo_trust_region_ctde"]["implemented_mechanisms"]
    assert "entropy_temperature" in by_algorithm["hasac_soft_actor_critic"]["implemented_mechanisms"]
