from __future__ import annotations

import importlib.util
from pathlib import Path

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
