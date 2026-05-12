from __future__ import annotations

import pytest

from vpp_dso_sim.learning.advanced_marl import (
    AlgorithmCandidate,
    ScoringWeights,
    TwinCriticSpec,
    build_matd3_twin_critic_spec,
    build_twin_critic,
    candidate_by_key,
    get_algorithm_registry,
    rank_algorithm_candidates,
)


def test_advanced_marl_registry_has_required_algorithm_candidates():
    registry = get_algorithm_registry()
    keys = {candidate.key for candidate in registry}

    assert len(registry) >= 20
    assert {"matd3", "happo", "hatrpo", "hasac"}.issubset(keys)
    assert candidate_by_key("matd3").critic_scope == "centralized_training_only"
    assert "continuous" in candidate_by_key("hasac").action_support
    assert candidate_by_key("happo").heterogeneity_support == "native"
    assert candidate_by_key("hatrpo").family == "trust_region_policy_gradient"


def test_algorithm_ranking_is_stable_and_prefers_privacy_preserving_ctde():
    first = rank_algorithm_candidates()
    second = rank_algorithm_candidates()

    assert [score.candidate_key for score in first] == [score.candidate_key for score in second]
    assert [score.total_score for score in first] == sorted(
        [score.total_score for score in first],
        reverse=True,
    )
    assert first[0].privacy_fit >= 0.9
    assert first[0].continuous_action_fit >= 0.9
    assert "qmix" in {score.candidate_key for score in first}
    assert first.index(next(score for score in first if score.candidate_key == "matd3")) < first.index(
        next(score for score in first if score.candidate_key == "qmix")
    )


def test_algorithm_ranking_uses_input_order_as_tie_breaker_for_custom_candidates():
    base = candidate_by_key("matd3")
    first = AlgorithmCandidate(
        **{
            **base.to_dict(),
            "key": "tie_first",
            "display_name": "Tie First",
            "action_support": tuple(base.action_support),
        }
    )
    second = AlgorithmCandidate(
        **{
            **base.to_dict(),
            "key": "tie_second",
            "display_name": "Tie Second",
            "action_support": tuple(base.action_support),
        }
    )

    scores = rank_algorithm_candidates([second, first])

    assert [score.candidate_key for score in scores] == ["tie_second", "tie_first"]


def test_custom_weights_can_emphasize_low_implementation_risk():
    scores = rank_algorithm_candidates(
        weights=ScoringWeights(
            privacy_fit=0.0,
            continuous_action_fit=0.0,
            heterogeneity_fit=0.0,
            general_sum_fit=0.0,
            ctde_fit=0.0,
            implementation_risk_fit=1.0,
        )
    )

    assert scores[0].implementation_risk == min(score.implementation_risk for score in scores)


def test_matd3_twin_critic_spec_records_training_only_privacy_boundary():
    spec = build_matd3_twin_critic_spec(
        critic_global_state_dim=59,
        joint_action_dim=64,
        hidden_dims=(32, 16),
    )
    payload = spec.to_dict()

    assert spec.input_dim == 123
    assert spec.critic_scope == "centralized_training_only"
    assert spec.algorithm_style == "matd3_centralized_twin_q"
    assert payload["hidden_dims"] == [32, 16]
    assert "all_agent_continuous_actions" in spec.input_contract


def test_twin_critic_builds_when_torch_is_available():
    torch = pytest.importorskip("torch")
    spec = TwinCriticSpec(state_dim=5, joint_action_dim=3, hidden_dims=(8, 8))

    critic = build_twin_critic(spec, require_torch=True)
    q1, q2 = critic(torch.zeros(2, 5), torch.zeros(2, 3))

    assert q1.shape == (2, 1)
    assert q2.shape == (2, 1)
    assert critic.q1_value(torch.zeros(2, 5), torch.zeros(2, 3)).shape == (2, 1)
