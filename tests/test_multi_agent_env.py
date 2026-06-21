from __future__ import annotations

from pathlib import Path

import pytest

from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv


def test_multi_agent_env_reset_and_step_returns_parallel_dicts():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=2,
    )

    observations, infos = env.reset(seed=7)
    assert "dso_global_guidance" in observations
    assert any(agent.endswith("_dispatch") for agent in observations)
    assert any(agent.endswith("_portfolio") for agent in observations)
    assert set(observations) == set(env.agents)
    assert "agent_role_map" in infos["dso_global_guidance"]
    assert infos["dso_global_guidance"]["centralized_critic_spec"]["observation_type"] == "critic_global_state"
    first_vpp = env.scenario.vpps[0].id
    assert infos[f"{first_vpp}_dispatch"]["ctde_actor_spec"]["policy_module_id"] == f"{first_vpp}_dispatch_actor"

    targets = {vpp.id: 0.0 for vpp in env.scenario.vpps}
    next_obs, rewards, terminations, truncations, step_infos = env.step(
        {
            "dso_global_guidance": {"targets": targets},
            f"{env.scenario.vpps[0].id}_portfolio": {"action": "keep"},
        }
    )

    assert set(next_obs) == set(env.agents)
    assert set(rewards) == set(env.agents)
    assert not any(terminations.values())
    assert not all(truncations.values())
    assert step_infos["dso_global_guidance"]["decoded_dso_targets"] == targets
    assert step_infos["dso_global_guidance"]["action_validation"]["is_valid"] is True
    assert step_infos["dso_global_guidance"]["critic_global_state"]["observation_type"] == "critic_global_state"
    assert step_infos["dso_global_guidance"]["training_only_critic_state"]["observation_type"] == "critic_global_state"
    assert step_infos["dso_global_guidance"]["critic_state_visibility"] == "training_only_not_actor_observation"
    assert step_infos["dso_global_guidance"]["agent_reward_components"]["dso_reward"] == rewards["dso_global_guidance"]
    dispatch_agent = f"{env.scenario.vpps[0].id}_dispatch"
    portfolio_agent = f"{env.scenario.vpps[0].id}_portfolio"
    assert "private_profit_proxy" in step_infos[dispatch_agent]["agent_reward_components"]
    assert "localized_dso_alignment_reward" in step_infos[portfolio_agent]["agent_reward_components"]
    assert rewards[dispatch_agent] != rewards["dso_global_guidance"]
    env.close()


def test_vpp_actor_observation_does_not_expose_other_vpp_private_costs():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset()
    dispatch_agent = next(agent for agent in observations if agent.endswith("_dispatch"))
    owner_vpp_id = dispatch_agent.removesuffix("_dispatch")
    obs_text = str(observations[dispatch_agent])
    other_vpp_ids = {vpp.id for vpp in env.scenario.vpps if vpp.id != owner_vpp_id}

    assert "cost_coefficients" in obs_text
    assert all(asset["owner_vpp_id"] == owner_vpp_id for asset in observations[dispatch_agent]["local_assets"])
    assert not any(other_vpp_id in obs_text for other_vpp_id in other_vpp_ids)
    assert "critic_global_state" not in observations[dispatch_agent]
    assert observations[dispatch_agent]["observation_type"] == "actor_observation_i"
    assert "operating_envelope" in observations[dispatch_agent]
    assert "service_signal" in observations[dispatch_agent]
    assert "dispatch_award" in observations[dispatch_agent]
    assert "der_mask" in observations[dispatch_agent]
    portfolio_agent = dispatch_agent.replace("_dispatch", "_portfolio")
    assert observations[portfolio_agent]["trainable_action_current_version"] is True
    assert observations[portfolio_agent]["physical_change_allowed"] is False
    env.close()


def test_vpp_dispatch_agent_forwards_der_level_actions_to_simulator():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset(seed=11)
    vpp = env.scenario.vpps[0]
    dispatch_agent = f"{vpp.id}_dispatch"
    envelope = observations[dispatch_agent]["operating_envelope"]
    selected_p = float(envelope["preferred_target_p_mw"])
    der_actions = {der.id: (-1.0 if index % 2 else 1.0) for index, der in enumerate(vpp.der_list)}

    _, _, _, _, infos = env.step(
        {
            dispatch_agent: {
                "selected_p_mw": selected_p,
                "der_actions": der_actions,
            }
        }
    )

    dso_info = infos["dso_global_guidance"]
    payload = dso_info["decoded_simulator_action_payload"][vpp.id]
    audit = dso_info["decoded_vpp_dispatch_adjustments"][vpp.id]
    results = env.simulator.collect_results()
    rl_dispatch = results["vpp_rl_disaggregation"]
    selected = rl_dispatch[rl_dispatch["vpp_id"] == vpp.id]

    assert payload["action_mode"] == "learned_der_disaggregation"
    assert audit["uses_learned_der_actions"] is True
    assert audit["der_action_count"] == len(vpp.der_list)
    assert not selected.empty
    assert selected["is_learned_der_action"].all()
    assert selected["projection_gap_mw"].max() < 1e-6
    env.close()


def test_out_of_envelope_dispatch_is_penalized_after_projection():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset(seed=12)
    vpp = env.scenario.vpps[0]
    dispatch_agent = f"{vpp.id}_dispatch"
    envelope = observations[dispatch_agent]["operating_envelope"]
    impossible_target = float(envelope["p_max_mw"]) + 10.0

    _, _, _, _, infos = env.step({dispatch_agent: {"selected_p_mw": impossible_target}})

    dso_info = infos["dso_global_guidance"]
    audit = dso_info["decoded_vpp_dispatch_adjustments"][vpp.id]
    components = dso_info["reward_components"]

    assert audit["projection_clipped"] is True
    assert audit["projection_gap_mw"] > 0.0
    assert audit["local_bounds_projection_gap_mw"] == audit["projection_gap_mw"]
    assert audit["projection_gap_scope"] == "local_der_bounds_not_ac_security"
    assert components["action_projection_gap_mw"] > 0.0
    assert components["local_bounds_projection_gap_mw"] == components["action_projection_gap_mw"]
    assert "post_ac_violation_count" in components
    assert components["action_projection_penalty"] > 0.0
    assert components["total_cost"] >= components["action_projection_penalty"]
    env.close()


def test_dispatch_agent_reward_components_include_action_landing_audit():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset(seed=13)
    vpp = env.scenario.vpps[0]
    dispatch_agent = f"{vpp.id}_dispatch"
    envelope = observations[dispatch_agent]["operating_envelope"]
    selected_p = float(envelope["preferred_target_p_mw"])
    baseline_before = float(vpp.current_power_mw())

    _, _, _, _, infos = env.step({dispatch_agent: {"selected_p_mw": selected_p}})

    components = infos[dispatch_agent]["agent_reward_components"]
    for key in (
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
    ):
        assert key in components
    assert components["baseline_p_mw"] == pytest.approx(baseline_before)
    assert components["decoded_target_p_mw"] == pytest.approx(selected_p)
    assert components["action_landing_ratio"] >= 0.0
    env.close()
