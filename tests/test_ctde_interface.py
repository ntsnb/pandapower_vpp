from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
from vpp_dso_sim.learning.ctde_interface import build_ctde_interface_contract
from vpp_dso_sim.simulation.scenario import load_scenario


def test_shared_ctde_contract_matches_current_shared_actor_critic_layout():
    scenario = load_scenario(Path("configs") / "european_lv_mixed_vpp.yaml")
    contract = build_ctde_interface_contract(scenario.vpps, policy_layout="shared_actor_critic")

    dso_spec = contract.actor_spec_for("dso_global_guidance")
    dispatch_spec = contract.actor_spec_for(f"{scenario.vpps[0].id}_dispatch")
    portfolio_spec = contract.actor_spec_for(f"{scenario.vpps[0].id}_portfolio")

    assert dso_spec is not None
    assert dso_spec.policy_module_id == "shared_actor_critic:dso_head"
    assert dispatch_spec is not None
    assert dispatch_spec.policy_module_id == "shared_actor_critic:vpp_dispatch_head"
    assert portfolio_spec is not None
    assert portfolio_spec.policy_module_id == "shared_actor_critic:vpp_portfolio_head"
    portfolio_module = next(
        module for module in contract.policy_modules if module.module_id == "shared_actor_critic:vpp_portfolio_head"
    )
    assert portfolio_module.trainable is True
    assert portfolio_spec.current_implementation == "trainable_slow_loop_head_with_physical_change_gate"
    assert contract.centralized_critic.observation_type == "critic_global_state"


def test_independent_ctde_contract_can_generate_per_vpp_actor_scaffold():
    scenario = load_scenario(Path("configs") / "european_lv_mixed_vpp.yaml")
    contract = build_ctde_interface_contract(
        scenario.vpps,
        policy_layout="independent_actor_scaffold",
        share_dispatch_policy=False,
        share_portfolio_policy=False,
    )

    for vpp in scenario.vpps:
        dispatch_spec = contract.actor_spec_for(f"{vpp.id}_dispatch")
        portfolio_spec = contract.actor_spec_for(f"{vpp.id}_portfolio")
        assert dispatch_spec is not None
        assert dispatch_spec.policy_module_id == f"{vpp.id}_dispatch_actor"
        assert portfolio_spec is not None
        assert portfolio_spec.policy_module_id == f"{vpp.id}_portfolio_actor"

    module_ids = {module.module_id for module in contract.policy_modules}
    assert "dso_actor" in module_ids
    assert f"{scenario.vpps[0].id}_dispatch_actor" in module_ids


def test_default_ctde_contract_uses_independent_per_vpp_layout():
    scenario = load_scenario(Path("configs") / "european_lv_mixed_vpp.yaml")
    contract = build_ctde_interface_contract(scenario.vpps)

    vpp = scenario.vpps[0]
    assert contract.policy_layout == "independent_actor_scaffold"
    assert contract.actor_spec_for(f"{vpp.id}_dispatch").policy_module_id == f"{vpp.id}_dispatch_actor"
    assert contract.actor_spec_for(f"{vpp.id}_portfolio").policy_module_id == f"{vpp.id}_portfolio_actor"


def test_env_action_validation_normalizes_legacy_and_current_payloads():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=1,
    )
    observations, infos = env.reset(seed=17)
    vpp = env.scenario.vpps[0]
    dispatch_agent = f"{vpp.id}_dispatch"
    portfolio_agent = f"{vpp.id}_portfolio"
    der_actions = {der.id: 0.25 for der in vpp.der_list}

    report = env.validate_action_payload(
        {
            "dso_global_guidance": [0.0] * len(env.scenario.vpps),
            dispatch_agent: {
                "target_p_mw": observations[dispatch_agent]["operating_envelope"]["preferred_target_p_mw"],
                "normalized_der_actions": der_actions,
                "policy_version": "test_policy",
            },
            portfolio_agent: "keep",
        }
    )

    normalized = report["normalized_actions"]
    assert report["is_valid"] is True
    assert normalized["dso_global_guidance"]["targets"][vpp.id] == 0.0
    assert normalized[dispatch_agent]["selected_p_mw"] == observations[dispatch_agent]["operating_envelope"]["preferred_target_p_mw"]
    assert normalized[dispatch_agent]["der_actions"] == der_actions
    assert normalized[portfolio_agent]["action"] == "keep"
    assert infos["dso_global_guidance"]["ctde_actor_spec"]["action_schema_id"] == "dso_guidance_targets_v1"
    assert infos["dso_global_guidance"]["centralized_critic_spec"]["observation_type"] == "critic_global_state"
    env.close()


def test_env_step_surfaces_action_validation_errors_without_breaking_legacy_path():
    env = MultiAgentVPPDSOEnv(
        config_path=Path("configs") / "european_lv_mixed_vpp.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset(seed=23)
    vpp = env.scenario.vpps[0]
    dispatch_agent = f"{vpp.id}_dispatch"

    _, _, _, _, step_infos = env.step(
        {
            "ghost_agent": {"foo": "bar"},
            dispatch_agent: {
                "selected_p_mw": observations[dispatch_agent]["operating_envelope"]["preferred_target_p_mw"],
                "der_actions": {"ghost_der": 0.5},
            },
        }
    )

    validation = step_infos["dso_global_guidance"]["action_validation"]
    fields = {(row["agent_id"], row["field"]) for row in validation["errors"]}

    assert validation["is_valid"] is False
    assert ("ghost_agent", "agent_id") in fields
    assert (dispatch_agent, "ghost_der") in fields
    assert step_infos["dso_global_guidance"]["validated_action"] is None
    env.close()
