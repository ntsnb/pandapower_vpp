from __future__ import annotations

import pytest

from vpp_dso_sim.dso.envelope.sensitivity_attention_v1 import SensitivityAttentionEnvelopePolicy
from vpp_dso_sim.dso.envelope.policy_switch import build_dso_envelope_policy
from vpp_dso_sim.envs.multi_agent_env import MultiAgentVPPDSOEnv
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.optimization.feasibility_region import compute_static_feasible_region
from vpp_dso_sim.simulation.scenario import load_scenario
from vpp_dso_sim.simulation.simulator import Simulator


def test_policy_switch_builds_rule_and_structured_modes() -> None:
    rule = build_dso_envelope_policy({"dso": {"envelope_policy": "rule_v0"}})
    structured = build_dso_envelope_policy({"dso": {"envelope_policy": "sensitivity_attention_v1"}})

    assert rule.policy_name == "rule_v0"
    assert isinstance(structured, SensitivityAttentionEnvelopePolicy)


def test_policy_switch_rejects_unknown_policy_without_silent_fallback() -> None:
    with pytest.raises(ValueError, match="Unsupported DSO envelope_policy"):
        build_dso_envelope_policy({"dso": {"envelope_policy": "unknown_policy"}})


def test_sensitivity_attention_policy_builds_decoded_guidance_envelope() -> None:
    scenario = load_scenario("configs/european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    bid = vpp.day_ahead_bid(0, price_hint=80.0)
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    policy = build_dso_envelope_policy(
        {
            "dso": {
                "envelope_policy": "sensitivity_attention_v1",
                "action_unit_granularity": "vpp_bus",
                "enable_q_channels": False,
                "actor": {"d_model": 32, "num_heads": 4, "num_layers": 1, "action_self_attention_layers": 1},
            },
            "selector": {
                "topk_low_voltage_buses": 1,
                "topk_high_voltage_buses": 1,
                "topk_lines": 1,
                "topk_trafos": 1,
            },
        }
    )

    envelope = policy.build(
        simulator,
        vpp,
        0,
        bid,
        fr,
        80.0,
        grid_state=scenario.dso.compute_network_state(),
    )

    assert envelope["source_policy"] == "sensitivity_attention_v1"
    assert envelope["award_status"] == "envelope_guidance"
    assert envelope["p_min_mw"] <= envelope["preferred_p_min_mw"]
    assert envelope["preferred_p_min_mw"] <= envelope["preferred_target_p_mw"]
    assert envelope["preferred_target_p_mw"] <= envelope["preferred_p_max_mw"]
    assert envelope["preferred_p_max_mw"] <= envelope["p_max_mw"]
    assert envelope["active_sensitivity_edges_shape"][2] == 9
    assert envelope["action_units"]
    assert envelope["selected_network_objects"]
    assert envelope["sensitivity_allocation_mode"] == "equal_pp_element_refs"
    assert envelope["sensitivity_allocation_weights"]


def test_env_routes_dso_global_guidance_to_unified_envelope_actor() -> None:
    env = MultiAgentVPPDSOEnv(
        config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
        horizon_steps=1,
    )
    env.reset(seed=123)
    payload: dict[str, object] = {
        "dso_global_guidance": {
            "envelope_action": {
                "center_ratio": [0.20] * 32,
                "width_ratio": [0.30] * 32,
                "guidance_strength": [0.90] * 32,
                "direction_logits": [[5.0, 0.0, -5.0]] * 32,
                "source": "test_unified_actor",
            }
        }
    }
    for vpp in env.scenario.vpps:
        payload[f"{vpp.id}_dispatch"] = {"normalized_setpoint_bias": 0.0}
        payload[f"{vpp.id}_portfolio"] = {"action": "keep"}

    _, _, _, _, infos = env.step(payload)

    rows = [
        row
        for row in env.simulator.records["dso_operating_envelope"]
        if int(row["step"]) == 0
    ]
    assert len(rows) == len(env.scenario.vpps)
    assert all(row["dso_decision_interface"] == "sensitivity_attention_v1_unified_actor" for row in rows)
    assert all(row["actor_override_source"] == "test_unified_actor" for row in rows)
    assert all(row["rule_warmstart_role"] in {"fallback_disabled", "teacher_reference_only"} for row in rows)
    assert infos["dso_global_guidance"]["decoded_dso_targets"] == {}


def test_sensitivity_env_converts_legacy_dso_targets_to_unified_envelope_override() -> None:
    env = MultiAgentVPPDSOEnv(
        config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset(seed=124)
    targets = {
        vpp.id: 0.5
        * (
            float(observations["dso_global_guidance"]["vpp_reports"][vpp.id]["p_min_mw"])
            + float(observations["dso_global_guidance"]["vpp_reports"][vpp.id]["p_max_mw"])
        )
        for vpp in env.scenario.vpps
    }

    _, _, _, _, infos = env.step({"dso_global_guidance": {"targets": targets}})

    rows = [
        row
        for row in env.simulator.records["dso_operating_envelope"]
        if int(row["step"]) == 0
    ]
    assert len(rows) == len(env.scenario.vpps)
    assert all(row["dso_decision_interface"] == "sensitivity_attention_v1_unified_actor" for row in rows)
    assert all(row["actor_override_source"] == "legacy_targets_converted_to_unified_envelope" for row in rows)
    assert infos["dso_global_guidance"]["decoded_dso_targets"] == targets
    assert "__dso_envelope_guidance__" in infos["dso_global_guidance"]["decoded_simulator_action_payload"]


def test_sensitivity_attention_policy_reuses_raw_sensitivity_cache_for_active_slice() -> None:
    scenario = load_scenario("configs/european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    bid = vpp.day_ahead_bid(0, price_hint=80.0)
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    policy = build_dso_envelope_policy(
        {
            "dso": {
                "envelope_policy": "sensitivity_attention_v1",
                "action_unit_granularity": "vpp_bus",
                "enable_q_channels": False,
                "actor": {"d_model": 32, "num_heads": 4, "num_layers": 1, "action_self_attention_layers": 1},
            },
            "selector": {
                "topk_low_voltage_buses": 1,
                "topk_high_voltage_buses": 1,
                "topk_lines": 1,
                "topk_trafos": 1,
            },
            "sensitivity": {
                "enabled": True,
                "cache_enabled": True,
                "cache_ttl_steps": 8,
            },
        }
    )

    first = policy.build(
        simulator,
        vpp,
        0,
        bid,
        fr,
        80.0,
        grid_state=scenario.dso.compute_network_state(),
    )
    second = policy.build(
        simulator,
        vpp,
        1,
        bid,
        fr,
        80.0,
        grid_state=scenario.dso.compute_network_state(),
    )

    assert first["sensitivity_cache_hit"] is False
    assert second["sensitivity_cache_hit"] is True
    assert second["sensitivity_source"] == "raw_sensitivity_cache"
    assert second["active_sensitivity_edges_shape"] == first["active_sensitivity_edges_shape"]


def test_sensitivity_attention_policy_refreshes_cache_when_update_period_elapsed() -> None:
    scenario = load_scenario("configs/european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    simulator = Simulator(scenario)
    vpp = scenario.vpps[1]
    bid = vpp.day_ahead_bid(0, price_hint=80.0)
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    policy = build_dso_envelope_policy(
        {
            "dso": {
                "envelope_policy": "sensitivity_attention_v1",
                "action_unit_granularity": "vpp_bus",
                "enable_q_channels": False,
                "actor": {"d_model": 32, "num_heads": 4, "num_layers": 1, "action_self_attention_layers": 1},
            },
            "selector": {
                "topk_low_voltage_buses": 1,
                "topk_high_voltage_buses": 1,
                "topk_lines": 1,
                "topk_trafos": 1,
            },
            "sensitivity": {
                "enabled": True,
                "cache_enabled": True,
                "update_period_steps": 1,
                "cache_ttl_steps": 8,
                "max_perturbed_action_units_per_update": 1,
            },
        }
    )

    first = policy.build(
        simulator,
        vpp,
        0,
        bid,
        fr,
        80.0,
        grid_state=scenario.dso.compute_network_state(),
    )
    second = policy.build(
        simulator,
        vpp,
        1,
        bid,
        fr,
        80.0,
        grid_state=scenario.dso.compute_network_state(),
    )

    assert first["sensitivity_cache_hit"] is False
    assert second["sensitivity_cache_hit"] is False
    assert second["sensitivity_source"] == "finite_difference_recomputed"
    assert "update_period_elapsed" in second["sensitivity_refresh_reasons"]
    assert second["sensitivity_priority_action_units"]
    assert second["sensitivity_partial_priority_refresh"] is True
    assert len(second["sensitivity_partial_refresh_action_unit_ids"]) == 1


def test_sensitivity_attention_policy_loads_structured_happo_checkpoint_actor(tmp_path) -> None:
    import torch

    from vpp_dso_sim.dso.models.bipartite_attention_actor import BipartiteSensitivityDSOActor

    scenario = load_scenario("configs/european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    bid = vpp.day_ahead_bid(0, price_hint=80.0)
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    actor = BipartiteSensitivityDSOActor(
        global_feature_dim=6,
        action_token_dim=13,
        object_token_dim=10,
        edge_feature_dim=9,
        d_model=32,
        num_heads=4,
        num_layers=1,
        action_self_attention_layers=1,
        dropout=0.0,
        min_width_ratio=0.10,
        max_width_ratio=1.00,
    )
    for parameter in actor.parameters():
        parameter.data.zero_()
    actor.center_head.bias.data.fill_(10.0)
    actor.width_head.bias.data.fill_(-10.0)
    actor.lambda_head.bias.data.fill_(10.0)
    actor.direction_head.bias.data[:] = torch.tensor([10.0, 0.0, -10.0])
    checkpoint_path = tmp_path / "structured_happo_checkpoint.pt"
    torch.save(
        {
            "dso_actor_type": "sensitivity_attention_v1_structured_happo",
            "actor_state_dict": {
                f"dso_actor.attention_actor.{key}": value.clone()
                for key, value in actor.state_dict().items()
            },
        },
        checkpoint_path,
    )
    policy = build_dso_envelope_policy(
        {
            "dso": {
                "envelope_policy": "sensitivity_attention_v1",
                "action_unit_granularity": "vpp_bus",
                "enable_q_channels": False,
                "actor": {
                    "d_model": 32,
                    "num_heads": 4,
                    "num_layers": 1,
                    "action_self_attention_layers": 1,
                    "checkpoint_path": str(checkpoint_path),
                },
            },
            "selector": {
                "topk_low_voltage_buses": 1,
                "topk_high_voltage_buses": 1,
                "topk_lines": 1,
                "topk_trafos": 1,
            },
        }
    )

    envelope = policy.build(
        simulator,
        vpp,
        0,
        bid,
        fr,
        80.0,
        grid_state=scenario.dso.compute_network_state(),
    )

    assert envelope["dso_actor_checkpoint_loaded"] is True
    assert envelope["dso_actor_checkpoint_path"] == str(checkpoint_path)
    assert min(envelope["dso_actor_raw_outputs"]["center_ratio"]) > 0.99
    assert min(envelope["dso_actor_raw_outputs"]["guidance_strength"]) > 0.99
    first_direction = envelope["direction_probs"][0]
    assert first_direction[0] > 0.99


def test_sensitivity_attention_policy_blends_rule_warmstart_with_actor_by_residual_eta(tmp_path) -> None:
    import torch

    from vpp_dso_sim.dso.models.bipartite_attention_actor import BipartiteSensitivityDSOActor

    scenario = load_scenario("configs/european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)
    simulator = Simulator(scenario)
    vpp = scenario.vpps[0]
    bid = vpp.day_ahead_bid(0, price_hint=80.0)
    fr = compute_static_feasible_region(vpp, 0, scope="bus_vector")
    grid_state = scenario.dso.compute_network_state()
    rule_envelope = simulator._build_dso_operating_envelope(vpp, 0, bid, fr, 80.0, grid_state=grid_state)
    actor = BipartiteSensitivityDSOActor(
        global_feature_dim=6,
        action_token_dim=13,
        object_token_dim=10,
        edge_feature_dim=9,
        d_model=32,
        num_heads=4,
        num_layers=1,
        action_self_attention_layers=1,
        dropout=0.0,
        min_width_ratio=0.10,
        max_width_ratio=1.00,
    )
    for parameter in actor.parameters():
        parameter.data.zero_()
    actor.center_head.bias.data.fill_(10.0)
    actor.width_head.bias.data.fill_(-10.0)
    actor.lambda_head.bias.data.fill_(10.0)
    checkpoint_path = tmp_path / "actor.pt"
    torch.save({"state_dict": actor.state_dict()}, checkpoint_path)

    def build_with_progress(progress: int) -> dict:
        policy = build_dso_envelope_policy(
            {
                "dso": {
                    "envelope_policy": "sensitivity_attention_v1",
                    "action_unit_granularity": "vpp_bus",
                    "enable_q_channels": False,
                    "enable_rule_warmstart": True,
                    "warmstart_steps": 0,
                    "residual_schedule_steps": 10,
                    "residual_progress_step": progress,
                    "actor": {
                        "d_model": 32,
                        "num_heads": 4,
                        "num_layers": 1,
                        "action_self_attention_layers": 1,
                        "checkpoint_path": str(checkpoint_path),
                    },
                },
                "selector": {
                    "topk_low_voltage_buses": 1,
                    "topk_high_voltage_buses": 1,
                    "topk_lines": 1,
                    "topk_trafos": 1,
                },
            }
        )
        return policy.build(simulator, vpp, 0, bid, fr, 80.0, grid_state=grid_state)

    warmstart_envelope = build_with_progress(0)
    learned_envelope = build_with_progress(10)

    assert warmstart_envelope["residual_rule_blend_enabled"] is True
    assert warmstart_envelope["residual_schedule_eta"] == 0.0
    assert learned_envelope["residual_schedule_eta"] == 1.0
    assert warmstart_envelope["preferred_target_p_mw"] == pytest.approx(rule_envelope["preferred_target_p_mw"])
    assert learned_envelope["preferred_target_p_mw"] != pytest.approx(rule_envelope["preferred_target_p_mw"])


def test_simulator_routes_to_sensitivity_attention_policy_from_config() -> None:
    scenario = load_scenario("configs/european_lv_benchmark_v2.yaml")
    scenario.config["dso"] = {
        "envelope_policy": "sensitivity_attention_v1",
        "action_unit_granularity": "vpp_bus",
        "enable_q_channels": False,
        "actor": {"d_model": 32, "num_heads": 4, "num_layers": 1, "action_self_attention_layers": 1},
    }
    scenario.config["selector"] = {
        "topk_low_voltage_buses": 1,
        "topk_high_voltage_buses": 1,
        "topk_lines": 1,
        "topk_trafos": 1,
    }
    simulator = Simulator(scenario)

    simulator.step(0)

    envelopes = simulator.records["dso_operating_envelope"]
    assert envelopes
    assert {row["source_policy"] for row in envelopes} == {"sensitivity_attention_v1"}
    assert all(row["award_status"] == "envelope_guidance" for row in envelopes)
