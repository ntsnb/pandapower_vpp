from __future__ import annotations

from vpp_dso_sim.envs.observations import (
    build_actor_observation,
    build_critic_global_state,
    privacy_visibility_records,
)
from vpp_dso_sim.simulation.scenario import load_scenario


def test_actor_observation_excludes_other_vpps_and_private_cost_by_default():
    scenario = load_scenario()
    vpp = scenario.vpps[0]
    obs = build_actor_observation(vpp, t=0)

    assert obs["agent_id"] == vpp.id
    assert obs["observation_type"] == "actor_observation_i"
    assert "network_state" not in obs
    assert "vpp_reports" not in obs
    assert set(obs["portfolio"]["connection_buses"]) == {der.bus for der in vpp.der_list}
    assert {asset["owner_vpp_id"] for asset in obs["local_assets"]} == {vpp.id}
    assert all("cost_coefficients" not in asset for asset in obs["local_assets"])
    assert all("metadata" not in asset for asset in obs["local_assets"])


def test_critic_global_state_is_marked_as_centralized_state():
    scenario = load_scenario()
    state = build_critic_global_state(scenario, t=0)

    assert state["agent_id"] == "critic"
    assert state["observation_type"] == "critic_global_state"
    assert "network_state" in state
    assert set(state["vpp_reports"]) == {vpp.id for vpp in scenario.vpps}


def test_privacy_visibility_records_include_schema_fields():
    rows = privacy_visibility_records()
    fields = {(row["schema"], row["field"]) for row in rows}
    assert ("DERSpec", "cost_coefficients") in fields
    assert ("VPPPortfolio", "connection_buses") in fields
