from __future__ import annotations

from dataclasses import fields

from vpp_dso_sim.dso.envelope.schemas import ActionUnitState, StructuredDSOObservation
from vpp_dso_sim.dso.observation.happo_structured import build_happo_structured_dso_observation
from vpp_dso_sim.simulation.scenario import load_scenario


def test_dso_structured_schema_does_not_define_private_cost_fields() -> None:
    forbidden = {
        "cost_coefficients",
        "comfort_penalty",
        "soc",
        "soc_min",
        "soc_max",
        "private_true_cost",
        "oracle_cost",
    }
    state_fields = {field.name for field in fields(ActionUnitState)}
    obs_fields = {field.name for field in fields(StructuredDSOObservation)}

    assert forbidden.isdisjoint(state_fields)
    assert forbidden.isdisjoint(obs_fields)


def test_happo_flattened_structured_dso_observation_keeps_privacy_metadata() -> None:
    scenario = load_scenario("configs/happo_sensitivity_attention_v1.yaml")
    flat, spec = build_happo_structured_dso_observation(
        scenario,
        step=0,
        config=scenario.config,
    )

    assert flat.shape == (spec.flat_dim,)
    assert spec.privacy_boundary == "dso_execution_actor_no_private_vpp_fields"
    forbidden = {
        "zone_id",
        "reliability",
        "private_true_cost",
        "oracle_cost",
        "comfort_preference",
        "private_soc_internal",
    }
    assert forbidden.isdisjoint(set(spec.field_names))
