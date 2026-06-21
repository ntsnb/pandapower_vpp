from __future__ import annotations

from dataclasses import fields
from pathlib import Path

from vpp_dso_sim.dso.envelope.schemas import NetworkObjectId, NetworkObjectState
from vpp_dso_sim.dso.sensitivity.selectors import select_critical_network_objects
from vpp_dso_sim.network.powerflow import run_powerflow
from vpp_dso_sim.simulation.scenario import load_scenario


def test_network_object_schema_has_no_zone_or_reliability_fields() -> None:
    names = {field.name for field in fields(NetworkObjectId)} | {field.name for field in fields(NetworkObjectState)}

    assert "zone_id" not in names
    assert "reliability" not in names


def test_select_critical_network_objects_returns_typed_tokens() -> None:
    scenario = load_scenario(Path("configs") / "european_lv_benchmark_v2.yaml")
    assert run_powerflow(scenario.net)

    objects = select_critical_network_objects(
        scenario.net,
        voltage_limits=scenario.dso.voltage_limits,
        line_loading_limit_percent=scenario.dso.line_loading_limit_percent,
        trafo_loading_limit_percent=scenario.dso.trafo_loading_limit_percent,
        topk_low_voltage_buses=2,
        topk_high_voltage_buses=2,
        topk_lines=2,
        topk_trafos=1,
    )

    assert objects
    assert {"bus", "line"}.issubset({obj.id.object_type for obj in objects})
    assert all(obj.id.endpoint_bus_ids for obj in objects)
    assert all(obj.value_kind in {"vm_pu", "line_loading_percent", "trafo_loading_percent"} for obj in objects)
