from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from marl_dashboard.backend.app import create_app
from marl_dashboard.backend.storage.metadata_store import MetadataStore


def _write_profile_config(output_dir: Path) -> Path:
    profile_dir = output_dir / "profiles" / "train_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    config_path = profile_dir / "scenario_config.yaml"
    config_path.write_text("extends: configs/scenarios/demo/ieee33_multi_vpp.yaml\n", encoding="utf-8")
    return config_path


def test_topology_endpoint_builds_pandapower_static_frames_from_profile_config(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_output"
    source_config = _write_profile_config(output_dir)
    MetadataStore(tmp_path / "dashboard_runs").initialize_run(
        run_id="topology_run",
        config={"algorithm": "happo", "environment": "paper_training"},
        metadata={"output_dir": str(output_dir)},
    )
    client = TestClient(create_app(data_dir=tmp_path / "dashboard_runs"))

    response = client.get("/api/runs/topology_run/topology")

    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == "topology_run"
    assert payload["source_config_path"] == str(source_config.resolve())
    assert payload["network"]["bus_count"] == 33
    assert payload["network"]["line_count"] == 32
    assert payload["network"]["vpp_count"] == 3
    assert {"bus_id", "x", "y", "is_pcc", "vpp_ids"}.issubset(payload["nodes"][0])
    assert {"edge_id", "from_bus", "to_bus", "edge_type"}.issubset(payload["edges"][0])
    assert any(node["is_pcc"] for node in payload["nodes"])
    assert any(asset["vpp_id"] == "vpp_1" and asset["der_type"] == "EVCSModel" for asset in payload["assets"])
    assert payload["pandapower_tables"]["bus_count"] == 33
    assert payload["pandapower_tables"]["line_count"] == 32
    assert payload["pandapower_tables"]["load_count"] >= 1
    assert payload["pandapower_tables"]["sgen_count"] >= 1
    assert payload["pandapower_tables"]["storage_count"] >= 1
    assert payload["sign_conventions"]["storage"] == "pandapower storage p_mw > 0 means charging; dashboard internal dispatch p_mw > 0 means export."
    assert any(
        mapping["vpp_id"] == "vpp_0"
        and mapping["pcc_bus"] == 5
        and 6 in mapping["asset_buses"]
        and "ess_0" in mapping["asset_ids"]
        for mapping in payload["vpp_bus_map"]
    )
    assert all("cost_coefficients" not in asset for asset in payload["assets"])
    assert all("metadata_json" not in asset for asset in payload["assets"])


def test_vpp_config_endpoint_returns_bilingual_vpp_asset_details(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_output"
    _write_profile_config(output_dir)
    MetadataStore(tmp_path / "dashboard_runs").initialize_run(
        run_id="vpp_config_run",
        config={"algorithm": "happo", "environment": "paper_training"},
        metadata={"output_dir": str(output_dir)},
    )
    client = TestClient(create_app(data_dir=tmp_path / "dashboard_runs"))

    response = client.get("/api/runs/vpp_config_run/vpp-config")

    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["vpp_count"] == 3
    residential = next(item for item in payload["vpps"] if item["vpp_id"] == "vpp_0")
    assert residential["display_name"] == "Residential VPP"
    assert residential["pcc_bus"] == 5
    assert residential["physical_mode"] == "multi_node"
    assert residential["privacy_mode_description"] == "完整信息 / Full information"
    assert residential["asset_counts"]["PVModel"] == 1
    assert residential["asset_counts"]["StorageModel"] == 1
    assert "PCC 母线" in residential["description"]
    assert "DER assets" in residential["description"]
    assert any("接入母线" in note and "connection buses" in note for note in residential["configuration_notes"])
    assert any("pandapower 写入" in note and "write target" in note for note in residential["configuration_notes"])
    assert residential["dispatch_capability"]["active_power_range_mw"] == [residential["p_min_mw"], residential["p_max_mw"]]
    assert residential["dispatch_capability"]["reactive_power_range_mvar"] == [residential["q_min_mvar"], residential["q_max_mvar"]]
    storage_asset = next(asset for asset in residential["assets"] if asset["der_id"] == "ess_0")
    assert storage_asset["configuration_summary"].startswith("储能 / Storage")
    assert "pandapower storage" in storage_asset["write_target"]
    assert any(asset["der_id"] == "ess_0" and asset["capacity_mwh"] == 1.0 for asset in residential["assets"])
