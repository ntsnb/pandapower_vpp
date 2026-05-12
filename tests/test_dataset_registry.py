from __future__ import annotations

from vpp_dso_sim.data_sources import dataset_registry_frame, default_dataset_registry


def test_dataset_registry_contains_real_training_sources():
    records = default_dataset_registry()
    frame = dataset_registry_frame(records)

    assert len(records) >= 8
    assert {"nrel_smart_ds", "simbench", "acn_data", "openei_urdb"}.issubset(
        set(frame["dataset_id"])
    )
    assert {"dataset_id", "category", "url", "integration_priority", "caveats"}.issubset(
        frame.columns
    )
    assert set(frame[frame["integration_priority"] == "P0"]["dataset_id"]) >= {
        "nrel_smart_ds",
        "simbench",
        "ieee_pes_test_feeders",
        "nrel_eulp",
        "acn_data",
    }
