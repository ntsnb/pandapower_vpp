from __future__ import annotations

from pathlib import Path

from vpp_dso_sim.data_sources.smart_ds import (
    analyze_smart_ds_dataset,
    export_smart_ds_analysis,
    smart_ds_dataset_suites,
)


def _downloaded_root() -> Path:
    return (
        Path("data")
        / "external"
        / "raw"
        / "smart_ds"
        / "v1.0"
        / "2018"
        / "AUS"
        / "P1U"
        / "base_timeseries"
        / "opendss"
    )


def test_smart_ds_dataset_suite_definitions_are_present_without_download():
    suites = smart_ds_dataset_suites("data/external/raw/example")

    assert len(suites) >= 3
    assert {
        "smart_ds_full_feeder_ctde",
        "smart_ds_lv_portfolio_suite",
        "hybrid_der_market_suite",
    }.issubset(set(suites["suite_id"]))


def test_downloaded_smart_ds_dataset_analysis_if_available():
    root = _downloaded_root()
    if not root.exists():
        return

    result = analyze_smart_ds_dataset(root)
    summary = result["summary"]

    assert summary["file_count"] >= 2000
    assert summary["primary_feeder_count"] >= 20
    assert summary["dss_file_count"] >= 100
    assert summary["total_size_mib"] > 100.0
    assert not result["feeders"].empty
    assert {"feeder_id", "line_count", "load_count", "transformer_count"}.issubset(
        result["feeders"].columns
    )
    assert not result["suites"].empty


def test_smart_ds_analysis_export_writes_outputs_if_available():
    root = _downloaded_root()
    if not root.exists():
        return

    paths = export_smart_ds_analysis(root, Path("outputs") / "test_smart_ds_analysis")

    assert paths["summary"].exists()
    assert paths["feeders"].exists()
    assert paths["suites"].exists()
    assert paths["report"].exists()
