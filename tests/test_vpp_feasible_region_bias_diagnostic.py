from __future__ import annotations

import pandas as pd

from scripts.analyze_vpp_feasible_region_bias import run_bias_diagnostic


def test_vpp_feasible_region_bias_diagnostic_writes_variant_summaries(tmp_path) -> None:
    result = run_bias_diagnostic(
        config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
        output_dir=tmp_path,
        horizon_steps=2,
        variants=("baseline", "load_scale_1p2", "no_ac_aware"),
    )

    detail_path = result["detail_csv"]
    summary_path = result["summary_csv"]
    assert detail_path.exists()
    assert summary_path.exists()

    detail = pd.read_csv(detail_path)
    summary = pd.read_csv(summary_path)
    required_detail_columns = {
        "step",
        "vpp_id",
        "variant",
        "p_min_mw",
        "p_max_mw",
        "midpoint_mw",
        "span_mw",
        "all_negative",
        "crosses_zero",
        "midpoint_negative",
        "preferred_target_p_mw",
        "preferred_target_negative",
        "injection_headroom_mw",
        "absorption_headroom_mw",
        "network_min_vm_pu",
        "network_max_vm_pu",
        "max_line_loading_percent",
        "ac_aware_enabled",
        "ac_aware_reason",
    }
    assert required_detail_columns.issubset(detail.columns)
    assert set(detail["variant"]) == {"baseline", "load_scale_1p2", "no_ac_aware"}
    assert {"negative_midpoint_rate", "preferred_target_negative_rate"}.issubset(summary.columns)
