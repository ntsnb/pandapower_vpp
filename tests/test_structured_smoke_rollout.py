from __future__ import annotations

import pandas as pd
import json

from vpp_dso_sim.experiments.dso_sensitivity_attention import run_smoke_rollout


def test_sensitivity_attention_smoke_rollout_writes_envelope_artifacts(tmp_path) -> None:
    summary = run_smoke_rollout(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        seed=0,
        steps=2,
        output_dir=tmp_path,
    )

    envelope_path = tmp_path / "dso_operating_envelope.csv"
    metrics_path = tmp_path / "smoke_step_metrics.csv"
    summary_path = tmp_path / "smoke_summary.json"
    assert summary["envelope_policy"] == "sensitivity_attention_v1"
    assert summary["seed"] == 0
    assert summary["config_hash"]
    assert envelope_path.exists()
    assert metrics_path.exists()
    assert summary_path.exists()
    envelopes = pd.read_csv(envelope_path)
    assert "source_policy" in envelopes.columns
    assert set(envelopes["source_policy"]) == {"sensitivity_attention_v1"}
    assert "active_sensitivity_edges_shape" in envelopes.columns
    with summary_path.open("r", encoding="utf-8") as handle:
        persisted_summary = json.load(handle)
    assert persisted_summary["config_hash"] == summary["config_hash"]
    for artifact in (
        "selected_network_objects.csv",
        "action_units.csv",
        "dso_actor_outputs.csv",
        "decoded_operating_envelope.csv",
        "sensitivity_edges.csv",
    ):
        assert (tmp_path / artifact).exists()
