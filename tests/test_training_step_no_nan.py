from __future__ import annotations

import importlib.util
import json

import pandas as pd
import pytest

from vpp_dso_sim.experiments.dso_sensitivity_attention import run_short_training_sanity


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch is not installed")
def test_short_training_sanity_writes_loss_metrics_without_nan(tmp_path) -> None:
    summary = run_short_training_sanity(
        config_path="configs/happo_sensitivity_attention_v1.yaml",
        seed=0,
        steps=4,
        output_dir=tmp_path,
    )

    loss_path = tmp_path / "dso_sensitivity_attention_short_train_loss_metrics.csv"
    checkpoint_path = tmp_path / "dso_sensitivity_attention_actor.pt"
    summary_path = tmp_path / "short_train_summary.json"
    decoded_path = tmp_path / "decoded_operating_envelope.csv"
    assert summary["nan_or_inf_detected"] is False
    assert summary["config_hash"]
    assert summary["seed"] == 0
    assert summary["steps_completed"] == 4
    assert loss_path.exists()
    assert checkpoint_path.exists()
    assert summary_path.exists()
    assert decoded_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        persisted_summary = json.load(handle)
    assert persisted_summary["config_hash"] == summary["config_hash"]
    losses = pd.read_csv(loss_path)
    assert not losses.empty
    assert losses.select_dtypes(include=["number"]).notna().all().all()
    assert (losses["bc_loss"] >= 0.0).all()
