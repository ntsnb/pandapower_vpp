from __future__ import annotations

from pathlib import Path

import pandas as pd

from scripts.reward_v2_result_audit import compare_evaluation_metrics, reward_abs_share_table, run_audit


def test_reward_abs_share_table_reports_dominant_terms() -> None:
    frame = pd.DataFrame(
        [
            {
                "algorithm": "happo",
                "dso_loss_cost": -2.0,
                "service_payment": 1.0,
                "contract_delivery_penalty": 1.0,
            },
            {
                "algorithm": "happo",
                "dso_loss_cost": -2.0,
                "service_payment": 1.0,
                "contract_delivery_penalty": 1.0,
            },
        ]
    )

    table = reward_abs_share_table(frame)
    by_term = {row["term"]: row for row in table.to_dict("records")}

    assert by_term["dso_loss_cost"]["abs_share"] == 0.5
    assert by_term["service_payment"]["abs_share"] == 0.25
    assert by_term["contract_delivery_penalty"]["abs_share"] == 0.25


def test_reward_abs_share_table_uses_effective_comfort_term_not_raw_diagnostic() -> None:
    frame = pd.DataFrame(
        [
            {
                "algorithm": "happo",
                "comfort_penalty": 10_000.0,
                "soc_penalty": 0.0,
                "scaled_comfort_soc_penalty": 5.0,
                "comfort_soc_weight": 0.02,
                "service_payment": 0.9,
            }
        ]
    )

    table = reward_abs_share_table(frame)
    by_term = {row["term"]: row for row in table.to_dict("records")}

    assert "comfort_penalty" not in by_term
    assert by_term["dispatch_comfort_soc_penalty"]["mean"] == 0.1
    assert by_term["dispatch_comfort_soc_penalty"]["abs_share"] < 0.2


def test_compare_evaluation_metrics_reports_new_minus_legacy(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy"
    new = tmp_path / "new"
    legacy.mkdir()
    new.mkdir()
    pd.DataFrame(
        [{"algorithm": "happo", "eval_total_reward": 1.0, "eval_total_cost": 10.0, "total_violation_cells": 2}]
    ).to_csv(legacy / "evaluation_seed_metrics.csv", index=False)
    pd.DataFrame(
        [{"algorithm": "happo", "eval_total_reward": 3.0, "eval_total_cost": 7.0, "total_violation_cells": 1}]
    ).to_csv(new / "evaluation_seed_metrics.csv", index=False)

    comparison = compare_evaluation_metrics(legacy_dir=legacy, new_dir=new)
    by_metric = {row["metric"]: row for row in comparison.to_dict("records")}

    assert by_metric["eval_total_reward"]["delta_new_minus_legacy"] == 2.0
    assert by_metric["eval_total_cost"]["delta_new_minus_legacy"] == -3.0
    assert by_metric["total_violation_cells"]["delta_new_minus_legacy"] == -1.0


def test_run_audit_writes_outputs(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    train_dir = run_dir / "runs" / "happo_base" / "train"
    train_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"algorithm": "happo", "dso_loss_cost": -2.0, "service_payment": 1.0}]
    ).to_csv(train_dir / "happo_step_metrics.csv", index=False)

    summary = run_audit(new_dir=run_dir, output_dir=tmp_path / "audit")

    assert summary["step_metric_rows"] == 1
    assert Path(summary["reward_component_abs_share_path"]).exists()
