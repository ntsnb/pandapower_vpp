from __future__ import annotations

from pathlib import Path

import pandas as pd

from marl_dashboard.backend.storage.query_service import QueryService
from marl_dashboard.integrations.paper_training import export_paper_training_dashboard


def test_export_paper_training_dashboard_writes_training_metrics(tmp_path: Path) -> None:
    result = {
        "output_dir": tmp_path / "paper_output",
        "manifest": {"schema_version": "paper_training_v1", "config": {"preset": "smoke"}},
        "run_index": pd.DataFrame(
            [
                {
                    "run_id": "happo_base_train_mixed_seed_1",
                    "algorithm": "happo_sequential_ctde",
                    "seed": 1,
                    "split": "train_profile",
                    "profile_variant": "train_mixed",
                    "hparam_case": "base",
                    "status": "completed",
                }
            ]
        ),
        "episode_metrics": pd.DataFrame(
            [
                {
                    "run_id": "happo_base_train_mixed_seed_1",
                    "algorithm": "happo_sequential_ctde",
                    "hparam_case": "base",
                    "episode": 0,
                    "episode_reward": -12.5,
                    "dso_episode_reward": -8.0,
                    "episode_cost": 3.4,
                    "violation_count": 1,
                    "projection_gap_mw": 0.05,
                },
                {
                    "run_id": "happo_base_train_mixed_seed_1",
                    "algorithm": "happo_sequential_ctde",
                    "hparam_case": "base",
                    "episode": 1,
                    "episode_reward": -10.0,
                    "dso_episode_reward": -6.0,
                    "episode_cost": 2.2,
                    "violation_count": 0,
                    "projection_gap_mw": 0.01,
                },
            ]
        ),
        "loss_metrics": pd.DataFrame(
            [
                {
                    "run_id": "happo_base_train_mixed_seed_1",
                    "algorithm": "happo_sequential_ctde",
                    "hparam_case": "base",
                    "global_step": 0,
                    "actor_loss": 0.2,
                    "critic_loss": 1.5,
                    "actor_grad_norm": 0.4,
                }
            ]
        ),
        "evaluation_seed_metrics": pd.DataFrame(
            [
                {
                    "run_id": "happo_base_train_mixed_seed_1_eval_holdout_peak",
                    "algorithm": "happo_sequential_ctde",
                    "seed": 1,
                    "profile_variant": "holdout_peak",
                    "eval_total_reward": -9.5,
                    "eval_total_cost": 2.9,
                    "total_violation_cells": 0,
                }
            ]
        ),
    }

    dashboard_run_id = export_paper_training_dashboard(result, data_dir=tmp_path / "dashboard_runs")

    service = QueryService(tmp_path / "dashboard_runs")
    assert dashboard_run_id == "paper_training_smoke"
    assert service.runs()[0]["run_id"] == "paper_training_smoke"
    selectors = service.selectors(dashboard_run_id)
    assert selectors["episode_ids"] == [0, 1]
    assert "happo_sequential_ctde/base" in selectors["policy_ids"]
    rewards = service.query_metric_table(run_id=dashboard_run_id, table="reward_terms")
    costs = service.query_metric_table(run_id=dashboard_run_id, table="cost_terms")
    losses = service.query_metric_table(run_id=dashboard_run_id, table="loss_terms")
    assert rewards["summary"]["row_count"] >= 3
    assert costs["summary"]["row_count"] >= 2
    assert losses["summary"]["row_count"] >= 3
