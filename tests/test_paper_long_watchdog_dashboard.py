from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from marl_dashboard.backend.storage.query_service import QueryService
from marl_dashboard.logging import ExperimentLogger


ROOT = Path(__file__).resolve().parents[1]


def test_watchdog_once_can_mirror_progress_to_dashboard(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00",
                "phase": "baseline_step",
                "message": "baseline step progress",
                "run_id": "rule_based_holdout_peak_seed_1",
                "algorithm": "rule_based",
                "seed": 1,
                "profile_variant": "holdout_peak",
                "step": 12,
                "step_progress_pct": 0.25,
                "horizon_steps": 48,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"
    audit_log = tmp_path / "audit.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(audit_log),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    assert service.runs()[0]["run_id"] == "paper_long_live"
    scalars = service.query_metric_table(run_id="paper_long_live", table="scalar_metrics")
    events = service.query_metric_table(run_id="paper_long_live", table="events")
    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries")
    assert scalars["summary"]["row_count"] >= 2
    assert events["summary"]["row_count"] >= 1
    assert dataset["summary"]["row_count"] >= 2


def test_watchdog_heartbeats_update_dashboard_without_hourly_audit_spam(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00",
                "phase": "baseline_step",
                "message": "baseline step progress",
                "run_id": "rule_based_holdout_peak_seed_1",
                "algorithm": "rule_based",
                "seed": 1,
                "profile_variant": None,
                "step": 18,
                "step_progress_pct": 0.375,
                "horizon_steps": 48,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"
    audit_log = tmp_path / "audit.jsonl"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(audit_log),
            "--interval-seconds",
            "3600",
            "--dashboard-heartbeat-seconds",
            "0.1",
            "--max-heartbeats",
            "2",
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert len(audit_log.read_text(encoding="utf-8").splitlines()) == 1
    service = QueryService(dashboard_dir)
    scalars = service.query_metric_table(run_id="paper_long_live", table="scalar_metrics")
    events = service.query_metric_table(run_id="paper_long_live", table="events")
    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries")
    assert scalars["summary"]["row_count"] >= 8
    assert events["summary"]["row_count"] >= 2
    assert dataset["summary"]["row_count"] >= 4
    assert all(row["env_id"] != "nan" for row in dataset["table_rows"])


def test_watchdog_compacts_dashboard_part_files_without_archiving_live_parts(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00",
                "phase": "baseline_step",
                "message": "baseline step progress",
                "run_id": "rule_based_holdout_peak_seed_1",
                "algorithm": "rule_based",
                "seed": 1,
                "step": 1,
                "step_progress_pct": 0.25,
                "horizon_steps": 4,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"
    logger = ExperimentLogger(run_id="paper_long_live", data_dir=str(dashboard_dir), config={}, async_writer=False)
    for time_index in range(4):
        logger.log_dataset(
            epoch_id=0,
            episode_id=1,
            env_id="env_0",
            vpp_id="vpp_001",
            date="2018-01-01",
            time_index=time_index,
            values={"electricity_price": 50.0 + time_index},
        )
    logger.close()

    partition_dir = (
        dashboard_dir
        / "paper_long_live"
        / "tables"
        / "dataset_timeseries"
        / "epoch_id=000000"
        / "vpp_id=vpp_001"
    )
    assert len(list(partition_dir.glob("part-*.parquet"))) == 4

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--dashboard-compact-every-heartbeats",
            "1",
            "--dashboard-compact-min-part-files",
            "2",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert list(partition_dir.glob("compact-*.parquet"))
    assert len(list(partition_dir.glob("part-*.parquet"))) == 4
    assert not (partition_dir / "_compacted_parts").exists()

    service = QueryService(dashboard_dir)
    table_files = service.duckdb_store.table_files(
        "paper_long_live",
        "dataset_timeseries",
        epoch_id=0,
        vpp_id="vpp_001",
    )
    assert len(table_files) == 1
    dataset = service.query_metric_table(
        run_id="paper_long_live",
        table="dataset_timeseries",
        metrics=["electricity_price"],
        vpp_id="vpp_001",
    )
    assert dataset["summary"]["row_count"] == 4


def test_watchdog_mirrors_completed_baseline_reward_and_cost(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:10:00",
                "phase": "baseline_done",
                "message": "baseline completed",
                "run_id": "rule_based_holdout_peak_seed_1",
                "algorithm": "rule_based",
                "seed": 1,
                "profile_variant": "holdout_peak",
                "reward_sum": 123.5,
                "total_cost": 456.25,
                "violations": 0,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    rewards = service.query_metric_table(run_id="paper_long_live", table="reward_terms")
    costs = service.query_metric_table(run_id="paper_long_live", table="cost_terms")
    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries")
    assert any(row["metric_name"] == "reward_sum" and row["value"] == 123.5 for row in rewards["table_rows"])
    assert any(row["metric_name"] == "total_cost" and row["value"] == 456.25 for row in costs["table_rows"])
    assert all(row["env_id"] != "nan" for row in dataset["table_rows"])


def test_watchdog_mirrors_completed_progress_events_that_are_no_longer_latest(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:10:00",
                "phase": "baseline_done",
                "message": "baseline completed",
                "run_id": "rule_based_holdout_peak_seed_1",
                "algorithm": "rule_based",
                "seed": 1,
                "profile_variant": "holdout_peak",
                "reward_sum": 123.5,
                "total_cost": 456.25,
                "violations": 0,
            },
            {
                "timestamp": "2026-01-01T00:10:05",
                "phase": "shared_rollout_step",
                "message": "training rollout progress",
                "run_id": "no_flex_holdout_peak_seed_1",
                "algorithm": "no_flex",
                "seed": 1,
                "episode": 1,
                "step": 24,
                "step_progress_pct": 0.5,
            },
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    rewards = service.query_metric_table(run_id="paper_long_live", table="reward_terms")
    costs = service.query_metric_table(run_id="paper_long_live", table="cost_terms")
    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries")
    assert any(row["metric_name"] == "reward_sum" and row["value"] == 123.5 for row in rewards["table_rows"])
    assert any(row["metric_name"] == "total_cost" and row["value"] == 456.25 for row in costs["table_rows"])
    assert all(row["env_id"] != "nan" for row in dataset["table_rows"])


def test_watchdog_mirrors_live_training_loss_progress_to_dashboard(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:20:00",
                "phase": "train_update",
                "message": "training update completed",
                "run_id": "happo_base_train_mixed_seed_1",
                "algorithm": "happo",
                "seed": 1,
                "hparam_case": "base",
                "train_variant": "train_mixed",
                "episode": 2,
                "step": 672,
                "global_step": 1344,
                "gradient_step": 2,
                "critic_loss": 0.25,
                "dso_policy_loss": -0.1,
                "dispatch_policy_loss": -0.05,
                "portfolio_policy_loss": -0.01,
                "critic_grad_norm": 1.5,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    losses = service.query_metric_table(run_id="paper_long_live", table="loss_terms")
    loss_values = {row["metric_name"]: row["value"] for row in losses["table_rows"]}
    assert loss_values["critic_loss"] == 0.25
    assert loss_values["dso_policy_loss"] == -0.1
    assert loss_values["dispatch_policy_loss"] == -0.05
    assert loss_values["portfolio_policy_loss"] == -0.01


def test_watchdog_heartbeat_does_not_duplicate_loss_terms_without_new_progress(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:20:00",
                "phase": "train_update",
                "message": "training update completed",
                "run_id": "happo_base_train_mixed_seed_1",
                "algorithm": "happo",
                "episode": 2,
                "step": 672,
                "gradient_step": 2,
                "critic_loss": 0.25,
                "dso_policy_loss": -0.1,
                "dispatch_policy_loss": -0.05,
                "portfolio_policy_loss": -0.01,
                "critic_grad_norm": 1.5,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--interval-seconds",
            "3600",
            "--dashboard-heartbeat-seconds",
            "0.1",
            "--max-heartbeats",
            "2",
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    losses = service.query_metric_table(run_id="paper_long_live", table="loss_terms")
    assert losses["summary"]["row_count"] == 5


def test_watchdog_can_restart_without_replaying_existing_dashboard_progress(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:20:00",
                "phase": "train_update",
                "message": "training update completed",
                "run_id": "happo_base_train_mixed_seed_1",
                "algorithm": "happo",
                "episode": 2,
                "step": 672,
                "gradient_step": 2,
                "critic_loss": 0.25,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--dashboard-skip-existing-progress",
            "--max-heartbeats",
            "1",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    losses = service.query_metric_table(run_id="paper_long_live", table="loss_terms")
    scalars = service.query_metric_table(run_id="paper_long_live", table="scalar_metrics")
    assert losses["summary"]["row_count"] == 0
    assert scalars["summary"]["row_count"] > 0


def test_watchdog_mirrors_profile_physical_dataset_per_vpp_before_dispatch_trace_exists(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    profile_dir = output_dir / "profiles" / "profile_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    load_path = profile_dir / "load_profile.csv"
    pv_path = profile_dir / "pv_profile.csv"
    price_path = profile_dir / "price_profile.csv"
    pd.DataFrame({"value": [1.0, 0.5]}).to_csv(load_path, index=False)
    pd.DataFrame({"value": [0.5, 0.25]}).to_csv(pv_path, index=False)
    pd.DataFrame({"value": [88.0, 99.0]}).to_csv(price_path, index=False)
    (profile_dir / "profile_metadata.json").write_text(
        json.dumps(
            {
                "source": "smart_ds_austin_profiles_local",
                "variant": "train_mixed",
                "seed": 1,
                "calendar_year": 2018,
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "scenario_config.yaml").write_text(
        f"""
simulation:
  horizon_steps: 2
  dt_hours: 0.25
profiles:
  load_profile_csv: {load_path}
  pv_profile_csv: {pv_path}
  price_profile_csv: {price_path}
vpps:
  - id: vpp_a
    assets:
      pv:
        - id: pv_a
          p_max_mw: 2.0
      storage:
        - id: storage_a
          capacity_mwh: 4.0
          soc: 0.5
      flexible_load:
        - id: flex_a
          baseline_p_mw: 3.0
      evcs:
        - id: evcs_a
          p_charge_max_mw: 1.0
  - id: vpp_b
    assets:
      pv:
        - id: pv_b
          p_max_mw: 1.0
      flexible_load:
        - id: flex_b
          baseline_p_mw: 2.0
""",
        encoding="utf-8",
    )
    output_dir.mkdir(exist_ok=True)
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:20:00",
                "phase": "shared_rollout_step",
                "message": "training rollout progress",
                "run_id": "happo_base_train_mixed_seed_1",
                "algorithm": "happo",
                "seed": 1,
                "train_variant": "train_mixed",
                "episode": 1,
                "worker_index": 0.0,
                "step": 1,
                "horizon_steps": 2,
                "step_progress_pct": 0.5,
                "reward_so_far": 1.0,
                "total_cost_so_far": 2.0,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    selectors = service.selectors("paper_long_live")
    assert selectors["vpp_ids"] == ["aggregate", "vpp_a", "vpp_b"]

    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries", vpp_id="vpp_a")
    values = {row["metric_name"]: row["value"] for row in dataset["table_rows"]}
    assert values["electricity_price"] == 99.0
    assert values["pv_power"] == 0.5
    assert values["base_load"] == 1.5
    assert values["ev_charging_load"] == 0.5
    assert values["storage_soc"] == 50.0
    assert dataset["table_rows"][0]["date"] == "2018-01-01"
    assert {row["time_index"] for row in dataset["table_rows"] if row["metric_name"] == "electricity_price"} == {0, 1}
    assert dataset["table_rows"][0]["env_id"] == "worker_0"


def test_watchdog_mirrors_dense_profile_dataset_for_progress_episode(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    profile_dir = output_dir / "profiles" / "profile_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    load_path = profile_dir / "load_profile.csv"
    pv_path = profile_dir / "pv_profile.csv"
    price_path = profile_dir / "price_profile.csv"
    pd.DataFrame({"value": [1.0, 0.8, 0.6, 0.4]}).to_csv(load_path, index=False)
    pd.DataFrame({"value": [0.1, 0.2, 0.3, 0.4]}).to_csv(pv_path, index=False)
    pd.DataFrame({"value": [70.0, 80.0, 90.0, 100.0]}).to_csv(price_path, index=False)
    (profile_dir / "profile_metadata.json").write_text(
        json.dumps({"source": "smart_ds_austin_profiles_local", "calendar_year": 2018}),
        encoding="utf-8",
    )
    (profile_dir / "scenario_config.yaml").write_text(
        f"""
simulation:
  horizon_steps: 4
  dt_hours: 0.25
profiles:
  load_profile_csv: {load_path}
  pv_profile_csv: {pv_path}
  price_profile_csv: {price_path}
vpps:
  - id: vpp_a
    assets:
      pv:
        - id: pv_a
          p_max_mw: 2.0
      flexible_load:
        - id: flex_a
          baseline_p_mw: 3.0
      evcs:
        - id: evcs_a
          p_charge_max_mw: 1.0
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:20:00",
                "phase": "shared_rollout_step",
                "message": "training rollout progress",
                "run_id": "happo_base_train_mixed_seed_1",
                "algorithm": "happo",
                "seed": 1,
                "train_variant": "train_mixed",
                "episode": 3,
                "worker_index": 0.0,
                "step": 2,
                "horizon_steps": 4,
                "step_progress_pct": 0.5,
                "reward_so_far": 1.0,
                "total_cost_so_far": 2.0,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    dataset = service.query_metric_table(
        run_id="paper_long_live",
        table="dataset_timeseries",
        metrics="electricity_price",
        vpp_id="vpp_a",
        episode_id=3,
        date="2018-01-01",
    )

    by_time = {row["time_index"]: row["value"] for row in dataset["table_rows"]}
    assert by_time == {0: 70.0, 1: 80.0, 2: 90.0, 3: 100.0}
    assert dataset["summary"]["row_count"] == 4


def test_watchdog_mirrors_progress_reward_cost_terms_per_vpp_before_dispatch_trace_exists(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    profile_dir = output_dir / "profiles" / "profile_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    load_path = profile_dir / "load_profile.csv"
    pv_path = profile_dir / "pv_profile.csv"
    price_path = profile_dir / "price_profile.csv"
    pd.DataFrame({"value": [1.0, 0.5]}).to_csv(load_path, index=False)
    pd.DataFrame({"value": [0.5, 0.25]}).to_csv(pv_path, index=False)
    pd.DataFrame({"value": [88.0, 99.0]}).to_csv(price_path, index=False)
    (profile_dir / "profile_metadata.json").write_text(
        json.dumps(
            {
                "source": "smart_ds_austin_profiles_local",
                "variant": "train_mixed",
                "seed": 1,
                "calendar_year": 2018,
            }
        ),
        encoding="utf-8",
    )
    (profile_dir / "scenario_config.yaml").write_text(
        f"""
simulation:
  horizon_steps: 2
  dt_hours: 0.25
profiles:
  load_profile_csv: {load_path}
  pv_profile_csv: {pv_path}
  price_profile_csv: {price_path}
vpps:
  - id: vpp_a
    assets:
      pv:
        - id: pv_a
          p_max_mw: 2.0
      flexible_load:
        - id: flex_a
          baseline_p_mw: 3.0
  - id: vpp_b
    assets:
      pv:
        - id: pv_b
          p_max_mw: 1.0
      flexible_load:
        - id: flex_b
          baseline_p_mw: 2.0
""",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:20:00",
                "phase": "shared_rollout_step",
                "message": "training rollout progress",
                "run_id": "happo_base_train_mixed_seed_1",
                "algorithm": "happo",
                "seed": 1,
                "train_variant": "train_mixed",
                "episode": 6,
                "worker_index": 0.0,
                "step": 1,
                "horizon_steps": 2,
                "step_progress_pct": 0.5,
                "reward_so_far": 11.0,
                "total_cost_so_far": 7.0,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    for vpp_id in ("vpp_a", "vpp_b"):
        rewards = service.query_metric_table(
            run_id="paper_long_live",
            table="reward_terms",
            vpp_id=vpp_id,
            episode_id=6,
        )
        costs = service.query_metric_table(
            run_id="paper_long_live",
            table="cost_terms",
            vpp_id=vpp_id,
            episode_id=6,
        )

        reward_values = {row["metric_name"]: row["value"] for row in rewards["table_rows"]}
        cost_values = {row["metric_name"]: row["value"] for row in costs["table_rows"]}
        assert reward_values["reward_so_far"] == 11.0
        assert reward_values["total_reward"] == 11.0
        assert cost_values["total_cost_so_far"] == 7.0
        assert cost_values["total_cost"] == 7.0
        assert all(row["agent_id"] == f"{vpp_id}_dispatch" for row in rewards["table_rows"])
        assert all(row["agent_id"] == f"{vpp_id}_dispatch" for row in costs["table_rows"])


def test_watchdog_mirrors_per_vpp_dispatch_trace_terms_to_dashboard(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    trace_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    trace_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "step": 3,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 88.0,
                "delivered_p_mw": 0.25,
                "dt_hours": 0.25,
                "baseline_p_mw": 0.1,
                "requested_delta_p_mw": 0.2,
                "accepted_delta_p_mw": 0.18,
                "actual_delta_p_mw": 0.16,
                "actual_target_p_mw": 0.26,
                "dispatch_reward_train": 1.5,
                "dispatch_reward_env": 1.25,
                "dispatch_private_profit_reward": 0.5,
                "private_profit_proxy": 25.0,
                "energy_market_revenue": 5.5,
                "pv_export_revenue_total": 1.2,
                "storage_discharge_revenue_total": 0.8,
                "evcs_user_revenue_total": 2.4,
                "service_payment": 7.0,
                "service_payment_weight": 0.0,
                "availability_payment": 3.0,
                "availability_payment_weight": 0.0,
                "storage_potential_raw": 12.5,
                "storage_potential_shaping_reward": 0.25,
                "storage_potential_shaping_weight": 0.02,
                "import_energy_cost_total": 1.1,
                "evcs_wholesale_cost_total": 0.6,
                "storage_charge_cost_total": 0.4,
                "battery_degradation_cost_total": 0.05,
                "der_operation_cost": 0.3,
                "reward_scaled_contract_delivery_penalty": 0.0,
                "reward_scaled_dispatch_projection_penalty": 0.7,
                "reward_scaled_training_projection_penalty": 0.7,
                "reward_scaled_total_projection_penalty": 1.4,
                "reward_scaled_comfort_soc_penalty": 0.06,
                "reward_scaled_battery_degradation_penalty": 0.04,
                "dispatch_private_profit_reward_formula": "private_profit_weight * private_profit_proxy",
                "energy_market_revenue_formula": "market_price * delivered_p_mw * dt_hours",
            },
            {
                "episode": 0,
                "step": 3,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_b_dispatch",
                "vpp_id": "vpp_b",
                "market_price": 88.0,
                "delivered_p_mw": -0.1,
                "dt_hours": 0.25,
                "baseline_p_mw": -0.2,
                "requested_delta_p_mw": 0.05,
                "accepted_delta_p_mw": 0.04,
                "actual_delta_p_mw": 0.03,
                "actual_target_p_mw": -0.17,
                "dispatch_reward_train": -0.2,
                "dispatch_reward_env": -0.15,
                "dispatch_private_profit_reward": -0.1,
                "private_profit_proxy": -5.0,
                "energy_market_revenue": -2.2,
                "pv_export_revenue_total": 0.0,
                "storage_discharge_revenue_total": 0.0,
                "evcs_user_revenue_total": 0.1,
                "service_payment": 0.0,
                "service_payment_weight": 0.0,
                "availability_payment": 0.0,
                "availability_payment_weight": 0.0,
                "storage_potential_raw": -2.5,
                "storage_potential_shaping_reward": -0.05,
                "storage_potential_shaping_weight": 0.02,
                "import_energy_cost_total": 2.0,
                "evcs_wholesale_cost_total": 1.2,
                "storage_charge_cost_total": 0.7,
                "battery_degradation_cost_total": 0.02,
                "der_operation_cost": 0.15,
                "reward_scaled_contract_delivery_penalty": 0.0,
                "reward_scaled_dispatch_projection_penalty": 0.1,
                "reward_scaled_training_projection_penalty": 0.1,
                "reward_scaled_total_projection_penalty": 0.2,
                "reward_scaled_comfort_soc_penalty": 0.02,
                "reward_scaled_battery_degradation_penalty": 0.01,
                "dispatch_private_profit_reward_formula": "private_profit_weight * private_profit_proxy",
                "energy_market_revenue_formula": "market_price * delivered_p_mw * dt_hours",
            },
        ]
    ).to_csv(trace_dir / "happo_dispatch_private_profit_trace_episode_0000.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    selectors = service.selectors("paper_long_live")
    assert selectors["vpp_ids"] == ["aggregate", "vpp_a", "vpp_b"]

    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries", vpp_id="vpp_a")
    dataset_values = {row["metric_name"]: row["value"] for row in dataset["table_rows"]}
    assert dataset_values["electricity_price"] == 88.0
    assert dataset_values["actual_delta_p_mw"] == 0.16
    assert dataset_values["actual_target_p_mw"] == 0.26

    rewards = service.query_metric_table(run_id="paper_long_live", table="reward_terms", vpp_id="vpp_a")
    reward_values = {row["metric_name"]: row["value"] for row in rewards["table_rows"]}
    assert reward_values["dispatch_reward_train"] == 1.5
    assert reward_values["energy_market_revenue"] == 5.5
    assert reward_values["evcs_user_revenue_total"] == 2.4
    assert reward_values["storage_discharge_revenue_total"] == 0.8
    assert reward_values["service_payment"] == 7.0
    assert reward_values["service_payment_weight"] == 0.0
    assert reward_values["storage_potential_raw"] == 12.5
    assert reward_values["storage_potential_shaping_reward"] == 0.25
    assert reward_values["storage_potential_shaping_weight"] == 0.02

    costs = service.query_metric_table(run_id="paper_long_live", table="cost_terms", vpp_id="vpp_b")
    cost_values = {row["metric_name"]: row["value"] for row in costs["table_rows"]}
    assert cost_values["import_energy_cost_total"] == 2.0
    assert cost_values["evcs_wholesale_cost_total"] == 1.2
    assert cost_values["storage_charge_cost_total"] == 0.7
    assert cost_values["reward_scaled_dispatch_projection_penalty"] == 0.1
    assert cost_values["reward_scaled_training_projection_penalty"] == 0.1
    assert cost_values["reward_scaled_total_projection_penalty"] == 0.2
    assert cost_values["reward_scaled_comfort_soc_penalty"] == 0.02
    assert cost_values["reward_scaled_battery_degradation_penalty"] == 0.01

    variable_names = {item["name"] for item in service.variables("paper_long_live")}
    assert {"reward_so_far", "electricity_price", "actual_delta_p_mw", "dispatch_reward_train"}.issubset(variable_names)


def test_watchdog_mirrors_only_new_dispatch_trace_rows_after_append(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    trace_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "happo_dispatch_private_profit_trace_episode_0000.csv"
    first_rows = [
        {
            "episode": 0,
            "step": 0,
            "algorithm": "happo_sequential_ctde",
            "agent_id": "vpp_a_dispatch",
            "vpp_id": "vpp_a",
            "market_price": 88.0,
            "dt_hours": 0.25,
            "dispatch_reward_train": 1.5,
        }
    ]
    pd.DataFrame(first_rows).to_csv(trace_path, index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    command = [
        sys.executable,
        str(ROOT / "scripts" / "watch_paper_long_run.py"),
        "--output-dir",
        str(output_dir),
        "--pid",
        str(os.getpid()),
        "--audit-log",
        str(tmp_path / "audit.jsonl"),
        "--dashboard-data-dir",
        str(dashboard_dir),
        "--dashboard-run-id",
        "paper_long_live",
        "--once",
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr

    pd.DataFrame(
        [
            *first_rows,
            {
                "episode": 0,
                "step": 1,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 89.0,
                "dt_hours": 0.25,
                "dispatch_reward_train": 2.5,
            },
        ]
    ).to_csv(trace_path, index=False)
    second = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr

    service = QueryService(dashboard_dir)
    rewards = service.query_metric_table(
        run_id="paper_long_live",
        table="reward_terms",
        vpp_id="vpp_a",
        metrics="dispatch_reward_train",
    )

    assert rewards["summary"]["row_count"] == 2
    assert [row["value"] for row in rewards["table_rows"]] == [1.5, 2.5]


def test_watchdog_migrates_legacy_dispatch_trace_state_before_append(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    trace_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "happo_dispatch_private_profit_trace_episode_0000.csv"
    first_rows = [
        {
            "episode": 0,
            "step": 0,
            "algorithm": "happo_sequential_ctde",
            "agent_id": "vpp_a_dispatch",
            "vpp_id": "vpp_a",
            "market_price": 88.0,
            "dt_hours": 0.25,
            "dispatch_reward_train": 1.5,
        }
    ]
    pd.DataFrame(first_rows).to_csv(trace_path, index=False)
    dashboard_dir = tmp_path / "dashboard_runs"
    run_id = "paper_long_live"
    trace_state = dashboard_dir / run_id / "mirrored_dispatch_trace_files.json"
    trace_state.parent.mkdir(parents=True)
    relative = trace_path.resolve().relative_to(output_dir.resolve())
    stat = trace_path.stat()
    legacy_key = f"{relative}:{stat.st_size}:{stat.st_mtime_ns}"
    physical_legacy_key = f"physical_dataset:{legacy_key}"
    trace_state.write_text(json.dumps([legacy_key, physical_legacy_key]), encoding="utf-8")

    command = [
        sys.executable,
        str(ROOT / "scripts" / "watch_paper_long_run.py"),
        "--output-dir",
        str(output_dir),
        "--pid",
        str(os.getpid()),
        "--audit-log",
        str(tmp_path / "audit.jsonl"),
        "--dashboard-data-dir",
        str(dashboard_dir),
        "--dashboard-run-id",
        run_id,
        "--once",
    ]
    first = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert first.returncode == 0, first.stderr

    pd.DataFrame(
        [
            *first_rows,
            {
                "episode": 0,
                "step": 1,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 89.0,
                "dt_hours": 0.25,
                "dispatch_reward_train": 2.5,
            },
        ]
    ).to_csv(trace_path, index=False)
    second = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert second.returncode == 0, second.stderr

    rewards = QueryService(dashboard_dir).query_metric_table(
        run_id=run_id,
        table="reward_terms",
        vpp_id="vpp_a",
        metrics="dispatch_reward_train",
    )

    assert rewards["summary"]["row_count"] == 1
    assert rewards["table_rows"][0]["value"] == 2.5


def test_watchdog_derives_per_vpp_physical_dataset_from_profiles_and_trace(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    profile_dir = output_dir / "profiles" / "profile_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    pd.DataFrame({"value": [0.5, 1.0]}).to_csv(profile_dir / "load_profile.csv", index=False)
    pd.DataFrame({"value": [0.2, 0.4]}).to_csv(profile_dir / "pv_profile.csv", index=False)
    pd.DataFrame({"value": [90.0, 100.0]}).to_csv(profile_dir / "price_profile.csv", index=False)
    (profile_dir / "scenario_config.yaml").write_text(
        """
simulation:
  dt_hours: 0.25
profiles:
  load_profile_csv: load_profile.csv
  pv_profile_csv: pv_profile.csv
  price_profile_csv: price_profile.csv
vpps:
- id: vpp_a
  assets:
    pv:
    - id: pv_a
      p_max_mw: 0.1
    storage:
    - id: ess_a
      capacity_mwh: 0.4
      soc: 0.5
    evcs:
    - id: evcs_a
      p_charge_max_mw: 0.2
    flexible_load:
    - id: flex_a
      baseline_p_mw: 0.3
    hvac_aggregator:
    - id: hvac_a
      rated_power_mw: 0.1
""",
        encoding="utf-8",
    )
    trace_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    trace_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "step": 1,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 100.0,
                "dt_hours": 0.25,
                "evcs_wholesale_cost_total": 5.0,
                "storage_discharge_revenue_total": 2.5,
                "storage_charge_cost_total": 1.25,
            }
        ]
    ).to_csv(trace_dir / "happo_dispatch_private_profit_trace_episode_0000.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    dataset = service.query_metric_table(run_id="paper_long_live", table="dataset_timeseries", vpp_id="vpp_a")
    values = {row["metric_name"]: row["value"] for row in dataset["table_rows"]}
    assert values["pv_power"] == pytest.approx(0.04)
    assert values["wind_power"] == 0.0
    assert values["base_load"] == pytest.approx(0.4)
    assert values["ev_charging_load"] == pytest.approx(0.2)
    assert values["storage_power"] == pytest.approx(0.05)
    assert values["storage_soc"] == pytest.approx(50.0)
    assert values["net_load"] == pytest.approx(0.51)


def test_watchdog_maps_dispatch_trace_steps_to_calendar_dates(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    profile_dir = output_dir / "profiles" / "profile_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    pd.DataFrame({"value": [1.0] * 120}).to_csv(profile_dir / "load_profile.csv", index=False)
    pd.DataFrame({"value": [0.5] * 120}).to_csv(profile_dir / "pv_profile.csv", index=False)
    pd.DataFrame({"value": [100.0] * 120}).to_csv(profile_dir / "price_profile.csv", index=False)
    (profile_dir / "profile_metadata.json").write_text(
        json.dumps({"source": "smart_ds_austin_profiles_local", "profiles_root": "/data/smart_ds/v1.0/2018/AUS/P1U/profiles"}),
        encoding="utf-8",
    )
    (profile_dir / "scenario_config.yaml").write_text(
        """
simulation:
  dt_hours: 0.25
profiles:
  load_profile_csv: load_profile.csv
  pv_profile_csv: pv_profile.csv
  price_profile_csv: price_profile.csv
vpps:
- id: vpp_a
  assets:
    pv:
    - id: pv_a
      p_max_mw: 0.2
""",
        encoding="utf-8",
    )
    trace_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    trace_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "step": 97,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 100.0,
                "dt_hours": 0.25,
                "worker_start_step": 672,
            }
        ]
    ).to_csv(trace_dir / "happo_dispatch_private_profit_trace_episode_0000.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    selectors = service.selectors("paper_long_live")
    assert "2018-01-02" in selectors["dates"]
    dataset = service.query_metric_table(
        run_id="paper_long_live",
        table="dataset_timeseries",
        vpp_id="vpp_a",
        date="2018-01-02",
        start_time_index=1,
        end_time_index=1,
    )
    assert dataset["summary"]["row_count"] > 0
    first_row = dataset["table_rows"][0]
    assert first_row["date"] == "2018-01-02"
    assert first_row["time_index"] == 1
    assert first_row["timestamp"].startswith("2018-01-02T00:15:00")
    assert first_row["global_env_step"] == 769


def test_watchdog_restart_adds_physical_dataset_without_replaying_core_trace_terms(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    profile_dir = output_dir / "profiles" / "profile_train_mixed_seed_1"
    profile_dir.mkdir(parents=True)
    pd.DataFrame({"value": [1.0]}).to_csv(profile_dir / "load_profile.csv", index=False)
    pd.DataFrame({"value": [0.5]}).to_csv(profile_dir / "pv_profile.csv", index=False)
    pd.DataFrame({"value": [100.0]}).to_csv(profile_dir / "price_profile.csv", index=False)
    (profile_dir / "scenario_config.yaml").write_text(
        """
profiles:
  load_profile_csv: load_profile.csv
  pv_profile_csv: pv_profile.csv
  price_profile_csv: price_profile.csv
vpps:
- id: vpp_a
  assets:
    pv:
    - id: pv_a
      p_max_mw: 0.2
""",
        encoding="utf-8",
    )
    trace_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    trace_dir.mkdir(parents=True)
    trace_path = trace_dir / "happo_dispatch_private_profit_trace_episode_0000.csv"
    pd.DataFrame(
        [
            {
                "episode": 0,
                "step": 0,
                "algorithm": "happo_sequential_ctde",
                "agent_id": "vpp_a_dispatch",
                "vpp_id": "vpp_a",
                "market_price": 100.0,
                "dt_hours": 0.25,
                "dispatch_reward_train": 1.0,
            }
        ]
    ).to_csv(trace_path, index=False)
    dashboard_dir = tmp_path / "dashboard_runs"
    run_id = "paper_long_live"
    trace_state = dashboard_dir / run_id / "mirrored_dispatch_trace_files.json"
    trace_state.parent.mkdir(parents=True)
    relative = trace_path.resolve().relative_to(output_dir.resolve())
    stat = trace_path.stat()
    core_key = f"{relative}:{stat.st_size}:{stat.st_mtime_ns}"
    trace_state.write_text(json.dumps([core_key]), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            run_id,
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    rewards = service.query_metric_table(run_id=run_id, table="reward_terms")
    dataset = service.query_metric_table(run_id=run_id, table="dataset_timeseries", vpp_id="vpp_a")
    values = {row["metric_name"]: row["value"] for row in dataset["table_rows"]}
    assert rewards["summary"]["row_count"] == 0
    assert values["pv_power"] == pytest.approx(0.1)


def test_watchdog_mirrors_update_metrics_to_loss_terms_by_vpp_and_policy(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    update_dir = output_dir / "runs" / "happo_base_train_mixed_seed_1" / "train"
    update_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "episode": 0,
                "epoch": 3,
                "role": "vpp_a_dispatch",
                "target_vpp_id": "vpp_a",
                "policy_loss": -0.25,
                "entropy_mean": 1.2,
                "approx_kl": 0.01,
                "grad_norm": 0.5,
            },
            {
                "episode": 0,
                "epoch": 3,
                "role": "dso_global_guidance",
                "policy_loss": -1.5,
                "entropy_mean": 2.0,
                "approx_kl": 0.02,
                "grad_norm": 1.1,
            },
        ]
    ).to_csv(update_dir / "happo_update_metrics.csv", index=False)
    dashboard_dir = tmp_path / "dashboard_runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--dashboard-data-dir",
            str(dashboard_dir),
            "--dashboard-run-id",
            "paper_long_live",
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    service = QueryService(dashboard_dir)
    selectors = service.selectors("paper_long_live")
    assert "vpp_a" in selectors["vpp_ids"]
    assert "vpp_a_dispatch" in selectors["policy_ids"]
    losses = service.query_metric_table(run_id="paper_long_live", table="loss_terms", vpp_id="vpp_a")
    values = {row["metric_name"]: row["value"] for row in losses["table_rows"]}
    assert values["policy_loss"] == pytest.approx(-0.25)
    assert values["entropy_mean"] == pytest.approx(1.2)
    assert values["approx_kl"] == pytest.approx(0.01)
    assert values["grad_norm"] == pytest.approx(0.5)


def test_watchdog_allows_sparse_optional_loss_columns(tmp_path: Path) -> None:
    output_dir = tmp_path / "paper_run"
    output_dir.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2026-01-01T00:00:00",
                "phase": "train_step",
                "message": "training update",
                "run_id": "happo_seed_1",
                "algorithm": "happo",
                "seed": 1,
                "step": 1,
            }
        ]
    ).to_csv(output_dir / "experiment_progress.csv", index=False)
    pd.DataFrame(
        [
            {"global_step": 1, "critic_loss": 0.25, "actor_loss": None},
            {"global_step": 2, "critic_loss": None, "actor_loss": 0.125},
        ]
    ).to_csv(output_dir / "training_loss_metrics.csv", index=False)

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "watch_paper_long_run.py"),
            "--output-dir",
            str(output_dir),
            "--pid",
            str(os.getpid()),
            "--audit-log",
            str(tmp_path / "audit.jsonl"),
            "--once",
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    record = json.loads(completed.stdout.splitlines()[-1])
    assert record["status"] == "ok"
    assert record["reasons"] == []
