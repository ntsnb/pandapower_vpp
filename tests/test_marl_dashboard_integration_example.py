from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

from marl_dashboard.backend.storage.query_service import QueryService


def test_integrate_logger_example_generates_minimal_dashboard_run(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    example = project_root / "examples" / "integrate_logger_example.py"
    env = {**os.environ, "PYTHONPATH": str(project_root / "src")}

    result = subprocess.run(
        [
            sys.executable,
            str(example),
            "--data-dir",
            str(tmp_path),
            "--run-id",
            "integration_example_run",
            "--dry-run",
        ],
        cwd=project_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "integration_example_run" in result.stdout
    assert "dry_run=true" in result.stdout

    service = QueryService(tmp_path)
    run = service.runs()[0]
    assert run["run_id"] == "integration_example_run"
    assert run["status"] == "finished"
    assert service.query_metric_table(run_id="integration_example_run", table="dataset_timeseries")["summary"]["row_count"] >= 8
    assert service.query_metric_table(run_id="integration_example_run", table="reward_terms")["summary"]["row_count"] >= 5
    assert service.query_metric_table(run_id="integration_example_run", table="cost_terms")["summary"]["row_count"] >= 4
    assert service.query_metric_table(run_id="integration_example_run", table="loss_terms")["summary"]["row_count"] >= 3
    assert service.query_metric_table(run_id="integration_example_run", table="events")["summary"]["row_count"] >= 2
