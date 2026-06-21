from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_harness_module():
    path = Path("scripts/agent_harness.py")
    spec = importlib.util.spec_from_file_location("agent_harness_for_test", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_agent_harness_runs_phase_tests_and_records_handoff(monkeypatch, tmp_path) -> None:
    module = _load_harness_module()
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append([str(part) for part in cmd])
        return subprocess.CompletedProcess(cmd, 0, stdout=" M src/example.py\n", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    handoff_path = tmp_path / "HANDOFF.md"

    result = module.run_phase(
        "phase_02_schemas",
        handoff_path=handoff_path,
        changed_files=("src/example.py",),
    )

    assert result == 0
    assert calls == [
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_action_units.py",
            "tests/test_network_objects.py",
            "tests/test_sensitivity_shapes.py",
        ]
    ]
    handoff = handoff_path.read_text(encoding="utf-8")
    assert "phase_02_schemas" in handoff
    assert "`src/example.py`" in handoff
    assert "Test result: passed" in handoff
