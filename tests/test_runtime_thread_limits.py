from __future__ import annotations

import os
import subprocess
import sys


def test_configure_numeric_thread_limits_sets_server_safe_defaults(monkeypatch):
    from vpp_dso_sim.utils.runtime import configure_numeric_thread_limits

    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)

    configured = configure_numeric_thread_limits(default_threads=8)

    assert configured == {
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "NUMEXPR_NUM_THREADS": "8",
        "VECLIB_MAXIMUM_THREADS": "8",
    }


def test_configure_numeric_thread_limits_preserves_user_overrides(monkeypatch):
    from vpp_dso_sim.utils.runtime import configure_numeric_thread_limits

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "2")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)

    configured = configure_numeric_thread_limits(default_threads=8)

    assert configured["OPENBLAS_NUM_THREADS"] == "2"
    assert configured["OMP_NUM_THREADS"] == "8"


def test_paper_training_cli_configures_numeric_thread_limits_before_project_imports():
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env.pop(name, None)
    code = """
import os
import runpy
runpy.run_path('examples/17_paper_training_experiment.py')
print(os.environ.get('OMP_NUM_THREADS'))
print(os.environ.get('OPENBLAS_NUM_THREADS'))
print(os.environ.get('MKL_NUM_THREADS'))
print(os.environ.get('NUMEXPR_NUM_THREADS'))
print(os.environ.get('VECLIB_MAXIMUM_THREADS'))
"""

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=os.getcwd(),
        env=env,
        text=True,
        capture_output=True,
    )

    assert result.stdout.splitlines() == ["8", "8", "8", "8", "8"]
