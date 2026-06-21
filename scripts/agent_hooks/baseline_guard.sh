#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-./.venv-server/bin/python}" -m pytest -q tests/test_legacy_baseline_unchanged.py tests/test_envelope_policy_switch.py
