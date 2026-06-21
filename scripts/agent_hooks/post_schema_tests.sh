#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-./.venv-server/bin/python}" -m pytest -q tests/test_action_units.py tests/test_network_objects.py tests/test_sensitivity_shapes.py
