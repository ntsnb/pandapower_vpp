#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-./.venv-server/bin/python}" -m pytest -q tests/test_structured_observation_shapes.py tests/test_bipartite_attention_actor.py tests/test_safe_decoder.py
