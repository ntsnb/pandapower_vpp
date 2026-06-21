#!/usr/bin/env bash
set -euo pipefail
git status --short
"${PYTHON:-./.venv-server/bin/python}" --version
