#!/usr/bin/env bash
set -euo pipefail
"${PYTHON:-./.venv-server/bin/python}" scripts/run_smoke.py \
  --config configs/baseline_rule_v0.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0
"${PYTHON:-./.venv-server/bin/python}" scripts/run_smoke.py \
  --config configs/happo_legacy_mlp.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0
"${PYTHON:-./.venv-server/bin/python}" scripts/run_smoke.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0
"${PYTHON:-./.venv-server/bin/python}" scripts/run_short_train.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 256 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0
