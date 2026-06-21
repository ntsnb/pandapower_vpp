# DSO Sensitivity Attention Runbook

Updated: 2026-05-28 Asia/Shanghai

## 1. Rule baseline

Run the legacy rule envelope baseline smoke:

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/baseline_rule_v0.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0
```

Guard tests:

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_legacy_baseline_unchanged.py \
  tests/test_envelope_policy_switch.py
```

## 2. Legacy MLP / flat-observation baseline

The legacy flat observation remains available through `legacy_flat`; the legacy
MLP actor remains the baseline path for non-structured trainer modes.

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_structured_observation_shapes.py \
  tests/test_hasac_happo.py \
  tests/test_hatrpo_training.py \
  tests/test_happo_hasac_trainers.py
```

Use this config as the legacy comparison point:

```text
configs/happo_legacy_mlp.yaml
```

## 3. sensitivity_attention_v1 smoke rollout

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0
```

Key outputs:

```text
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/dso_operating_envelope.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/action_units.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/selected_network_objects.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/sensitivity_edges.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/dso_actor_outputs.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/decoded_operating_envelope.csv
```

## 4. Short training sanity

The hook-level short training sanity uses 256 BC warm-start updates.

```bash
./.venv-server/bin/python scripts/run_short_train.py \
  --config configs/happo_sensitivity_attention_v1.yaml \
  --seed 0 \
  --steps 256 \
  --output-dir outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0
```

Key outputs:

```text
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/dso_sensitivity_attention_short_train_loss_metrics.csv
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/dso_sensitivity_attention_actor.pt
outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0/short_train_summary.json
```

## 5. Ablation configs

Run each ablation with the same smoke command pattern:

```bash
./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/ablation_no_sensitivity_edges.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/ablation_no_sensitivity_edges_smoke_seed0

./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/ablation_no_action_self_attention.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/ablation_no_action_self_attention_smoke_seed0

./.venv-server/bin/python scripts/run_smoke.py \
  --config configs/ablation_no_width_penalty.yaml \
  --seed 0 \
  --steps 2 \
  --output-dir outputs/dso_sensitivity_attention/ablation_no_width_penalty_smoke_seed0
```

## 6. Full verification

```bash
./.venv-server/bin/python -m pytest -q
bash scripts/agent_hooks/smoke_training.sh
```

## 7. Experiment and curve log

Every smoke/training run must update:

```text
docs/experiments/dso_sensitivity_attention_experiment_log.md
docs/experiments/dso_sensitivity_attention_artifacts/
```

## 8. Completion warning

Smoke rollout and BC warm-start success prove interface health and optimizer sanity.
They must not be described as paper-long convergence.
