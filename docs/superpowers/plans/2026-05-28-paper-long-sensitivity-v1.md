# Paper-Long Sensitivity V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a paper-long experiment path that adapts the existing `paper_long` protocol to the current DSO `sensitivity_attention_v1` + structured HAPPO actor while preserving legacy paper-long baselines.

**Architecture:** Keep the existing paper-training runner and add an explicit sensitivity-v1 paper-long preset/config. The new path uses the large European LV benchmark scenario, structured DSO observation, finite-difference sensitivity features, HAPPO stability settings from YAML/preset, and frozen checkpoint evaluation; legacy HATRPO/MATD3/HASAC remain available only as legacy-observation comparisons unless they are later upgraded.

**Tech Stack:** Python 3.12, PyTorch, pandapower, pandas, YAML configs, pytest, existing `examples/17_paper_training_experiment.py` runner.

---

### Task 1: Add Regression Tests For Sensitivity Paper-Long Preset

**Files:**
- Modify: `tests/test_paper_training_experiment.py`

- [ ] **Step 1: Write the failing test**

Add a test that imports `paper_training_preset("paper_long_sensitivity_v1")` and asserts:

```python
def test_paper_long_sensitivity_v1_preset_uses_structured_happo_config():
    cfg = paper_training_preset("paper_long_sensitivity_v1")

    assert cfg.preset == "paper_long_sensitivity_v1"
    assert str(cfg.config_path).endswith("configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml")
    assert cfg.algorithms == ("rule_based", "no_flex", "ac_validated_search_reference", "happo")
    assert cfg.seeds == (9401, 9402, 9403, 9404, 9405)
    assert cfg.eval_variants == ("holdout_peak", "holdout_cloudy", "holdout_reverseflow")
    assert cfg.horizon_steps == 672
    assert cfg.eval_horizon_steps == 672
    assert cfg.train_episodes == 120
    assert cfg.checkpoint_selection == "both"
    assert cfg.happo_critic_use_action_summary is True
    assert cfg.happo_use_yaml_trainer_settings is True
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
./.venv-server/bin/python -m pytest -q tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_preset_uses_structured_happo_config
```

Expected before implementation: fails with `Unknown paper training preset`.

### Task 2: Add Large Scenario Sensitivity Config

**Files:**
- Create: `configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml`

- [ ] **Step 1: Create config by extending the large benchmark scenario**

The config must keep the large `european_lv_benchmark_v2` network and VPP list, and add:

```yaml
dso:
  envelope_policy: sensitivity_attention_v1
  observation_mode: structured_bipartite
  action_unit_granularity: vpp_bus
  enable_q_channels: false
  enable_rule_warmstart: true
  warmstart_steps: 20000
  residual_schedule_steps: 50000
  max_action_units: 32
  max_network_objects: 20
  actor:
    d_model: 128
    num_heads: 4
    num_layers: 2
    action_self_attention_layers: 1
    dropout: 0.02
    min_width_ratio: 0.10
    max_width_ratio: 1.00
```

Also add `selector`, `sensitivity`, and `trainer` sections compatible with `configs/happo_sensitivity_attention_v1.yaml`, adjusted for paper-long:

```yaml
trainer:
  name: happo
  gamma: 0.995
  gae_lambda: 0.95
  clip_param: 0.2
  target_kl: 0.02
  entropy_coef: 0.01
  max_grad_norm: 0.5
  normalize_observations: true
  normalize_advantages: true
  nan_guard: true
  critic_use_action_summary: true
```

- [ ] **Step 2: Verify scenario loads and dimensions fit**

Run:

```bash
./.venv-server/bin/python -c "from vpp_dso_sim.simulation.scenario import load_scenario; s=load_scenario('configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml'); print(len(s.vpps), s.config['dso']['envelope_policy'], s.config['dso']['max_action_units'])"
```

Expected: prints `7 sensitivity_attention_v1 32`.

### Task 3: Add Paper Training Preset And HAPPO YAML-Stability Switch

**Files:**
- Modify: `src/vpp_dso_sim/experiments/paper_training.py`

- [ ] **Step 1: Add dataclass field**

Add:

```python
happo_use_yaml_trainer_settings: bool = False
```

to `PaperTrainingExperimentConfig`.

- [ ] **Step 2: Add preset**

Add `paper_long_sensitivity_v1` returning:

```python
PaperTrainingExperimentConfig(
    config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
    output_dir="outputs/paper_training_long_sensitivity_v1",
    preset="paper_long_sensitivity_v1",
    algorithms=("rule_based", "no_flex", "ac_validated_search_reference", "happo"),
    seeds=(9401, 9402, 9403, 9404, 9405),
    train_variants=("train_mixed",),
    eval_variants=("holdout_peak", "holdout_cloudy", "holdout_reverseflow"),
    hparam_cases=("base", "lower_lr", "higher_entropy", "larger_network"),
    horizon_steps=672,
    eval_horizon_steps=672,
    train_episodes=120,
    hidden_dim=256,
    gamma=0.995,
    batch_size=256,
    replay_capacity=300_000,
    warmup_steps=2_000,
    ppo_epochs=4,
    checkpoint_selection="both",
    ac_reference_max_candidates=16,
    happo_critic_use_action_summary=True,
    happo_use_yaml_trainer_settings=True,
)
```

- [ ] **Step 3: Pass HAPPO stability fields**

In `_train_algorithm`, if `cfg.happo_use_yaml_trainer_settings` is true, build the HAPPO config from YAML defaults plus paper-training overrides so `target_kl`, observation normalization, advantage normalization, max grad norm, GAE lambda, and nan guard are not silently dropped.

- [ ] **Step 4: Verify test passes**

Run:

```bash
./.venv-server/bin/python -m pytest -q tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_preset_uses_structured_happo_config
```

Expected: pass.

### Task 4: Add Smoke Test For Structured HAPPO Through Paper Training Runner

**Files:**
- Modify: `tests/test_paper_training_experiment.py`

- [ ] **Step 1: Add small smoke test**

Create a temporary output run using `paper_training_preset("smoke")` but with:

```python
config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml"
algorithms=("happo",)
seeds=(9704,)
horizon_steps=2
eval_horizon_steps=2
train_episodes=1
hparam_cases=("base",)
checkpoint_selection="final"
tensorboard=False
export_html=False
happo_critic_use_action_summary=True
happo_use_yaml_trainer_settings=True
```

Assert the training summary has:

```python
dso_actor_type == "sensitivity_attention_v1_structured_happo"
dso_actor_observation_mode == "structured_bipartite"
normalize_observations is True
nan_guard is True
```

- [ ] **Step 2: Run smoke test**

Run:

```bash
./.venv-server/bin/python -m pytest -q tests/test_paper_training_experiment.py::test_paper_training_structured_happo_sensitivity_smoke
```

Expected: pass and write finite 2-step artifacts.

### Task 5: Update Documentation And Run Commands

**Files:**
- Modify: `README.md`
- Create: `docs/experiments/paper_long_sensitivity_v1_protocol.md`

- [ ] **Step 1: Document new command**

Recommended pilot:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_pilot_YYYYMMDD \
  --seeds 9401 \
  --horizon-steps 24 \
  --eval-horizon-steps 24 \
  --train-episodes 2 \
  --hparam-cases base \
  --algorithms rule_based,no_flex,ac_validated_search_reference,happo \
  --progress-interval-seconds 60
```

Recommended full run:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_YYYYMMDD \
  --progress-interval-seconds 60
```

- [ ] **Step 2: Document scope**

State that `happo` is the current main algorithm for `sensitivity_attention_v1`; HATRPO/MATD3/HASAC are legacy-observation comparisons unless separately upgraded.

### Task 6: Verification

**Files:**
- No production files.

- [ ] **Step 1: Run focused tests**

```bash
./.venv-server/bin/python -m pytest -q \
  tests/test_paper_training_experiment.py::test_paper_long_sensitivity_v1_preset_uses_structured_happo_config \
  tests/test_paper_training_experiment.py::test_paper_training_structured_happo_sensitivity_smoke \
  tests/test_structured_happo_training.py
```

- [ ] **Step 2: Run smoke command**

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --output-dir outputs/paper_training_long_sensitivity_v1_preflight_smoke \
  --seeds 9401 \
  --horizon-steps 2 \
  --eval-horizon-steps 2 \
  --train-episodes 1 \
  --hparam-cases base \
  --algorithms rule_based,no_flex,happo \
  --checkpoint-selection final \
  --no-html \
  --no-tensorboard
```

Expected:

- `experiment_manifest.json` exists.
- `run_index.csv` includes baseline and structured HAPPO rows.
- HAPPO train summary records `sensitivity_attention_v1_structured_happo`.
- `evaluation_seed_metrics.csv` is non-empty.
