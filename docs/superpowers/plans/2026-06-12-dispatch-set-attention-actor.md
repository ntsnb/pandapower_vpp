# Dispatch Set-Attention Actor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Strengthen the VPP dispatch actor neural network for paper-long MARL experiments without changing the physical decoder, reward semantics, or AC safety shield.

**Architecture:** Add an optional `set_attention_v1` dispatch actor encoder beside the existing DeepSets encoder. The new actor keeps the same public output contract, but replaces mean/max-only DER pooling with mask-aware DER self-attention and token-level DER action heads.

**Tech Stack:** Python, PyTorch, pytest, HAPPO/HASAC/MATD3/HATRPO training utilities, pandapower simulation configs.

---

### Task 1: Add Actor Architecture Tests

**Files:**
- Create: `tests/test_dispatch_actor_set_attention.py`
- Modify: none

- [ ] **Step 1: Write failing tests**

Add tests that call `build_privacy_separated_ctde_modules(..., dispatch_actor_encoder_type="set_attention_v1")`, verify metadata, output shapes, finite outputs, and token-action masking behavior.

- [ ] **Step 2: Verify RED**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_dispatch_actor_set_attention.py -q
```

Expected before implementation: fail because `dispatch_actor_encoder_type` is not accepted.

### Task 2: Implement Set-Attention Dispatch Actor

**Files:**
- Modify: `src/vpp_dso_sim/learning/ctde_networks.py`
- Modify: `src/vpp_dso_sim/learning/deep_rl.py`

- [ ] **Step 1: Add `dispatch_actor_encoder_type` to the network builder**

The default must remain `deepset_v1` so existing tests and old checkpoints keep their expected metadata.

- [ ] **Step 2: Add `SetAttentionDispatchEncoder`**

The encoder must:
- split context and DER tokens using the existing dimensions,
- encode context and tokens with LayerNorm + MLP,
- apply mask-aware `nn.MultiheadAttention` over DER tokens,
- fuse context, attention pooled tokens, max pooled tokens, and token-count ratio,
- expose token latents for DER-level action heads.

- [ ] **Step 3: Preserve the actor output contract**

`VPPDispatchActor.forward(x)` must still return:

```python
aggregate_mean, aggregate_log_std, der_mean, der_log_std
```

where `aggregate_mean` is shape `[batch, 1]` and `der_mean` is shape `[batch, max_der_per_vpp]`.

- [ ] **Step 4: Verify GREEN**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_dispatch_actor_set_attention.py tests/test_deep_rl_training.py::test_privacy_separated_ctde_trainer_uses_separate_modules_and_writes_artifacts -q
```

Expected: new set-attention tests pass and legacy CTDE test still sees `deep_sets_shared_token_mlp`.

### Task 3: Wire Config Through Main Algorithms

**Files:**
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Modify: `src/vpp_dso_sim/learning/matd3.py`
- Modify: `src/vpp_dso_sim/learning/hatrpo.py`
- Modify: `src/vpp_dso_sim/experiments/paper_training.py`

- [ ] **Step 1: Add config field**

Add `dispatch_actor_encoder_type: str = "deepset_v1"` to HAPPO, HASAC, MATD3, and HATRPO configs.

- [ ] **Step 2: Pass config to builders**

Every `_build_privacy_separated_networks(...)` call for dispatch-capable algorithms must pass `dispatch_actor_encoder_type`.

- [ ] **Step 3: Align HATRPO dispatch actor**

HATRPO must use a dispatch policy adapter that wraps the shared VPP dispatch actor interface so HATRPO can train the same architecture family.

- [ ] **Step 4: Verify algorithm compatibility**

Run:

```bash
./.venv-server/bin/python -m pytest tests/test_hasac_happo.py::test_happo_training_writes_sequential_update_artifacts tests/test_matd3_training.py::test_matd3_training_smoke_writes_artifacts tests/test_hatrpo_training.py::test_hatrpo_training_smoke_writes_artifacts -q
```

Expected: all smoke tests pass.

### Task 4: Enable Paper-Long Set-Attention Settings

**Files:**
- Modify: `configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1.yaml`
- Modify: `configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml`
- Modify: `src/vpp_dso_sim/experiments/paper_training.py`

- [ ] **Step 1: Set HAPPO paper-long trainer to `set_attention_v1`**

Add the trainer setting in the canonical sensitivity config so paper-long sensitivity runs inherit it.

- [ ] **Step 2: Surface architecture metadata**

The training summaries and checkpoints must report `vpp_encoder_type=set_attention_v1_masked_self_attention`.

- [ ] **Step 3: Keep old baseline available**

Default configs and legacy smoke tests must still be able to run `deepset_v1`.

### Task 5: Run Verification and Start Paper-Long

**Files:**
- No source modifications unless verification reveals a defect.

- [ ] **Step 1: Unit and smoke tests**

Run targeted tests for actor architecture, HAPPO, HATRPO, MATD3, reward trace, and paper training config.

- [ ] **Step 2: GPU smoke training**

Run a 1-episode, 2-step HAPPO smoke with `dispatch_actor_encoder_type="set_attention_v1"` and `device="auto"`; verify the resolved device and architecture metadata.

- [ ] **Step 3: Start paper-long experiment**

Start a paper-long sensitivity run in a new output directory using GPU-capable configuration and progress logs.

Preferred command:

```bash
./.venv-server/bin/python examples/17_paper_training_experiment.py \
  --preset paper_long_sensitivity_v1 \
  --config-path configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v3_1_market_safety.yaml \
  --output-dir outputs/paper_training_long_dispatch_set_attention_v1_20260612 \
  --algorithms happo \
  --checkpoint-selection both \
  --progress-interval-seconds 60
```

Expected: run starts, writes progress artifacts, and summary metadata records the set-attention dispatch actor.
