# Unified DSO Sensitivity Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Merge the learned DSO guidance path and `sensitivity_attention_v1` operating-envelope path so `sensitivity_attention_v1` is the single DSO decision interface, with rule warm-start only as teacher/fallback, and add diagnostics for negative VPP feasible-region bias.

**Architecture:** HAPPO/HATRPO structured DSO actors must output envelope parameters for DSO action units instead of legacy per-VPP active-power targets. The environment forwards those envelope parameters into the DSO envelope policy; the simulator builds the final AC-aware DOE/envelope and VPP dispatch agents respond within that envelope.

**Tech Stack:** Python, PyTorch, Gymnasium-style multi-agent environment, pandapower, pytest, pandas.

---

### Task 1: Add Tests For Unified DSO Envelope Action Payload

**Files:**
- Modify: `tests/test_envelope_policy_switch.py`
- Modify: `tests/test_structured_happo_training.py`
- Target source: `src/vpp_dso_sim/envs/multi_agent_env.py`
- Target source: `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`
- Target source: `src/vpp_dso_sim/dso/models/structured_happo_actor.py`

- [ ] **Step 1: Write failing env/policy test**

Add a test that sends `dso_global_guidance: {"envelope_action": ...}` and asserts the simulator records `dso_decision_interface == "sensitivity_attention_v1_unified_actor"` and no legacy decoded DSO target is used.

```python
def test_env_routes_dso_global_guidance_to_unified_envelope_actor() -> None:
    env = MultiAgentVPPDSOEnv(
        config_path="configs/european_lv_benchmark_v2_sensitivity_attention_v1.yaml",
        horizon_steps=1,
    )
    observations, _ = env.reset(seed=123)
    vpp_count = len(env.scenario.vpps)
    payload = {
        "dso_global_guidance": {
            "envelope_action": {
                "center_ratio": [0.20] * 32,
                "width_ratio": [0.30] * 32,
                "guidance_strength": [0.90] * 32,
                "direction_logits": [[5.0, 0.0, -5.0]] * 32,
                "source": "test_unified_actor",
            }
        }
    }
    for vpp in env.scenario.vpps:
        payload[f"{vpp.id}_dispatch"] = {"normalized_setpoint_bias": 0.0}
        payload[f"{vpp.id}_portfolio"] = {"action": "keep"}

    _, _, _, _, infos = env.step(payload)

    rows = [
        row for row in env.simulator.records["dso_operating_envelope"]
        if int(row["step"]) == 0
    ]
    assert len(rows) == vpp_count
    assert all(row["dso_decision_interface"] == "sensitivity_attention_v1_unified_actor" for row in rows)
    assert all(row["rule_warmstart_role"] in {"fallback_disabled", "teacher_reference_only"} for row in rows)
    assert infos["dso_global_guidance"]["decoded_dso_targets"] == {}
```

- [ ] **Step 2: Run test to verify RED**

Run: `./.venv-server/bin/python -m pytest tests/test_envelope_policy_switch.py::test_env_routes_dso_global_guidance_to_unified_envelope_actor -q`

Expected: FAIL because `envelope_action` is currently not accepted by validation/policy routing and `decoded_dso_targets` is populated only from legacy target actions.

- [ ] **Step 3: Write failing actor shape test**

Add a test that structured DSO actor returns envelope-action dimensions, not one scalar per VPP.

```python
def test_structured_dso_actor_outputs_action_unit_envelope_parameters() -> None:
    # Build a tiny StructuredDSOFlatSpec with 3 action units and 2 VPPs.
    # Forward one batch and assert action_dim == max_action_units * 6.
    # The six channels are center, width, strength, and three direction logits.
```

- [ ] **Step 4: Run actor test to verify RED**

Run: `./.venv-server/bin/python -m pytest tests/test_structured_happo_training.py::test_structured_dso_actor_outputs_action_unit_envelope_parameters -q`

Expected: FAIL because `StructuredDSOGaussianActor` currently returns one mean per VPP.

### Task 2: Route DSO Actor Output Through `sensitivity_attention_v1`

**Files:**
- Modify: `src/vpp_dso_sim/learning/ctde_interface.py`
- Modify: `src/vpp_dso_sim/envs/multi_agent_env.py`
- Modify: `src/vpp_dso_sim/simulation/simulator.py`
- Modify: `src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py`

- [ ] **Step 1: Accept envelope action payload**

Update `_validate_dso_action()` so dict payloads with `envelope_action` are preserved and do not become legacy `targets`.

- [ ] **Step 2: Forward envelope action into simulator**

Add an internal simulator action key such as `__dso_envelope_guidance__` from `MultiAgentVPPDSOEnv.step()`. Keep legacy target handling available only for non-envelope payloads and tests.

- [ ] **Step 3: Pass per-VPP actor override into policy**

Add `actor_override` keyword to `Simulator._build_dso_operating_envelope_for_policy()` and `SensitivityAttentionEnvelopePolicy.build()`.

- [ ] **Step 4: Decode actor override as envelope parameters**

Inside `SensitivityAttentionEnvelopePolicy.build()`, if `actor_override` exists, use its `center_ratio`, `width_ratio`, `direction_logits` or `direction_probs`, and `guidance_strength` for the current action units. Then apply safe decoding and AC-aware DOE projection exactly once.

- [ ] **Step 5: Log rule warm-start as teacher/fallback only**

Record `rule_warmstart_role`, `rule_teacher_target_p_mw`, `actor_target_p_mw`, `final_target_p_mw`, `actor_override_source`, `dso_decision_interface`, and `fallback_reason`.

- [ ] **Step 6: Run Task 1 env test to verify GREEN**

Run: `./.venv-server/bin/python -m pytest tests/test_envelope_policy_switch.py::test_env_routes_dso_global_guidance_to_unified_envelope_actor -q`

Expected: PASS.

### Task 3: Expand Structured DSO Actor From VPP Target Head To Envelope Head

**Files:**
- Modify: `src/vpp_dso_sim/dso/models/structured_happo_actor.py`
- Modify: `src/vpp_dso_sim/learning/advanced_marl.py`
- Modify: `tests/test_structured_happo_training.py`

- [ ] **Step 1: Replace VPP scalar head with fixed action-unit envelope head**

Return Gaussian mean/log_std with shape `max_action_units * 6`. Channel order: center, width, strength, direction_absorb_logit, direction_balanced_logit, direction_inject_logit.

- [ ] **Step 2: Add helper to convert normalized actor action to envelope payload**

Add a helper that maps sampled normalized action to arrays:

```text
center_ratio = 0.5 * (clip(action_center, -1, 1) + 1)
width_ratio = min_width + (max_width - min_width) * 0.5 * (clip(action_width, -1, 1) + 1)
guidance_strength = 0.5 * (clip(action_strength, -1, 1) + 1)
direction_logits = clipped raw direction channels
```

- [ ] **Step 3: HAPPO rollout emits envelope payload**

In `train_happo()`, replace `{"targets": dso_targets}` with `{"envelope_action": envelope_payload}` when using structured DSO mode.

- [ ] **Step 4: Preserve legacy fallback for flat DSO actor**

For legacy MLP modes, keep existing `_targets_from_normalized_actions()` behavior.

- [ ] **Step 5: Run actor test and smoke HAPPO test**

Run:

```bash
./.venv-server/bin/python -m pytest \
  tests/test_structured_happo_training.py::test_structured_dso_actor_outputs_action_unit_envelope_parameters \
  tests/test_structured_happo_training.py::test_happo_uses_structured_dso_actor_when_config_requests_it \
  -q
```

Expected: PASS.

### Task 4: Increase `sensitivity_attention_v1` Capacity For Paper-Long Runs

**Files:**
- Modify: `configs/experiments/paper_long/sensitivity_attention_v1/european_lv_benchmark_v2_sensitivity_attention_v1.yaml`
- Modify: `configs/algorithms/dso_sensitivity_attention/v1/happo_sensitivity_attention_v1.yaml`
- Modify: `tests/test_bipartite_attention_actor.py`

- [ ] **Step 1: Increase paper-long defaults**

Set paper-long actor defaults to `d_model: 256`, `num_heads: 8`, `num_layers: 3`, `action_self_attention_layers: 2`, `dropout: 0.05`.

- [ ] **Step 2: Increase short algorithm defaults conservatively**

Set short HAPPO sensitivity config to `d_model: 128`, `num_heads: 4`, `num_layers: 2`, `action_self_attention_layers: 1`, `dropout: 0.02`.

- [ ] **Step 3: Verify actor still handles masks**

Run: `./.venv-server/bin/python -m pytest tests/test_bipartite_attention_actor.py -q`

Expected: PASS.

### Task 5: Add VPP Feasible-Region Negative-Bias Diagnostic Experiment

**Files:**
- Create: `scripts/analyze_vpp_feasible_region_bias.py`
- Create/Modify: `tests/test_vpp_feasible_region_bias_diagnostic.py`

- [ ] **Step 1: Write failing CLI test**

Test a 4-step diagnostic run writes `vpp_feasible_region_bias_summary.csv` and includes current, load-up, and no-AC variants.

- [ ] **Step 2: Implement diagnostic script**

The script must load a scenario, iterate VPPs and steps, compute FR bounds and DSO envelope, and write:

```text
step, vpp_id, variant, p_min_mw, p_max_mw, midpoint_mw, span_mw,
all_negative, crosses_zero, midpoint_negative, preferred_target_p_mw,
preferred_target_negative, injection_headroom_mw, absorption_headroom_mw,
network_min_vm_pu, network_max_vm_pu, max_line_loading_percent,
ac_aware_enabled, ac_aware_reason
```

- [ ] **Step 3: Add load-scaling perturbation variants**

Run variants: `baseline`, `load_scale_1p2`, `load_scale_1p5`, `pv_scale_0p8`, `pv_scale_1p2`, `no_ac_aware`.

- [ ] **Step 4: Run diagnostic test**

Run: `./.venv-server/bin/python -m pytest tests/test_vpp_feasible_region_bias_diagnostic.py -q`

Expected: PASS.

### Task 6: Validation And Experiment Launch

**Files:**
- Read/write outputs only under `outputs/unified_dso_sensitivity_attention_*`

- [ ] **Step 1: Run targeted unit tests**

Run:

```bash
./.venv-server/bin/python -m pytest \
  tests/test_envelope_policy_switch.py \
  tests/test_structured_happo_training.py \
  tests/test_bipartite_attention_actor.py \
  tests/test_vpp_feasible_region_bias_diagnostic.py \
  -q
```

- [ ] **Step 2: Run 2-8 step smoke training**

Run a short HAPPO/HATRPO smoke with `device=auto` and output under `outputs/unified_dso_sensitivity_attention_smoke_YYYYMMDD`.

- [ ] **Step 3: Run feasible-region diagnostic experiment**

Run:

```bash
./.venv-server/bin/python scripts/analyze_vpp_feasible_region_bias.py \
  --config configs/european_lv_benchmark_v2_sensitivity_attention_v1_reward_v2_minimal.yaml \
  --horizon-steps 24 \
  --output-dir outputs/unified_dso_sensitivity_attention_bias_audit_YYYYMMDD
```

- [ ] **Step 4: Report evidence**

Report changed files, tests run, smoke output, bias-audit output, whether negative feasible-region tendency is caused by hard DER bounds, AC-aware DOE, missing regional load, or their combination.
