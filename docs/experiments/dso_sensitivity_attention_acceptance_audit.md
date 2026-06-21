# DSO Sensitivity Attention Acceptance Audit

Updated: 2026-05-28 Asia/Shanghai

This audit maps the prompt acceptance criteria to current evidence. Smoke and
short training are sanity checks; they are not paper-long convergence evidence.

| # | Acceptance criterion | Status | Evidence |
|---:|---|---|---|
| 1 | `rule_v0` baseline runs | Verified | `bash scripts/agent_hooks/smoke_training.sh`; `outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0/smoke_summary.json`; `tests/test_legacy_baseline_unchanged.py` |
| 2 | Legacy flat DSO observation still works | Verified | `src/vpp_dso_sim/dso/observation/legacy_flat.py`; `tests/test_structured_observation_shapes.py::test_legacy_flat_dso_observation_still_matches_existing_encoder` |
| 3 | Legacy DSO MLP actor still works | Verified | legacy trainer tests in full `pytest -q`; `configs/happo_legacy_mlp.yaml`; `outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0/smoke_summary.json`; archived `current_happo_legacy_mlp_smoke_summary.json` |
| 4 | `sensitivity_attention_v1` deterministic smoke rollout runs | Verified | `scripts/run_smoke.py`; `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0/smoke_summary.json`; `tests/test_structured_smoke_rollout.py` |
| 5 | `sensitivity_attention_v1` short training update has no NaN/Inf | Verified | `bash scripts/agent_hooks/smoke_training.sh` runs 256 BC steps; `current_bc_short_train_summary.json` has `nan_or_inf_detected = false` |
| 6 | No `zone_id` field introduced | Verified for v1 actor schema/observation | `tests/test_action_units.py`; `tests/test_structured_observation_shapes.py`; `tests/test_privacy_no_private_cost_leak.py` |
| 7 | No `reliability` field introduced | Verified for v1 actor schema/observation | `tests/test_action_units.py`; `tests/test_structured_observation_shapes.py`; `tests/test_privacy_no_private_cost_leak.py` |
| 8 | DSO actor observation contains no VPP private true cost | Verified | `StructuredDSOObservation.metadata["contains_private_true_cost"] = False`; `tests/test_privacy_no_private_cost_leak.py` |
| 9 | ActionUnit supports VPP-PCC or VPP-bus | Verified | `src/vpp_dso_sim/dso/sensitivity/selectors.py::build_action_units`; `tests/test_action_units.py`; `tests/test_sensitivity_finite_difference.py::test_finite_difference_records_action_unit_allocation_weights` |
| 10 | NetworkObject supports bus / line / trafo | Verified | `NetworkObjectId`, `select_critical_network_objects`; `tests/test_network_objects.py` |
| 11 | Sensitivity tensor supports masks | Verified | `SensitivityEdgeTensor.edge_valid_mask`; `tests/test_sensitivity_shapes.py` |
| 12 | Q channels masked if Q control disabled | Verified | `q_channel_mask = False`; Q channel values zero; `tests/test_sensitivity_shapes.py::test_sensitivity_tensor_shape_and_q_mask_when_q_disabled` |
| 13 | Safe decoder invariants are tested | Verified | `src/vpp_dso_sim/dso/envelope/safe_decoder.py`; `tests/test_safe_decoder.py` |
| 14 | Active sensitivity is a slice of raw sensitivity | Verified | `active_sensitivity_slice`; `tests/test_sensitivity_shapes.py::test_active_sensitivity_edges_are_slice_of_raw_tensor` |
| 15 | All configs log seed and config hash | Verified for smoke/HAPPO artifacts | `scripts/run_smoke.py`; `advanced_marl.py`; `current_smoke_summary.json`; `current_happo_training_summary.json` |
| 16 | Reward components are separately logged | Verified | `Simulator.records["reward_components"]`; smoke summary reports `reward_components = 2`; full pytest covers reward tests |
| 17 | `AGENTS.md` updated | Verified | `AGENTS.md` section `20. DSO sensitivity_attention_v1 改造规则` |
| 18 | Memory and handoff docs exist | Verified | `docs/agents/MEMORY.md`, `DECISIONS.md`, `HANDOFF.md`, `KNOWN_FAILURES.md`, `RUNBOOK.md`, `subagents.md`, `repo_map.md` |
| 19 | Baseline and new configs both exist | Verified | `configs/baseline_rule_v0.yaml`, `configs/happo_legacy_mlp.yaml`, `configs/happo_sensitivity_attention_v1.yaml`, ablation configs |
| 20 | Final report lists changed files, tests, risks, next steps | Verified | `docs/experiments/dso_sensitivity_attention_final_implementation_report.md`; detailed Chinese process/loss report at `docs/experiments/dso_sensitivity_attention_full_process_report.md` |

## Additional Prompt Requirements Audited

| Requirement | Status | Evidence |
|---|---|---|
| Do not directly replace FR/DOE hard bounds with neural output | Verified | `safe_decoder.py`; `DecodedOperatingEnvelopeRecord.award_status = envelope_guidance`; `tests/test_safe_decoder.py` |
| `dso_operating_envelope` remains guidance, not market award | Verified | envelope records use `award_status = envelope_guidance`; final report research boundaries |
| Finite-difference sensitivity is not all-zero initialization | Verified | `compute_finite_difference_sensitivity_tensor`; `tests/test_sensitivity_finite_difference.py::test_finite_difference_sensitivity_is_not_zero_initialized` |
| Sensitivity cache supports update triggers and priority refresh | Verified | `decide_sensitivity_refresh`, `merge_sensitivity_update`; `tests/test_sensitivity_shapes.py`; `tests/test_envelope_policy_switch.py` |
| VPP-PCC/VPP-bus perturbation allocation is recorded | Verified | `sensitivity_allocation_mode`, `sensitivity_allocation_weights`; `tests/test_sensitivity_finite_difference.py::test_finite_difference_records_action_unit_allocation_weights` |
| Behavior cloning warm start and residual schedule exist | Verified | `run_short_train.py`; `SensitivityAttentionEnvelopePolicy._residual_schedule_eta`; `tests/test_envelope_policy_switch.py` |
| HAPPO stability logging exists | Verified | `happo_update_metrics.csv`; `tests/test_structured_happo_training.py`; artifacts under `docs/experiments/dso_sensitivity_attention_artifacts/` |
| Hooks and harness exist | Verified | `scripts/agent_harness.py`; `scripts/agent_hooks/*.sh`; `bash scripts/agent_hooks/smoke_training.sh` |
| Harness runs tests and records handoff | Verified | `scripts/agent_harness.py::run_phase`; `tests/test_agent_harness.py` |
| Architecture plan is current | Verified | `docs/dso_sensitivity_attention_architecture_plan.md` documents structured HAPPO, cache refresh, partial merge and allocation weights |

## Latest Verification Commands

```bash
./.venv-server/bin/python -m pytest -q
./.venv-server/bin/python -m pytest --collect-only -q
bash scripts/agent_hooks/smoke_training.sh
```

Latest observed results:

- Full `pytest -q`: exit code 0, one jupyter path deprecation warning.
- Full `pytest --collect-only -q`: 160 collected tests, one jupyter path deprecation warning.
- Hook smoke: rule baseline 2 steps, structured smoke 2 steps, 256-step BC sanity; no NaN/Inf.
- 256-step BC loss curve archived at `docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_curve.svg`.

## Remaining Scientific Boundary

The implementation goal is satisfied at the engineering/sanity-test level. It
does not prove paper-long convergence, statistical significance, or final
paper-ready ablation conclusions.
