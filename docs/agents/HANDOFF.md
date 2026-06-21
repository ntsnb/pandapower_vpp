# DSO Sensitivity Attention Handoff

Updated: 2026-05-28 Asia/Shanghai

## Phase 0: Repository map

- Status: completed.
- Evidence:
  - `docs/agents/repo_map.md`
  - subagent Cicero read-only scan
- Key adapter point:
  - `src/vpp_dso_sim/simulation/simulator.py::_build_dso_operating_envelope`
- Key trainer model point:
  - `src/vpp_dso_sim/learning/ctde_networks.py::build_privacy_separated_ctde_modules`

## Phase 1: Agents and memory

- Status: completed.
- Files created:
  - `docs/agents/subagents.md`
  - `docs/agents/MEMORY.md`
  - `docs/agents/DECISIONS.md`
  - `docs/agents/HANDOFF.md`
  - `docs/agents/KNOWN_FAILURES.md`
  - `docs/agents/RUNBOOK.md`
  - `scripts/agent_harness.py`
  - `scripts/agent_hooks/*.sh`
- Next phase: no pending harness phase; continue with pilot/paper-long experiments only after separate user instruction.

## Phase 2-8: Schema, sensitivity, observation, model, decoder, routing

- Status: completed for current smoke/sanity slice.
- New package:
  - `src/vpp_dso_sim/dso/`
- Key tests:
  - `tests/test_action_units.py`
  - `tests/test_network_objects.py`
  - `tests/test_sensitivity_shapes.py`
  - `tests/test_sensitivity_finite_difference.py`
  - `tests/test_structured_observation_shapes.py`
  - `tests/test_bipartite_attention_actor.py`
  - `tests/test_safe_decoder.py`
  - `tests/test_envelope_policy_switch.py`
  - `tests/test_privacy_no_private_cost_leak.py`
- Result: structured test set passed.

## Phase 9-11: Reward/training, configs, smoke, short sanity

- Status: completed for smoke, BC warm-start sanity, minimal structured HAPPO chain and structured frozen eval.
- Configs:
  - `configs/baseline_rule_v0.yaml`
  - `configs/happo_legacy_mlp.yaml`
  - `configs/happo_sensitivity_attention_v1.yaml`
  - ablation configs
- Scripts:
  - `scripts/run_smoke.py`
  - `scripts/run_short_train.py`
- Output directories:
  - `outputs/dso_sensitivity_attention/baseline_rule_v0_smoke_seed0`
  - `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0`
  - `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0`
  - `outputs/dso_sensitivity_attention/sensitivity_attention_v1_smoke_seed0_current`
  - `outputs/dso_sensitivity_attention/sensitivity_attention_v1_short_train_seed0_current`
  - `outputs/dso_sensitivity_attention/happo_structured_minimal_seed0_current`
  - `outputs/dso_sensitivity_attention/happo_structured_frozen_eval_seed1_current`
- Recent fixes:
  - VPP dispatch preferred-range bonus is now gated by `guidance_strength_lambda`, preferred-width gate and effectiveness gate.
  - `evaluate_happo_checkpoint()` now reconstructs `sensitivity_attention_v1_structured_happo` from checkpoint metadata and structured flat spec.
  - `SensitivityAttentionEnvelopePolicy` now records and uses raw sensitivity cache hits via `active_sensitivity_slice()`.
  - `SensitivityAttentionEnvelopePolicy` can load direct or structured-HAPPO attention actor checkpoints via `dso.actor.checkpoint_path`.
- New 2026-05-28 fixes:
  - `train_happo(config_path=...)` now reads YAML `simulation.seed`, `simulation.horizon_steps` and `trainer.*` stability fields when explicit `HAPPOConfig` is not passed.
  - Runtime `sensitivity_attention_v1` now supports rule warm-start / residual schedule blending and logs `residual_schedule_eta`.
  - HAPPO flattened structured DSO observation now carries `privacy_boundary` and field names for private-field audit.
  - Strict-format final report exists at `docs/experiments/dso_sensitivity_attention_final_implementation_report.md`.
  - Sensitivity cache refresh decisions now cover update period, TTL, voltage/loading drift, FR width changes, projection-gap history, missing ActionUnits/NetworkObjects, priority ActionUnits, and partial priority refresh merge.
  - Finite-difference sensitivity now logs `sensitivity_allocation_mode` and `sensitivity_allocation_weights`.
  - `scripts/agent_harness.py` now runs phase tests, records touched files/test results, and returns nonzero on failed phase tests.
  - `configs/happo_legacy_mlp.yaml` direct 2-step smoke artifact exists under `outputs/dso_sensitivity_attention/happo_legacy_mlp_smoke_seed0` and archived artifacts.
- Latest verification:
  - Affected pytest set: 34 passed, one jupyter path deprecation warning.
  - Full `./.venv-server/bin/python -m pytest -q`: exit code 0, one jupyter path deprecation warning.
  - Full `./.venv-server/bin/python -m pytest --collect-only -q`: 160 collected tests, one jupyter path deprecation warning.
  - `bash scripts/agent_hooks/smoke_training.sh`: baseline smoke, structured smoke and 256-step BC sanity completed with no NaN/Inf.
  - Detailed Chinese process report: `docs/experiments/dso_sensitivity_attention_full_process_report.md`.
  - Archived loss curve: `docs/experiments/dso_sensitivity_attention_artifacts/current_bc_loss_curve.svg`.
- Remaining limitations:
  - Full prompt completion remains unclaimed because full paper-long convergence and broader long-horizon ablation evidence are not yet proven.
  - `docs/architecture.md` already exists as a file, so the architecture plan is stored at `docs/dso_sensitivity_attention_architecture_plan.md` instead of `docs/architecture/dso_sensitivity_attention_plan.md`.

## phase_06_safe_decoder

- Goal: Validate safe envelope decoder invariants.
- Files touched:
  - `GENTS.md`
  - `rc/vpp_dso_sim/entities/dso.py`
  - `rc/vpp_dso_sim/envs/multi_agent_env.py`
  - `rc/vpp_dso_sim/envs/reward_design.py`
  - `rc/vpp_dso_sim/experiments/paper_training.py`
  - `rc/vpp_dso_sim/learning/advanced_marl.py`
  - `rc/vpp_dso_sim/optimization/ac_security_projection.py`
  - `rc/vpp_dso_sim/simulation/scenario.py`
  - `rc/vpp_dso_sim/simulation/simulator.py`
  - `ests/test_ac_security_projection.py`
  - `ests/test_paper_training_experiment.py`
  - `ests/test_reward_rebalance.py`
  - `configs/ablation_no_action_self_attention.yaml`
  - `configs/ablation_no_sensitivity_edges.yaml`
  - `configs/ablation_no_width_penalty.yaml`
  - `configs/baseline_rule_v0.yaml`
  - `configs/happo_legacy_mlp.yaml`
  - `configs/happo_sensitivity_attention_v1.yaml`
  - `docs/agents/`
  - `docs/dso_sensitivity_attention_architecture_plan.md`
  - `docs/dso_sensitivity_attention_upgrade_report.md`
  - `docs/experiments/`
  - `docs/reward_terms_chinese_explained.md`
  - `paper_method_experiment_cn.aux`
  - `paper_method_experiment_cn.fdb_latexmk`
  - `paper_method_experiment_cn.fls`
  - `paper_method_experiment_cn.pdf`
  - `paper_method_experiment_cn.xdv`
  - `scripts/agent_harness.py`
  - `scripts/agent_hooks/`
  - `scripts/run_short_train.py`
  - `scripts/run_smoke.py`
  - `src/vpp_dso_sim/dso/`
  - `src/vpp_dso_sim/experiments/dso_sensitivity_attention.py`
  - `tests/test_action_units.py`
  - `tests/test_agent_harness.py`
  - `tests/test_bipartite_attention_actor.py`
  - `tests/test_envelope_policy_switch.py`
  - `tests/test_legacy_baseline_unchanged.py`
  - `tests/test_network_objects.py`
  - `tests/test_privacy_no_private_cost_leak.py`
  - `tests/test_safe_decoder.py`
  - `tests/test_sensitivity_finite_difference.py`
  - `tests/test_sensitivity_shapes.py`
  - `tests/test_structured_happo_training.py`
  - `tests/test_structured_observation_shapes.py`
  - `tests/test_structured_smoke_rollout.py`
  - `tests/test_training_step_no_nan.py`
- Tests run: `/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim/.venv-server/bin/python -m pytest -q tests/test_safe_decoder.py`
- Test result: passed
