# Progress

## 2026-05-02: MATD3 Multi-Head Critic And Runnable HAPPO/HASAC Scaffolds

- Upgraded `learning/matd3.py` from a DSO Q plus mean dispatch reward path to a
  centralized twin critic with one DSO Q head and one independent dispatch Q
  head per VPP.
- Stored multi-head reward vectors in MATD3 replay, persisted critic head names
  in checkpoints, and exposed per-head diagnostics in update metrics.
- Added runnable HAPPO research scaffold in `learning/advanced_marl.py`:
  sequential role updates for DSO / shared dispatch / shared portfolio actors,
  cumulative importance correction, and centralized multi-head value baselines.
- Added runnable HASAC research scaffold in `learning/advanced_marl.py`:
  off-policy replay, squashed Gaussian actors, centralized twin soft Q,
  automatic entropy tuning, and soft target backup for the continuous DSO/VPP
  dispatch path.
- Added `examples/16_train_happo_hasac.py` and focused tests in
  `tests/test_hasac_happo.py`.
- Validation:
  - `python -m pytest tests\test_advanced_marl.py tests\test_matd3_training.py tests\test_hasac_happo.py -q --basetemp=outputs\pytest_tmp_advanced_algorithms -o cache_dir=outputs\pytest_cache_advanced_algorithms`
    passed with `13 passed`.
  - `python -m pytest tests\test_deep_rl_training.py tests\test_algorithm_search.py -q --basetemp=outputs\pytest_tmp_advanced_regression -o cache_dir=outputs\pytest_cache_advanced_regression`
    passed with `12 passed`.
  - `python examples\16_train_happo_hasac.py --algorithm hasac --episodes 1 --horizon-steps 2 --hidden-dim 16 --batch-size 2 --warmup-steps 2 --seed 71 --output-dir outputs\example_advanced_marl --eval`
    completed and wrote a frozen eval summary.

## 2026-05-01: Safety Projection Reward Penalty

- Clarified and fixed the learning issue where safety projection could repair
  infeasible raw DSO/VPP actions without a unified environment-level cost.
- Added `action_projection_gap_mw`, `action_projection_count`, and
  `action_projection_penalty` to DSO reward components.
- `Simulator.step()` now passes raw-to-projected target gaps into
  `DSO.calculate_reward_or_cost()`.
- `MultiAgentVPPDSOEnv` now records dispatch projection gaps in the decoded
  action audit and forwards pre-simulator projection gaps to the simulator.
- Removed the ad hoc extra projection penalty from the privacy-separated CTDE
  trainer so the environment reward is the single source of truth.
- Updated README and static HTML/Dash reward explanations to state that
  `total_cost` includes `action_projection_penalty`.
- Refreshed root HTML reports and benchmark-aware HTML reports.
- Validation:
  - `python -m pytest tests\test_multi_agent_env.py tests\test_timeseries_smoke.py tests\test_env_smoke.py tests\test_deep_rl_training.py -q --basetemp=outputs\pytest_tmp_projection_reward_deep -o cache_dir=outputs\pytest_cache_projection_reward_deep`
    passed with `14 passed`.
  - `python -m pytest tests\test_visualization_data.py -q --basetemp=outputs\pytest_tmp_projection_reward_viz -o cache_dir=outputs\pytest_cache_projection_reward_viz`
    passed with `6 passed`.

## 2026-05-01: Benchmark-Aware HTML Synchronization

- Fixed the algorithm/UI synchronization gap for Benchmark V2.1.
- Added `visualization/benchmark_report.py`, which reads the benchmark output
  directory directly instead of rerunning a generic scenario.
- `examples/11_run_benchmark_experiment.py` now refreshes benchmark-local HTML
  by default:
  - `benchmark_report.html`
  - `interactive_report.html`
  - `rl_architecture.html`
  - `vpp_first_person/index.html`
  - `dashboard_data/benchmark_*.csv`
- Benchmark pages now expose `holdout_reverseflow`,
  `policy_evaluation_mode=frozen_deterministic_mean_policy`,
  `privacy_preserving_proxy`, projection gaps, FR binding, reverse-flow rates,
  and per-run `step_summary.csv` links.
- Added pytest assertions so benchmark HTML synchronization is checked during
  `tests/test_benchmark_v2_experiment.py`.
- Validation:
  - `python -m pytest tests\test_benchmark_v2_experiment.py -q --basetemp=outputs\pytest_tmp_benchmark_ui_sync -o cache_dir=outputs\pytest_cache_benchmark_ui_sync`
    passed with `8 passed`.
  - `python -m pytest tests\test_visualization_data.py -q --basetemp=outputs\pytest_tmp_visualization_sync -o cache_dir=outputs\pytest_cache_visualization_sync`
    passed with `6 passed`.

## 2026-05-01: Project Agent Consolidation And Memory Governance

- Added `agents/` as a reusable project-level agent pack.
- Added `agents/project_agent_registry.yaml` to map overlapping global agents
  into project-specific roles.
- Added project agent docs for:
  - main supervision
  - MARL architecture
  - grid/VPP experiment audit
  - power-system modeling audit
  - UI visualization synchronization
  - memory curation
- Added `ppvpp-*.toml` project-agent registration drafts for future reuse or
  copying into a global Codex agent directory if desired.
- Added `agents/subagent_overlap_audit.md` summarizing overlapping global
  agent groups and project-level resolution.
- Added `memory/user_preferences.md` with durable user preferences and work
  rules.
- Updated memory rules and decisions to require algorithm/UI/experiment/memory
  synchronization gates.

Open follow-up:

- Decide whether to install the `agents/ppvpp-*.toml` drafts into the global
  `C:\Users\admin\.codex\agents` registry later, or keep them as project-local
  read-before-routing definitions.

Entries are newest first.

## 2026-05-01: Benchmark V2 Experiment Orchestration

- Added `configs/european_lv_benchmark_v2.yaml`, a branch-rich 123-bus
  European-LV benchmark candidate with 7 VPPs, mixed single-PCC and multi-node
  portfolios, DER coverage across feeders F1-F6, and calibrated near-limit
  network stress.
- Replaced repeated 96-point CSV replay in `european_lv_mixed_vpp.yaml` with
  the non-repeating `benchmark_profile_pack` path.
- Added non-repeating profile generation and daily profile quality summaries in
  `simulation/profiles.py`.
- Added `experiments/benchmark_runner.py` and
  `examples/11_run_benchmark_experiment.py` for multi-seed
  train/eval/topology-holdout benchmark orchestration.
- Fixed `network/builder.py` so European-LV electrical builder kwargs are
  filtered separately from DSO constraint keys such as `voltage_limits`.
- Added `docs/benchmark_v2_experiment_plan.md` to document the second-stage
  benchmark protocol, current evidence boundary, and next CTDE/MARL integration
  steps.
- Validation:
  - `python -m pytest tests\test_benchmark_v2_experiment.py tests\test_multi_agent_env.py tests\test_deep_rl_training.py -q --basetemp=outputs\pytest_tmp_benchmark_marl_fix_a -o cache_dir=outputs\pytest_cache_benchmark_marl_fix_a`
    passed with `16 passed`.
  - `python examples\11_run_benchmark_experiment.py --output-dir outputs\benchmark_v2_research_candidate`
    produced 20 rollouts, `min_voltage=0.9338`, `max_line_loading=92.04`,
    and `security_pass_rate=1.000`.

## 2026-05-01: Benchmark V2.1 Protocol Upgrade

- Connected `privacy_separated_ctde_actor_critic` to
  `experiments/benchmark_runner.py` through train-then-frozen-eval protocol.
- Added frozen deterministic checkpoint evaluation in
  `learning/deep_rl.py`.
- Added policy compatibility signatures to `MultiAgentVPPDSOEnv` and CTDE
  checkpoint summaries.
- Added `reward_privacy_mode` to DSO reward calculation. Benchmark configs now
  use `privacy_preserving_proxy` by default, while private VPP operation cost is
  retained as a reference field.
- Added `holdout_reverseflow` profile behavior and PV capacity scaling during
  reverse-flow evaluation.
- Added `configs/european_lv_benchmark_v2_safety_tight.yaml` with tighter
  voltage/loading limits plus F1 load and F2 line-capacity perturbations.
- Added feeder-level network perturbation support:
  `feeder_load_scale` and `line_capacity_scale_by_feeder`.
- Expanded benchmark metrics with near-voltage/near-line step rates,
  percentiles, projection gap/clipping, FR binding, service-request counts,
  reverse-flow rate, and per-run `step_summary.csv`.
- Validation:
  - `python -m pytest tests\test_benchmark_v2_experiment.py tests\test_multi_agent_env.py tests\test_deep_rl_training.py -q --basetemp=outputs\pytest_tmp_v21_b -o cache_dir=outputs\pytest_cache_v21_b`
    passed with `18 passed`.
  - `python examples\11_run_benchmark_experiment.py --output-dir outputs\benchmark_v21_pilot --horizon-steps 48 --seeds 5201 --train-variants train_mixed --eval-variants holdout_peak,holdout_cloudy,holdout_reverseflow --algorithms rule_based,privacy_separated_ctde_actor_critic --ctde-train-episodes 2 --ctde-train-horizon-steps 48 --ctde-eval-horizon-steps 48`
    produced 11 pilot rollouts, `security_pass_rate=0.818`, and expected
    failures under `safety_tight_limits`.

## 2026-05-01

- Upgraded the primary `privacy_separated_ctde_actor_critic` trainer from a
  flat padded dispatch baseline to a light Deep Sets style VPP dispatch
  encoder plus an action-conditioned centralized critic summary path.
- Added `learning/ctde_networks.py` to hold the shared-token DER encoder,
  joint-action summary encoder, and privacy-separated CTDE module builder so
  `learning/deep_rl.py` no longer carries all architecture logic inline.
- Kept the existing `16 + K * 15` VPP dispatch observation contract and
  training script entrypoint stable while changing the internal encoder.
- Extended `deep_rl_training_summary.csv/json`, step metrics, trajectory rows,
  and checkpoints with architecture metadata:
  `architecture_version`, `vpp_encoder_type`, `critic_type`,
  `action_conditioned_critic`, and critic action-summary dimensions.
- Updated `learning/rl_architecture.py` so dashboard/report DataFrames now
  describe the implemented Deep Sets encoder and action-conditioned critic
  instead of labeling them as future work.
- Extended `tests/test_deep_rl_training.py` to assert the new architecture
  fields, checkpoint parameter keys, and the `16 + K * 15` token split.
- Validation:
  `python -m pytest tests/test_deep_rl_training.py -q`,
  `python examples/10_train_deep_rl.py --episodes 1 --horizon-steps 2`, and
  `python -m pytest tests/test_visualization_data.py -q` all passed.

## 2026-04-30

- Reworked the RL/MARL report visualization layer without modifying
  `learning/*` algorithms.
- Added `visualization/rl_architecture_diagram.py`, a shared paper-style arrow
  workflow component with grouped colors, slow-loop portfolio branch, and
  clickable agent detail chips.
- Integrated the shared workflow figure into
  `visualization/rl_architecture_report.py` and
  `visualization/interactive_report.py`.
- Regenerated `outputs/interactive_report.html` and `outputs/rl_architecture.html`
  so the root outputs now include the new arrow-based architecture view instead
  of relying on cards/tables alone.
- Validation: `python -m pytest tests/test_visualization_data.py -q` passed;
  `python examples/07_interactive_report.py` refreshed the interactive report.

## 2026-04-30

- Added a minimal CTDE engineering contract in
  `learning/ctde_interface.py` without rewriting the current shared
  actor-critic trainer.
- The new contract exposes explicit DSO / VPP actor specs, centralized critic
  metadata, and policy-module scaffolds for both
  `shared_actor_critic` and `independent_actor_scaffold` layouts.
- `MultiAgentVPPDSOEnv` now exposes CTDE actor/critic specs on `reset()` and
  action-validation traces on `step()`.
- Added action-schema normalization and validation for legacy and current
  payloads, including `selected_p_mw`, `normalized_setpoint_bias`,
  `target_p_mw`, `response_bias`, `der_actions`, and portfolio action labels.
- Added focused tests for CTDE contract generation, action validation success
  and failure paths, and kept the existing multi-agent / deep-RL smoke tests
  passing.
- Validation run:
  `python -m pytest tests/test_ctde_interface.py tests/test_multi_agent_env.py tests/test_deep_rl_training.py -q`
  and `ruff check` on the touched files both passed.

## 2026-04-29

- Implemented Phase 2 / early Phase 3 code:
  `entities/schemas.py`, `optimization/feasibility_region.py`, and
  `envs/observations.py`.
- Added tests for schema serialization, physical injection mapping,
  feasibility-region projection, and privacy-filtered observations.
- Extended `Simulator.step()` with command projection audit records:
  `raw_action`, `device_bounds`, `fr_doe`, `pandapower_write`, and
  `powerflow_result`.
- Extended dashboard frames with `vpp_portfolio`, `feasible_region`,
  `fr_envelope_state`, `projection_trace`, and `privacy_visibility`.
- Added FR/DOE, Projection Audit, and Privacy Visibility tables to offline HTML
  and Dash overview.
- Implemented `optimization/local_flex_market.py` for Phase 4 v0:
  stress-driven `LocalFlexNeed`, `LocalFlexPrice`, rule-based VPP bids, and
  simple clearing to `DispatchAward`.
- Fixed a package import cycle by making `entities.__init__` load `DSO` and
  `VPPAggregator` lazily.
- Implemented the first heterogeneous MARL baseline harness:
  `learning/agent_roles.py`, `learning/encoders.py`,
  `learning/marl_baselines.py`, and `learning/tuning.py`.
- Added `examples/09_run_marl_baselines.py`, which runs IPPO/MAPPO/MADDPG/QMIX
  style baselines and a training supervisor hyperparameter sweep.
- Confirmed local `torch` is available; current baselines are lightweight
  runnable policies, with PyTorch network replacement left as the next training
  implementation step.
- Extended training outputs with `training_summary.csv` and explicit
  `handoff_target` / `handoff_message` fields for non-convergence review.
- Extended Dash and offline HTML to display `Training Supervisor Status`,
  baseline episode metrics, tuning trials and the agent role map.
- Re-ran subagent review using `gpt-5.5` with `xhigh` reasoning after the user
  requested model quality consistency.
- Recorded the future subagent model policy in the root `AGENTS.md` and project
  memory.
- Algorithm review confirmed the next minimum milestones: schema / physical
  injection contracts, FR/DOE v0, privacy observation filters, LFP/LFB v0-v1,
  VPP inner solver, and multi-VPP clearing.
- UI review confirmed the matching visualization path: FR/DOE envelope panels,
  Local Flex Market heatmaps, clearing / settlement tables, privacy visibility
  inspector, RL diagnostics, and capability / portfolio pages.
- Read the updated root `AGENTS.md` and aligned the project roadmap with the
  bidirectional DSO / multi-VPP guidance direction.
- Added the `memory/` documentation layer:
  `concepts.md`, `decisions.md`, `rules.md`, `pitfalls.md`,
  `progress.md`, `open_questions.md`, and `experiments.md`.
- Updated `docs/roadmap.md` into a phased plan from schema / physical
  consistency through FR/DOE, LFP/LFB, VPP inner solver, multi-VPP clearing,
  MARL/HRL, capability learning, portfolio search, and oracle baselines.
- Recorded the rule that UI / explainability work must accompany algorithmic
  changes.

## 2026-04-29 Complex Scenario / First-Person Replay

- Added `network/european_lv.py`, a 123-bus European-LV-style balanced pandapower feeder with 121 LV lines, one 10/0.4 kV transformer, and normally-open tie-switch metadata.
- Added `configs/european_lv_mixed_vpp.yaml` and package resource copy. The scenario has 6 VPPs, 28 DER assets, both `single_pcc` and `multi_node` VPPs, and two slow portfolio events.
- Extended `SimulationScenario` and `Simulator` with deterministic `portfolio_events`, `vpp_portfolio_history`, and `portfolio_change_log` records. Portfolio events change commercial VPP ownership only; physical bus and pandapower element indices remain unchanged.
- Extended dashboard frames with `vpp_first_person_timeline` and `vpp_first_person_scope_detail` for VPP first-person Saw / Inferred / Decided replay.
- Updated offline HTML and Dash overview to expose first-person replay, scope details, and slow portfolio changes.
- Added `MultiAgentVPPDSOEnv`, a minimal PettingZoo-parallel-style dict interface with `dso_global_guidance`, per-VPP dispatch agents, and per-VPP portfolio agents.
- Strengthened VPP actor privacy observations by removing `cost_coefficients` and private `metadata` keys by default.
- `examples/03_timeseries_multi_vpp.py` and `examples/07_interactive_report.py` now use the complex mixed-VPP LV scenario by default.
- Validation: `python -m pytest -q` -> 44 passed. Full complex example generated `steps=288`, `min_voltage_vm_pu=0.9441`, `max_line_loading_percent=90.64`, `portfolio_change_log=2`, and `vpp_first_person_timeline=78`.

## 2026-05-02 Role-Specific General-Sum Rewards

- Replaced the multi-agent environment's shared global reward fan-out with
  role-specific rewards:
  DSO grid-safety/procurement reward, VPP dispatch self-interest settlement
  reward, and VPP portfolio long-horizon profit/reliability reward with
  localized DSO-alignment credit.
- Added `src/vpp_dso_sim/envs/reward_design.py` as the single source for
  reward formulas and per-agent reward components.
- Updated `train_deep_rl_actor_critic()` and `train_privacy_separated_ctde()`
  so DSO, dispatch, and portfolio policy losses use separate role returns.
- Updated architecture/report generators so `interactive_report.html`,
  `rl_architecture.html`, Dash, and benchmark pages describe the new
  general-sum incentive-aligned reward design.
- Added tests asserting VPP dispatch and portfolio rewards expose local profit
  and localized DSO-alignment components instead of simply sharing DSO reward.

## 2026-05-11 MARL Convergence Stabilization And HATRPO

- Applied DeepRL convergence fixes to the main MARL family:
  HAPPO now defaults to a state-only centralized value baseline, uses scaled
  value targets, clips cumulative importance correction, and records the
  baseline mode in training summaries.
- Stabilized HASAC and MATD3 with reward scaling, target-Q clipping, gradient
  clipping, and separated DSO actor objective from VPP dispatch actor objective.
  HASAC dispatch log-prob now masks padded DER action dimensions.
- Added `learning/hatrpo.py`: a runnable CTDE HATRPO implementation with
  Gaussian DSO/dispatch policies, categorical portfolio policy, centralized
  training-only value critic, Fisher-vector product, conjugate gradient, KL
  trust-region line search, checkpointing, and frozen deterministic evaluation.
- Added HATRPO to the algorithm registry, capability metadata, package exports,
  `examples/16_train_happo_hasac.py`, and `paper_training` train/eval dispatch.
- Hardened `network/powerflow.py` for the migrated server by disabling
  pandapower numba execution paths and replacing broken numba helper dispatchers
  with pure Python helpers before `runpp`.
- Validation completed:
  targeted HAPPO/HATRPO/HASAC/MATD3 tests passed, physical/privacy/DeepRL
  regression tests passed, full `pytest -q` passed, `paper_training` smoke with
  HATRPO passed, and a 24-step/2-episode compressed pilot produced finite
  metrics with zero violation cells for all evaluated algorithms.

## 2026-05-11 Paper-Long Preflight Hardening

- Integrated the paper-long audit findings from the algorithm, experiment, and
  distribution-system subagents before launching long campaigns.
- `paper_training` now disables stale artifact reuse by default; reuse requires
  `--resume-completed`, and diagnostics record when reuse is enabled.
- Baseline and frozen RL evaluation profiles are now paired by the same
  holdout profile seed and `profile_hash`, so baseline/RL comparisons are not
  polluted by different load/PV/price traces.
- Evaluation summaries now split `dso_reward_sum`, `dispatch_reward_sum`,
  `portfolio_reward_sum`, `total_agent_reward_sum`, `raw_objective_reward_sum`,
  total cost, and post-AC violation counters.
- Frozen evaluation reward semantics were aligned with training for
  HAPPO/HATRPO/MATD3/HASAC: shield-intervention penalties are included, and
  portfolio rewards are masked outside slow decision steps where applicable.
- Paper-long defaults now use `gamma=0.995`, final-checkpoint evaluation by
  default, lower HASAC initial entropy temperature, and runtime/config/profile
  hashes in the manifest.
- `holdout_reverseflow` profile generation now applies the same PV capacity
  amplification as the benchmark protocol.
- `run_powerflow()` now catches both BFSW and NR failure paths and returns
  `converged=False` instead of aborting the long campaign.
- Validation completed: targeted MARL/paper-training tests passed, a fresh
  all-algorithm 8-step smoke run wrote paired profile hashes and split reward
  metrics, reverseflow config generation showed PV amplification, and full
  `pytest -q` passed.

## 2026-05-11 AC-Aware DOE And Baseline Claim Guardrails

- Implemented the high-severity cross-agent fixes from `marl_architecture_engineer`,
  `power_system_modeling_auditor`, and `grid_vpp_experiment_auditor`.
- Kept the MARL actor action contract unchanged, but converted scalar VPP
  dispatch targets into bus/zone/DER-scope vector targets inside the simulator.
  Multi-node VPP disaggregation now repairs residuals within each local
  electrical scope instead of across the whole VPP.
- Added `optimization/ac_security_projection.py`: each joint VPP candidate
  dispatch is replayed through pandapower AC power flow on a network copy, then
  accepted or backed off from the current dispatch. Projection traces now include
  `bus_vector_doe` and `ac_pf_certificate` stages plus certificate status,
  accepted alpha, and repair gap.
- Reward and shield-intervention metrics now include
  `ac_certified_projection_gap_mw` and `ac_certificate_failed_count`, so a
  policy cannot hide unsafe requests behind a local FR/DOE projection.
- Added `optimization/oracle_baseline.py` and the
  `ac_validated_search_reference` baseline. The legacy `opf_oracle_proxy` path
  is explicitly degraded to a static FR price heuristic and is not allowed to
  support OPF/oracle/upper-bound claims.
- `paper_training` now writes baseline claim metadata
  (`baseline_role`, `is_ac_validated`, `is_search_based`,
  `is_upper_bound_claim_allowed`, search budget, feasible candidate count) and
  hard diagnostics such as `paper_claim_blocked`,
  `oracle_proxy_not_upper_bound`, and `ac_reference_not_upper_bound`.
- Baseline stale-result reuse is now also gated by `resume_completed`, matching
  the train/eval artifact reuse policy.
- Added a Chinese dynamic explanation page at
  `outputs/high_severity_fix_explainer.html`, generated by
  `examples/18_high_severity_fix_explainer.py`, with before/after diagrams for
  multi-bus DOE, the AC power-flow certificate shield, and baseline claim
  boundaries.
- Validation completed: targeted feasibility/timeseries/paper-training tests
  passed, full `pytest -q` passed, and `paper_training` smoke completed all 12
  baseline/train/frozen-eval runs with HAPPO, HATRPO, MATD3, and HASAC.

## 2026-05-11 Paper-Long Gate Fixes

- Re-ran paper-long gate review with the MARL architecture, grid/VPP
  experiment, and power-system modeling subagents.
- Fixed the experiment-auditor high blocker in `ac_validated_search_reference`:
  the best AC-validated candidate now returns explicit `der_dispatch_p_mw`
  actions, and the simulator can execute those DER setpoints directly instead
  of collapsing them back to a scalar target and redisaggregating them.
- Fixed the power-system high blocker where the AC certificate preview wrote
  pandapower tables with a different Q/set-power semantic than final execution.
  The certificate now previews dispatch through a copied DER object's formal
  `set_power(net, p, 0.0)` path, matching `VPPAggregator.apply_dispatch_to_net`
  without mutating live DER state.
- Added first-class multi-node vector actions via `selected_p_by_scope` /
  `target_by_bus`; the simulator projects them with the vector FR/DOE path and
  records `vector_scope_target` execution.
- Hardened pre-dispatch power-flow failure handling: envelope generation now
  records `pre_dispatch_powerflow_converged`, avoids stale grid results, skips
  AC sensitivity tightening, and falls back to holding current dispatch when the
  pre-dispatch runpp path fails.
- Paper-long defaults no longer include the duplicate
  `static_fr_price_extreme_proxy`; it remains available explicitly, but the
  default baseline set is now `rule_based`, `no_flex`, and
  `ac_validated_search_reference`.
- Added `ac_reference_max_candidates` to the paper-training config and CLI so
  long-run AC-search reference cost is controlled and auditable.
- Updated README/paper-training text to remove stale `opf_oracle_proxy` launch
  guidance and avoid oracle/upper-bound language unless a strict AC OPF/MILP
  reference is added.
- Validation completed: targeted AC/security/timeseries/paper-training/grid
  tests passed, fresh paper-training smoke completed 11 runs with no high
  diagnostics, and full `pytest -q` passed with 42 tests.
