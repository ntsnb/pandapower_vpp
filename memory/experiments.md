# Experiments

Record reproducible experiment runs here.

## Template

- Date:
- Config:
- Command:
- Horizon:
- Experiment level: smoke | demo | benchmark | paper-claim-ready
- Algorithm mode:
- Privacy mode:
- Seed count:
- Holdout scenario:
- Oracle baseline:
- Baselines:
- Reviewed by architecture/domain subagent:
- Reviewed by experiment subagent:
- UI artifact:
- UI refresh command:
- Key metrics:
- Accepted / rejected:
- Notes:

## Experiment Level Rules

- `smoke`: code path executes; no performance claim.
- `demo`: scenario and visualization are interpretable; still no generalization
  claim.
- `benchmark`: multiple seeds, repeatable configs, comparable baselines, and
  explicit train/eval or holdout split.
- `paper-claim-ready`: benchmark evidence plus realistic public data,
  oracle/full-information baseline, ablations, robustness, and defensible
  settlement/economic metrics.

Any result from `examples/10_train_deep_rl.py --episodes 1 --horizon-steps 2`
or similar is automatically `smoke`, even if reward improves.

## 2026-05-01: Benchmark V2 Research Candidate

- Config: `configs/european_lv_benchmark_v2.yaml`
- Command: `python examples/11_run_benchmark_experiment.py --output-dir outputs\benchmark_v2_research_candidate`
- Horizon: `288` steps at 15-minute resolution.
- Experiment level: `benchmark` / `research-grade candidate`.
- Algorithm mode: `rule_based` executable baseline.
- Privacy mode: mixed `full_information` and `representative_data` VPPs.
- Seed count: `5` (`3101` to `3105`).
- Splits:
  - `train_profile / train_mixed`
  - `eval_profile / holdout_peak`
  - `eval_profile / holdout_cloudy`
  - `topology_holdout / holdout_peak` on IEEE33
- Holdout scenario: `configs/ieee33_multi_vpp.yaml`.
- Oracle baseline: not yet implemented.
- Baselines: `rule_based` only in this runner; CTDE/MARL must be connected next.
- Reviewed by domain/experiment subagents:
  - Grid/VPP domain audit identified repeated-day profiles, weak feeder coverage,
    and aggregate-only DSO guidance.
  - MARL audit identified that current runner is not yet a train-then-frozen-eval
    MARL benchmark and must connect CTDE training artifacts next.
- UI artifact:
  - `outputs/benchmark_v2_research_candidate/benchmark_report.html`
- Key metrics:
  - Rollouts: `20`
  - Splits: `train_profile`, `eval_profile`, `topology_holdout`
  - Global `min_voltage_vm_pu=0.9338`
  - Global `max_line_loading_percent=92.04`
  - Global `max_trafo_loading_percent=60.69`
  - Security pass rate: `1.000`
  - Train split: `min_voltage≈0.9383`, `max_line≈90.55`
  - Eval split: `min_voltage≈0.9338`, `max_line≈92.04`
  - Topology holdout: `min_voltage≈0.9614`, `max_line≈24.83`
- Accepted / rejected: accepted as a second-stage research candidate benchmark,
  not accepted as final paper evidence.
- Notes:
  - `european_lv_benchmark_v2` was calibrated to near-limit operation with
    `base_load_scale=1.60`, `transformer_sn_mva=2.05`,
    `line_resistance_scale=1.16`, and `line_reactance_scale=1.10`.
  - The scenario now has 7 VPPs and DER coverage across feeders F1-F6.
  - `benchmark_profile_pack` replaces repeated 96-point daily replay.
  - Current benchmark proves the experiment protocol and rule-based baseline,
    but not MARL superiority.

## 2026-05-01: Benchmark V2.1 CTDE Protocol Pilot

- Configs:
  - `configs/european_lv_benchmark_v2.yaml`
  - `configs/european_lv_benchmark_v2_safety_tight.yaml`
- Command:
  - `python examples/11_run_benchmark_experiment.py --output-dir outputs\benchmark_v21_pilot --horizon-steps 48 --seeds 5201 --train-variants train_mixed --eval-variants holdout_peak,holdout_cloudy,holdout_reverseflow --algorithms rule_based,privacy_separated_ctde_actor_critic --ctde-train-episodes 2 --ctde-train-horizon-steps 48 --ctde-eval-horizon-steps 48`
- Horizon: `48` steps pilot.
- Experiment level: protocol pilot, not performance claim.
- Algorithms:
  - `rule_based`
  - `privacy_separated_ctde_actor_critic` with train-then-frozen-eval protocol.
- Splits:
  - `train_profile`
  - `eval_profile`
  - `safety_tight_limits`
  - `topology_holdout` for rule-based only
- Key metrics:
  - Rollouts: `11`
  - `security_pass_rate=0.818`
  - `safety_tight_limits` triggers safety failures for both rule-based and CTDE,
    so it is useful for safety-margin testing.
  - `holdout_reverseflow` now uses high PV capacity and records reverse-flow
    metrics such as `reverse_flow_step_rate` and `min_line_p_from_mw`.
  - Short-budget CTDE cost is higher than rule-based in this pilot, so no
    superiority claim is allowed.
- Accepted / rejected: accepted as protocol and metrics pilot.
- Notes:
  - Added `privacy_preserving_proxy` reward mode to avoid using VPP private
    operating cost as the default benchmark reward.
  - Added `step_summary.csv` and expanded `seed_metrics.csv` fields for
    step-rate, projection, FR binding, service-request and reverse-flow metrics.

## 2026-04-29: Phase 2/3/4 v0 Smoke Validation

- Config: `configs/ieee33_multi_vpp.yaml`
- Commands:
  - `python -m pytest`
  - `python examples/03_timeseries_multi_vpp.py`
  - `python examples/08_run_dashboard.py --check`
  - `python examples/05_random_rl_env_rollout.py`
- Horizon: 288 steps for the main simulation, 10 steps for random RL rollout.
- Algorithm mode:
  - Rule-based VPP dispatch
  - Static local-bound FR/DOE v0
  - Stress-driven LFP/LFB v0 pure-function market interface
- Privacy mode:
  - Existing scenario remains `full_information`
  - New actor-observation builder defaults to redacting private DER cost fields
- UI artifact:
  - `outputs/interactive_report.html`
  - `outputs/dashboard_data/vpp_portfolio.csv`
  - `outputs/dashboard_data/feasible_region.csv`
  - `outputs/dashboard_data/fr_envelope_state.csv`
  - `outputs/dashboard_data/projection_trace.csv`
  - `outputs/dashboard_data/privacy_visibility.csv`
- Key metrics:
  - `33 passed in 8.40s`
  - `steps=288`
  - `min_voltage_vm_pu=0.9636`
  - `max_line_loading_percent=24.29`
  - dashboard frames loaded with `projection_trace=6048`,
    `fr_envelope_state=5184`, `feasible_region=2592`,
    `privacy_visibility=23`.
- Accepted / rejected: accepted.
- Notes:
  - LFP/LFB is implemented as a network-service signal, not LMP.
  - FR/DOE v0 is local-bound based and not yet OPF-certified.

## 2026-04-29: Heterogeneous MARL Baseline Smoke

- Config: `configs/ieee33_multi_vpp.yaml`
- Command: `python examples/09_run_marl_baselines.py`
- Horizon:
  - Baseline comparison: 8 steps x 2 episodes per algorithm.
  - Tuning sweep: 6 steps x 1 episode per trial.
- Algorithms:
  - `ippo`
  - `mappo`
  - `maddpg`
  - `qmix`
- Agent roles:
  - `dso_global_guidance`
  - per-VPP `vpp_dispatch_agent`
  - per-VPP `vpp_portfolio_agent`
  - `deep_training_supervisor`
- Output:
  - `outputs/marl_baselines/agent_role_map.csv`
  - `outputs/marl_baselines/encoder_role_map.csv`
  - `outputs/marl_baselines/episode_metrics.csv`
  - `outputs/marl_baselines/step_metrics.csv`
  - `outputs/marl_baselines/tuning_trials.csv`
  - `outputs/marl_baselines/training_summary.csv`
  - `outputs/marl_baselines/baseline_summary.json`
  - `outputs/marl_baselines/training_summary.json`
- Key result:
  - Best baseline in this smoke run: `mappo`
  - Best episode reward: `-250.828`
  - Training supervisor status: `converged`
  - `deep_learning_available=True`
- Accepted / rejected: accepted as experiment harness smoke.
- Notes:
  - Current baselines are lightweight runnable approximations of classic MARL
    families. They are not yet full PyTorch IPPO/MAPPO/MADDPG/QMIX training
    implementations.

## 2026-04-29: Complex European-LV Mixed-VPP Scenario Smoke

- Config: `configs/european_lv_mixed_vpp.yaml`
- Commands:
  - `python examples/03_timeseries_multi_vpp.py`
  - `python examples/08_run_dashboard.py --check`
  - `python -m pytest -q`
  - `python examples/09_run_marl_baselines.py`
- Network:
  - 123 buses
  - 121 LV lines
  - 1 transformer
  - 6 VPPs
  - 28 DER assets
  - 2 deterministic slow portfolio events
- Horizon: 288 steps at 0.25 h per step.
- Algorithm mode:
  - Rule-based target projection and DER disaggregation
  - Static local-bound FR/DOE
  - First-person VPP timeline generated from actual simulated traces
  - Minimal parallel multi-agent dict env added for DSO / VPP dispatch / VPP portfolio roles
- UI artifact:
  - `outputs/interactive_report.html`
  - `outputs/dashboard_data/vpp_first_person_timeline.csv`
  - `outputs/dashboard_data/vpp_first_person_scope_detail.csv`
  - `outputs/dashboard_data/vpp_portfolio_history.csv`
  - `outputs/dashboard_data/portfolio_change_log.csv`
- Key metrics:
  - `python -m pytest -q`: 44 passed
  - `steps=288`
  - `min_voltage_vm_pu=0.9441`
  - `max_line_loading_percent=90.64`
  - `alert_event=0`
  - `vpp_first_person_timeline=78`
  - `vpp_first_person_scope_detail=11712`
- Accepted / rejected: accepted as the new complex scenario smoke baseline.
- Notes:
  - The 123-bus feeder is an engineering demo model inspired by European LV topology, not a verbatim regulatory benchmark.
  - Next training milestone is a true trajectory buffer plus PyTorch IPPO/MAPPO loop over `MultiAgentVPPDSOEnv`.
