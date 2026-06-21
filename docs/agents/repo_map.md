# Repository Map For DSO Sensitivity-Attention Upgrade

Updated: 2026-05-28 Asia/Shanghai

## 1. Objective

Map the current repository before implementing `sensitivity_attention_v1`.

Authoritative prompt:

```text
/mnt/sda/home/niutiansen/dso_sensitivity_attention_agent_prompt.md
```

## 2. Existing Project Root

```text
/mnt/sda/home/niutiansen/panda power/pandapower-vpp-dso-sim
```

## 3. High-Level Existing Modules

| Path | Current role | Upgrade relevance |
|---|---|---|
| `src/vpp_dso_sim/envs/multi_agent_env.py` | Multi-agent DSO/VPP environment and DSO/VPP observations/actions | DSO observation and operating envelope routing |
| `src/vpp_dso_sim/simulation/simulator.py` | Main simulator, power-flow step, DSO envelope construction, logging | Existing `_build_dso_operating_envelope` adapter point |
| `src/vpp_dso_sim/optimization/feasibility_region.py` | Static feasible region / FR object logic | Hard bounds source for safe decoder |
| `src/vpp_dso_sim/optimization/ac_security_projection.py` | AC security validation/repair | Must remain post-dispatch safety layer |
| `src/vpp_dso_sim/network/sensitivity.py` | Existing sensitivity helper | Needs review before adding ActionUnit x NetworkObject tensor implementation |
| `src/vpp_dso_sim/envs/observations.py` | Actor and critic observation builders | Legacy flat/baseline observation preservation and structured DSO observation |
| `src/vpp_dso_sim/envs/reward_design.py` | Role reward maps | DSO width/projection/effective-response reward additions |
| `src/vpp_dso_sim/learning/deep_rl.py` | Privacy-separated CTDE actor-critic path | Legacy DSO MLP actor and observation path |
| `src/vpp_dso_sim/learning/advanced_trainers.py` | Advanced HAPPO/HASAC helper networks/losses | May need structured actor support |
| `src/vpp_dso_sim/learning/hatrpo.py` | HATRPO trainer | Baseline preservation; structured support if routed |
| `src/vpp_dso_sim/learning/matd3.py` | MATD3 trainer | Keep legacy route available |
| `src/vpp_dso_sim/experiments/paper_training.py` | Paper-long campaign runner | Experiment configs and output metrics path |
| `configs/*.yaml` | Compatibility wrappers for historical config paths | Keep old commands and manifests readable |
| `configs/scenarios/demo/` | Demo and smoke-test scenario configs | Canonical home for small scenarios |
| `configs/scenarios/benchmark/` | Benchmark and holdout scenario configs | Canonical home for benchmark feeder variants |
| `configs/algorithms/dso_sensitivity_attention/v1/` | Sensitivity-attention v1 baseline/new/ablation configs | Canonical algorithm config family |
| `configs/experiments/paper_long/sensitivity_attention_v1/` | Paper-long sensitivity-v1 experiment configs | Canonical paper-long config family |
| `configs/rewards/v2_minimal/` | Reusable reward-v2 config fragments | Canonical reward config family |
| `configs/registry.yaml` | Alias registry for `load_yaml()` | Prefer aliases in new scripts where helpful |
| `outputs/_archive/` | Inactive tests, audits, reports, and paper-long outputs | Active training outputs remain at root until complete |
| `outputs/_manifests/output_archive_manifest.csv` | Old-to-new output archive mapping | Use when tracing historical results |
| `tests/*.py` | Existing smoke, trainer, projection, privacy tests | Add Phase 2-11 tests |

## 4. Existing Tests Of Interest

| Test | What it protects |
|---|---|
| `tests/test_env_smoke.py` | single-agent Gym smoke |
| `tests/test_multi_agent_env.py` | multi-agent env behavior |
| `tests/test_ctde_interface.py` | CTDE actor/critic contract |
| `tests/test_privacy_observations.py` | actor observation privacy boundary |
| `tests/test_grid_aware_envelope.py` | grid-aware envelope behavior |
| `tests/test_ac_security_projection.py` | AC projection and repair |
| `tests/test_feasibility_region.py` | feasible region logic |
| `tests/test_deep_rl_training.py` | CTDE training artifacts and privacy |
| `tests/test_hasac_happo.py` | HAPPO/HASAC training artifacts |
| `tests/test_hatrpo_training.py` | HATRPO training artifacts |
| `tests/test_happo_hasac_trainers.py` | trainer loss functions |

## 5. Target New Module Layout

The prompt suggests `src/dso/...`; this repository uses package root `src/vpp_dso_sim`.

Adapted target layout:

```text
src/vpp_dso_sim/dso/envelope/rule_v0.py
src/vpp_dso_sim/dso/envelope/sensitivity_attention_v1.py
src/vpp_dso_sim/dso/envelope/safe_decoder.py
src/vpp_dso_sim/dso/envelope/schemas.py

src/vpp_dso_sim/dso/sensitivity/finite_difference.py
src/vpp_dso_sim/dso/sensitivity/cache.py
src/vpp_dso_sim/dso/sensitivity/selectors.py

src/vpp_dso_sim/dso/observation/legacy_flat.py
src/vpp_dso_sim/dso/observation/structured_bipartite.py

src/vpp_dso_sim/dso/models/legacy_mlp_actor.py
src/vpp_dso_sim/dso/models/bipartite_attention_actor.py
src/vpp_dso_sim/dso/models/structured_critic.py
```

## 6. Phase 0 Open Questions

1. Exact current `_build_dso_operating_envelope` fields and logging columns still need line-level inspection.
2. Existing `network/sensitivity.py` must be inspected for reuse versus wrapping.
3. Current trainer action dimensions must be inspected before wiring structured DSO actor.
4. Existing config loading should be inspected to decide where new `dso:` flags are parsed.

## 7. Dirty Worktree Warning

Before core implementation, the worktree already contained modifications in:

```text
src/vpp_dso_sim/experiments/paper_training.py
src/vpp_dso_sim/optimization/ac_security_projection.py
src/vpp_dso_sim/simulation/simulator.py
tests/test_ac_security_projection.py
tests/test_paper_training_experiment.py
```

Do not revert unrelated existing edits.
